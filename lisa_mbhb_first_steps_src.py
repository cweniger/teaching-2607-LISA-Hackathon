# %% [markdown]
# # Sequential SBI on LISA data — the zoom, live (bare-bones version)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_mbhb_first_steps.ipynb)
#
# The real dynamic-SBI loop on the 9-parameter LDC1-1 massive black-hole
# binary, with waveforms simulated live, round by round. This is the stripped
# version of the production pipeline (Lyu et al, 2026, in preparation): **no** Procrustes basis stabilisation,
# **no** Pareto-smoothed importance sampling, **no** late-time noise schedule,
# **no** EMA. The networks *are* warm-started — they keep learning across
# rounds, as in production.
#
# What you should see in 4 rounds: posterior widths falling (~25% in
# $\log_{10}D_L$ and $\cos\iota$) and the PCA spectrum collapsing toward a
# handful of informative components.
#
# Runtime: ~1 min per round on a Colab T4 (~10 s on the L4 this was developed
# on); the GPU does the flow training and density evaluation, the CPU the
# waveforms.
#
# Four rounds take ~1 min on a T4. The posteriors track the production run on
# the same source: sd(log10 D_L) 0.198 vs 0.191 at round 1, sd(cos iota) 0.579
# vs 0.540, sd(log10 Mc) 7e-4 -> 4e-4 over four rounds against 1.0e-3 -> 7e-4.
# The blue overlay in section 5 is that production posterior, shipped with the
# repo so you can see the gap without installing anything.

# %%
import importlib.util
import os
import subprocess
import sys
import time

if importlib.util.find_spec('lisabeta') is None:
    print('installing lisabeta (pre-built wheel, ~20 s) ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lisabeta'],
                   check=True)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import lisabeta.lisa.lisa as lisa
import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.lisa.pyLISAnoise as pyLISAnoise

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    torch.set_num_threads(min(8, os.cpu_count() or 1))
print(f'device: {dev}')

# %% [markdown]
# ## 1. The observation
#
# LDC1-1 (Radler) massive black-hole binary, one day at 10 s cadence, TDI A and
# E, whitened with the SciRD noise model. We infer **9 parameters**; the two
# gauge angles ($\varphi_0$, $\psi$) are randomised into every simulation and
# never inferred.
#
# The prior is the MCMC-narrowed box of the production campaign — distance,
# inclination and sky still wide open, chirp mass known to ~0.4%.
#
# Two settings below make this match the production analysis of the same
# source, and both matter:
#
# `TC_ABS` — the LISA constellation moves. lisabeta's `t0` says which point of
# the orbit the data window sits at, and leaving it at the default puts the
# constellation ~0.8 years away from where the LDC data actually is. That is
# not a normalisation: the antenna pattern changes, so the E channel comes out
# 4.8x too strong relative to A and everything lands 480 s late. Since cos(iota)
# is constrained through the *polarisation* content rather than the total
# amplitude, getting this wrong specifically distorts the distance-inclination
# posterior. We set t0 so that the window sits at the LDC coalescence time.
#
# `SIGNAL_SCALE` — with the orbit right, this stack still gives ~1.25x the
# production SNR (whitening-convention bookkeeping). One constant fixes it.
# Set both to their "naive" values (t0 absent, SIGNAL_SCALE = 1) and the SNR
# comes out 3x too high with the polarisations scrambled.

# %%
T_OBS, DT = 86400.0, 10.0
N_T = int(T_OBS / DT)
freqs = np.fft.rfftfreq(N_T, d=DT)
YEAR = 31558149.8
TC_ABS = 24_960_000.0                # LDC1-1 MBHB CoalescenceTime [s]
SIGNAL_SCALE = 0.815                 # -> production SNR (390) once t0 is set

wvf_pars = dict(minf=1e-5, maxf=0.1, timetomerger_max=1.0,
                TDI='TDIAET', acc=1e-4,
                approximant='IMRPhenomD',
                LISAconst=pyresponse.LISAconstProposal,
                responseapprox='full', frozenLISA=False, TDIrescaled=False)

psd = pyLISAnoise.evaluate_AET_psd(freqs[1:], TDIT=False,
                                   LISAnoise=pyLISAnoise.LISAnoiseSciRDv1,
                                   TDIrescaled=False)
S_A, S_E = np.asarray(psd['TDIA']), np.asarray(psd['TDIE'])
WHITE_A = 1.0 / np.sqrt(S_A * T_OBS / 4)
WHITE_E = 1.0 / np.sqrt(S_E * T_OBS / 4)


def whiten_td(h_fd, white):
    """FD -> whitened TD with unit noise variance per sample. lisabeta uses the
    e^{+2pi i f t} convention (opposite to numpy), so conjugate first."""
    w = np.zeros_like(h_fd)
    w[1:] = np.conj(h_fd[1:]) * white
    return np.fft.irfft(w, n=N_T) * np.sqrt(N_T / 2)


NAMES = ['log10_DL', 'cos_iota', 'log10_Mc', 'eta', 't_c_yrs',
         'lambda', 'sin_beta', 'a1', 'a2']
PRIOR_LO = np.array([4.0, -1.0, 6.186798, 0.212865, -0.000006,
                     3.219192, -0.076401, 0.660765, 0.344118])
PRIOR_HI = np.array([5.5, 1.0, 6.190189, 0.223907, 0.000009,
                     3.887023, 0.512674, 0.856587, 0.846804])
D = 9

Z_TRUE = np.array([4.74823, -0.33939, 6.18864, 0.21885, 0.0,
                   3.50910, 0.28853, 0.75348, 0.62159])
GAUGE_TRUE = (6.24790, 0.20445)

rng = np.random.default_rng(0)


def sim_one(z9, phi, psi):
    """One whitened, noise-free (A, E) time series, concatenated."""
    dl, ci, lmc, eta, tc, lam, sb, a1, a2 = z9
    Mc = 10 ** lmc
    Mtot = Mc / eta ** 0.6
    m1 = 0.5 * Mtot * (1 + np.sqrt(1 - 4 * eta))
    Deltat = T_OBS / 2 + tc * YEAR
    p = {'m1': float(m1), 'm2': float(Mtot - m1), 'chi1': float(a1),
         'chi2': float(a2), 'Deltat': float(Deltat), 'dist': float(10 ** dl),
         'inc': float(np.arccos(ci)), 'phi': float(phi), 'lambda': float(lam),
         'beta': float(np.arcsin(sb)), 'psi': float(psi)}
    # t0 places the data window on the real LISA orbit (see note in section 1)
    h = lisa.GenerateLISATDIFreqseries_SMBH(
        p, freqs, t0=float((TC_ABS - Deltat) / YEAR), **wvf_pars)[(2, 2)]
    return SIGNAL_SCALE * np.concatenate([whiten_td(h['chan1'], WHITE_A),
                                          whiten_td(h['chan2'], WHITE_E)])


def sim_batch(z9s, noise=True):
    """Noisy whitened data for a batch of parameters, gauge angles random."""
    out = np.empty((len(z9s), 2 * N_T), dtype=np.float32)
    for i, z in enumerate(z9s):
        out[i] = sim_one(z, rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
    if noise:
        out += rng.standard_normal(out.shape).astype(np.float32)
    return out


x_clean = sim_one(Z_TRUE, *GAUGE_TRUE).astype(np.float32)
x_obs = x_clean + np.random.default_rng(2607).standard_normal(
    2 * N_T).astype(np.float32)
print(f'observation built, clean A+E SNR = {np.sqrt((x_clean ** 2).sum()):.0f}')

def unwhiten_td(y_white, white):
    """Exact inverse of whiten_td: whitened TD -> raw (coloured) strain TD.
    Applied to the SAME realisation, so the two plots below show the same
    data before and after whitening."""
    Y = np.fft.rfft(y_white)
    Y[0] = 0.0
    Y[1:] = Y[1:] / (white * np.sqrt(N_T / 2))
    return np.fft.irfft(Y, n=N_T) / DT


tgrid = np.arange(N_T) * DT

# --- raw data: what the instrument actually delivers, noise-dominated
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(tgrid / 3600, unwhiten_td(x_obs[:N_T], WHITE_A), lw=.4,
        label='observed (A, raw)')
ax.plot(tgrid / 3600, unwhiten_td(x_clean[:N_T], WHITE_A), 'C1', lw=.8,
        label='hidden signal')
ax.legend(loc='upper left', fontsize=8)
ax.set(xlabel='t [hours]', ylabel='strain (TDI A)',
       title='before whitening: raw TDI A strain (merger visible, inspiral buried)')
fig.tight_layout()

# --- after whitening: unit-variance noise, signal visible at merger
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(tgrid / 3600, x_obs[:N_T], lw=.4, label='observed (A, whitened)')
ax.plot(tgrid / 3600, x_clean[:N_T], 'C1', lw=.8, label='hidden signal')
ax.legend(loc='upper left', fontsize=8)
ax.set(xlabel='t [hours]', ylabel='whitened amplitude',
       title='after whitening: this is what the network sees')
fig.tight_layout()

# %% [markdown]
# ## 2. The inference machinery
#
# Flow matching in ten lines, unchanged from the main tutorial: a velocity
# field $v(w,t,s)$ trained to transport noise to samples, an Euler sampler, and
# an exact-divergence log-density via the continuity equation.

# %%
def mlp(d_in, d_out, hidden, layers):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, d_out))


def fm_loss(net, w1, cond):
    w0 = torch.randn_like(w1)
    t = torch.rand(len(w1), 1, device=w1.device)
    wt = (1 - t) * w0 + t * w1
    v = net(wt, t, cond)
    return ((v - (w1 - w0)) ** 2).mean()


class VelocityNet(nn.Module):
    def __init__(self, d_w, d_cond, hidden=256, layers=4):
        super().__init__()
        self.freqs = torch.tensor([1., 2., 4., 8.])
        self.net = mlp(d_w + 9 + d_cond, d_w, hidden, layers)

    def forward(self, w, t, cond):
        ft = 2 * np.pi * t * self.freqs.to(t.device)
        temb = torch.cat([t, ft.sin(), ft.cos()], 1)
        return self.net(torch.cat([w, temb, cond], 1))


@torch.no_grad()
def fm_sample(net, cond, d_w, steps=64):
    w = torch.randn(len(cond), d_w, device=cond.device)
    for i in range(steps):
        t = torch.full((len(cond), 1), (i + 0.5) / steps, device=cond.device)
        w = w + net(w, t, cond) / steps
    return w


def fm_logprob(net, w1, cond, steps=32):
    w = w1.clone()
    logdet = torch.zeros(len(w1), device=w1.device)
    for i in range(steps):
        t = torch.full((len(w1), 1), 1 - (i + 0.5) / steps, device=w1.device)
        with torch.enable_grad():
            wg = w.requires_grad_(True)
            v = net(wg, t, cond)
            div = sum(torch.autograd.grad(v[:, d].sum(), wg,
                                          retain_graph=(d < w1.shape[1] - 1))[0][:, d]
                      for d in range(w1.shape[1]))
        w = (w - v / steps).detach()
        logdet = logdet - div.detach() / steps
    base = -0.5 * (w ** 2).sum(1) - 0.5 * w.shape[1] * np.log(2 * np.pi)
    return base + logdet


def zscore(a, mean, std):
    return (a - mean) / std


# --- prior <-> latent -------------------------------------------------------
# The flows work on a STANDARD-NORMAL latent u, not on theta directly: the
# uniform prior box maps to N(0,I) through the probit transform. This is the
# single most important structural choice here. Asking a CNF to model a 9-D
# uniform box means asking it to reproduce hard cliffs at every face, which it
# cannot do; log q is then wrong exactly where the importance weights are
# largest. In latent space the round-1 buffer IS N(0,I) -- smooth, trivial to
# fit -- and the prior term in the weights is just -|u|^2/2. Measured effect on
# the round-1 proposal ESS: 2% (box) vs >50% (latent).
SQRT2 = float(np.sqrt(2.0))


def to_lat(th, lo, hi):
    u = ((th - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
    return SQRT2 * torch.erfinv(2 * u - 1)


def from_lat(u, lo, hi):
    return lo + (hi - lo) * 0.5 * (1 + torch.erf(u / SQRT2))


def log_prior_lat(u):
    return -0.5 * (u ** 2).sum(1) - 0.5 * u.shape[1] * np.log(2 * np.pi)

# %% [markdown]
# ## 3. Compression: PCA refit every round
#
# 17280 samples -> 64 PCA coefficients, refit each round because zooming
# concentrates the signal variance in ever fewer components. The SVD goes
# through the Gram matrix of the (much smaller) simulation batch.
#
# The production pipeline additionally Procrustes-aligns consecutive bases and
# Wiener-weights the coefficients. Neither is needed here: we refit the
# networks from scratch each round, so nothing has to stay in a stable frame.

# %%
K = 64


def fit_pca(z9s):
    """Clean sims at z9s -> (mean, top-K basis, per-component SNR)."""
    X = np.empty((len(z9s), 2 * N_T), dtype=np.float32)
    for i, z in enumerate(z9s):
        X[i] = sim_one(z, rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
    mu = X.mean(0)
    Xc = torch.from_numpy(X - mu)
    C = (Xc @ Xc.T).double() / (len(z9s) - 1)
    evals, U = torch.linalg.eigh(C)
    evals, U = evals.flip(0), U.flip(1)
    eigs = evals.clamp(min=0).sqrt()
    V = (Xc.T.double() @ U[:, :K]) / (eigs[:K] * np.sqrt(len(z9s) - 1))
    return mu, V.T.float().numpy(), eigs.float().numpy()

# %% [markdown]
# ## 4. The sequential zoom
#
# Per round:
# 1. refit the PCA basis and the z-score gauges on the current buffer,
# 2. keep training the conditional $q_c(\theta|s)$ and marginal $q_m(\theta)$
#    flows (warm-started: the same two networks, every round),
# 3. build one weighted pool from a mixture of $q_c$, $q_m$ **and the prior** —
#    used twice, at two different temperatures:
#    * $\gamma = 0.3$ picks which parameters to **simulate next** (a deliberately
#      broad target, so the buffer keeps covering more than the posterior),
#    * $\gamma = 1$ gives the **posterior readout** you plot.
#
# That second point is the one subtlety worth keeping. Sampling $q_c$ directly
# is tempting and wrong: $q_c$ learns the posterior *of the current buffer*,
# and once the buffer is itself tempered, $q_c \propto \pi L^{1+\gamma}$ —
# over-sharp. Measured on the production run: identical at round 1, but 18% too
# narrow in $\log_{10}D_L$ by round 4. The importance-weighted readout costs
# five lines and fixes it.

# %%
N_ROUNDS = 4          # <-- the knob for Exercise 2
GAMMA = 0.3           # collection temperature (production value)
N_BUF, N_KEEP = 8000, 2000        # train buffer, and fresh sims per round (25%)
N_VAL, N_KEEP_VAL = 1000, 250     # held-out val buffer, same 25% turnover
N_PCA, N_POOL = 1024, 8192        # PCA fit batch, weighted-pool size
N_STEPS_MAX = 12000               # training steps per net per round (a CAP:
EVAL_EVERY, PATIENCE = 250, 6     # early stopping on the val loss usually
                                  # stops well before it)

LO_T = torch.tensor(PRIOR_LO, dtype=torch.float32, device=dev)
HI_T = torch.tensor(PRIOR_HI, dtype=torch.float32, device=dev)

t0 = time.time()
buf_theta = PRIOR_LO + (PRIOR_HI - PRIOR_LO) * rng.uniform(size=(N_BUF, D))
buf_x = sim_batch(buf_theta)
# A SEPARATE validation buffer, never trained on, refreshed at the same rate as
# the train buffer so the two always describe the same distribution. Without it
# there is nothing to early-stop on, and the flows quietly overfit: more
# training then makes the posterior WORSE, which is impossible to notice from
# the training loss alone.
val_theta = PRIOR_LO + (PRIOR_HI - PRIOR_LO) * rng.uniform(size=(N_VAL, D))
val_x = sim_batch(val_theta)
print(f'initial buffers: {N_BUF} train + {N_VAL} val live sims '
      f'in {time.time() - t0:.0f} s')

# The networks are created ONCE and keep learning across rounds (warm start).
# Training from scratch each round throws away everything the previous round
# learned and makes every round return the same resolution, so nothing appears
# to improve; the zoom you see below is the buffer contracting, on top of nets
# that keep getting better.
qc, qm = VelocityNet(D, K).to(dev), VelocityNet(D, K).to(dev)
opt_c = torch.optim.Adam(qc.parameters(), lr=1e-3)
opt_m = torch.optim.Adam(qm.parameters(), lr=1e-3)

posts, spectra = [], []
for rnd in range(1, N_ROUNDS + 1):
    t_r = time.time()
    # -- 1. gauges: PCA refit on the current buffer + z-scores
    idx = rng.choice(len(buf_theta), N_PCA, replace=False)
    mu, V, eigs = fit_pca(buf_theta[idx])
    spectra.append(eigs[:200].copy())
    # WIENER weighting, not a z-score. Each PCA coefficient carries signal
    # variance eigs^2 and (because the basis rows are orthonormal) noise
    # variance 1, so the MMSE weight is
    #     lam/(lam + sigma^2)/sqrt(lam)  with lam = eigs^2, sigma^2 = 1
    #   = eigs/(eigs^2 + 1)
    # For eigs >> 1 this normalises an informative component to unit signal
    # variance; for eigs << 1 it SUPPRESSES it. A z-score would instead inflate
    # every noise-dominated component back to unit variance and hand the
    # network ~50 channels of pure noise.
    wien = (eigs[:K] / (eigs[:K] ** 2 + 1.0)).astype(np.float32)

    def summarize(x):                       # (..., 2*N_T) -> (..., K)
        return ((x - mu) @ V.T) * wien

    s_t = torch.from_numpy(summarize(buf_x).astype(np.float32)).to(dev)
    s_v = torch.from_numpy(summarize(val_x).astype(np.float32)).to(dev)
    th_t = torch.from_numpy(buf_theta.astype(np.float32)).to(dev)
    th_v = torch.from_numpy(val_theta.astype(np.float32)).to(dev)
    w1, sc = to_lat(th_t, LO_T, HI_T), s_t
    w1v, scv = to_lat(th_v, LO_T, HI_T), s_v
    so = torch.from_numpy(summarize(x_obs[None]).astype(np.float32)).to(dev)

    # -- 2. keep training the SAME two networks (warm start), early-stopping on
    #       the held-out val buffer. The val loss uses a FIXED (t, w0) draw so
    #       it is deterministic given the network -- otherwise its own Monte
    #       Carlo noise swamps the improvement we are trying to detect.
    gv = torch.Generator().manual_seed(1234)
    t_val = torch.rand(len(w1v), 1, generator=gv).to(dev)
    w0_val = torch.randn(len(w1v), D, generator=gv).to(dev)
    wt_val = (1 - t_val) * w0_val + t_val * w1v
    tgt_val = w1v - w0_val

    def val_loss(net, condv):
        with torch.no_grad():
            return float(((net(wt_val, t_val, condv) - tgt_val) ** 2).mean())

    used = []
    for net, opt, cond, condv in [(qc, opt_c, sc, scv),
                                  (qm, opt_m, torch.zeros_like(sc),
                                   torch.zeros_like(scv))]:
        best, bad, step = float('inf'), 0, 0
        while step < N_STEPS_MAX:
            step += 1
            i = torch.randint(0, len(w1), (256,), device=dev)
            loss = fm_loss(net, w1[i], cond[i])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % EVAL_EVERY == 0:
                v = val_loss(net, condv)
                if v < best - 1e-4:
                    best, bad = v, 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        used.append(step)

    # -- 3. ONE mixture pool, log-weights at any temperature.
    #    THREE components: q_c, q_m, and the prior itself.  The prior draws are
    #    what make this stable: once the buffer contracts, q_m is tiny over most
    #    of the (still uniform) prior box, so a q_c+q_m proposal misses the very
    #    region the gamma=1 target lives in and the weights degenerate to a
    #    handful of samples.  Measured without it: readout ESS 2521 -> 19 by
    #    round 2.  A third of the pool drawn from the prior fixes that for the
    #    price of three lines.
    n3 = N_POOL // 3
    wp = torch.cat([fm_sample(qc, so.expand(n3, K), D),
                    fm_sample(qm, torch.zeros(n3, K, device=dev), D),
                    torch.randn(N_POOL - 2 * n3, D, device=dev)])  # <- the prior
    lqc = fm_logprob(qc, wp, so.expand(len(wp), K))
    lqm = fm_logprob(qm, wp, torch.zeros(len(wp), K, device=dev))
    lpi = log_prior_lat(wp)
    th_p = from_lat(wp, LO_T, HI_T)      # always inside the box: no mask needed

    def logw_at(gamma):
        lmix = torch.logsumexp(torch.stack([lqc, lqm, lpi]), 0) - np.log(3.0)
        lw = gamma * (lqc - lqm) + lpi - lmix
        return torch.where(torch.isfinite(lw), lw, torch.full_like(lw, -torch.inf))

    # 3a. collection at gamma=0.3: keep N_KEEP without replacement, simulate LIVE
    lw_c = logw_at(GAMMA)
    ess = float(torch.exp(2 * torch.logsumexp(lw_c, 0) - torch.logsumexp(2 * lw_c, 0)))
    gum = -torch.log(-torch.log(torch.rand_like(lw_c)))
    keep = torch.topk(lw_c + gum, N_KEEP + N_KEEP_VAL).indices
    new_theta = th_p[keep].cpu().numpy().astype(np.float64)
    new_x = sim_batch(new_theta)
    # split the fresh draws RANDOMLY between the two buffers: topk returns them
    # weight-sorted, so slicing head/tail would bias one buffer
    perm = rng.permutation(len(new_theta))
    iv, itr = perm[:N_KEEP_VAL], perm[N_KEEP_VAL:]
    buf_theta = np.concatenate([new_theta[itr], buf_theta])[:N_BUF]
    buf_x = np.concatenate([new_x[itr], buf_x])[:N_BUF]
    val_theta = np.concatenate([new_theta[iv], val_theta])[:N_VAL]
    val_x = np.concatenate([new_x[iv], val_x])[:N_VAL]

    # 3b. posterior readout at gamma=1 (resample the SAME pool with replacement)
    lw_p = logw_at(1.0)
    w_p = torch.exp(lw_p - torch.logsumexp(lw_p, 0))
    ess_p = float(1.0 / (w_p ** 2).sum())
    sel = torch.multinomial(w_p, 4000, replacement=True)
    posts.append(th_p[sel].cpu().numpy())

    print(f'round {rnd} [{time.time() - t_r:4.0f} s]: steps {used[0]}/{used[1]} '
          f'(early-stopped), collect ESS {ess:5.0f}/{N_POOL}, '
          f'readout ESS {ess_p:5.0f}/{N_POOL}, '
          f'PCA comps > noise {(eigs > 1).sum():3d}, '
          f'sd(log10_Mc) {posts[-1][:, 2].std():.1e}, '
          f'sd(log10_DL) {posts[-1][:, 0].std():.3f}')
print(f'total {time.time() - t0:.0f} s')

# %% [markdown]
# ## 5. What happened

# %%
def corner(samples, labels, truth, names, figsize=13, colors=None):
    n = samples[0].shape[1]
    fig, ax = plt.subplots(n, n, figsize=(figsize, figsize))
    cols = colors or plt.cm.viridis(np.linspace(0, .85, len(samples)))
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if j > i:
                a.axis('off'); continue
            for z, c in zip(samples, cols):
                if i == j:
                    a.hist(z[:, i], bins=40, histtype='step', color=c, density=True)
                else:
                    a.plot(z[:, j], z[:, i], '.', ms=1, alpha=.15, color=c)
            if i == j:
                a.axvline(truth[i], color='r', lw=.8); a.set_yticks([])
            else:
                a.axvline(truth[j], color='r', lw=.6)
                a.axhline(truth[i], color='r', lw=.6)
            if i == n - 1: a.set_xlabel(names[j], fontsize=7)
            else: a.set_xticklabels([])
            if j == 0 and i > 0: a.set_ylabel(names[i], fontsize=7)
            elif j != 0: a.set_yticklabels([])
            a.tick_params(labelsize=5)
    h = [plt.Line2D([], [], color=c, lw=2, label=l)
         for l, c in zip(labels, cols)]
    h += [plt.Line2D([], [], color='r', lw=1, label='injected truth')]
    fig.legend(handles=h, loc='upper right', fontsize=11, frameon=False,
               bbox_to_anchor=(.97, .97))
    fig.tight_layout(rect=[0, 0, 1, .97])
    return fig


# Overlay: the production pipeline on the SAME source and prior -- 60 rounds,
# ~750k live simulations, ~1.6 h on an L4, with all the machinery listed at the
# bottom of this notebook. Shipped as a 5000-sample file so you can see how far
# four bare-bones rounds get without installing anything.
try:
    prod = np.load('production_posterior.npy')
except FileNotFoundError:      # colab: fetch from the repo
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/cweniger/'
        'teaching-2607-LISA-Hackathon/main/production_posterior.npy',
        'production_posterior.npy')
    prod = np.load('production_posterior.npy')

show = [0, len(posts) - 1] if len(posts) > 1 else [0]
corner([posts[i] for i in show] + [prod],
       [f'round {i + 1}' for i in show] + ['production, 60 rounds'],
       Z_TRUE, NAMES, colors=['#E69F00', '#D55E00', '#0072B2'][-(len(show) + 1):])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cols = plt.cm.viridis(np.linspace(0, .85, len(posts)))
for r, (p, c) in enumerate(zip(posts, cols), 1):
    ax[0].plot(p[:, 0], p[:, 1], '.', ms=1.5, alpha=.2, color=c, label=f'round {r}')
ax[0].plot(Z_TRUE[0], Z_TRUE[1], 'r*', ms=14)
ax[0].set(xlabel=r'$\log_{10} D_L$', ylabel=r'$\cos\iota$', xlim=(4.3, 5.4),
          ylim=(-1, 1), title='distance-inclination: the slow direction')
ax[0].legend(markerscale=8, fontsize=8)
for r, (e, c) in enumerate(zip(spectra, cols), 1):
    ax[1].semilogy(e, color=c, label=f'round {r}')
ax[1].axhline(1, color='r', ls='--', lw=1)
ax[1].set(xlabel='PCA component', ylabel='component SNR',
          title='compression gets easier as the zoom tightens')
ax[1].legend(fontsize=8)
fig.tight_layout()

widths = np.array([p.std(0) for p in posts])
print('posterior width per round:')
print('   ' + '  '.join(f'{n:>9s}' for n in NAMES))
for r, w in enumerate(widths, 1):
    print(f'r{r} ' + '  '.join(f'{v:9.4f}' for v in w))

# %% [markdown]
# Two things to read off:
#
# 1. **Widths fall round by round** — same networks size, same per-round
#    simulation budget; only the *training distribution* moved. Chirp mass,
#    $\eta$ and the spins tighten fastest.
# 2. **Compression gets easier** — the number of PCA components above the noise
#    line drops as the buffer contracts, which is why refitting the summaries
#    every round matters.
#
# And one thing that should *not* have converged: $\cos\iota$ (and with it
# $D_L$). The distance-inclination degeneracy is the last direction to resolve;
# in the production run it needs tens of rounds. If it looks converged after
# four, be suspicious of the readout, not pleased with the result.
#
# **Exercise 1.** Truth recovery: `(posts[-1].mean(0) - Z_TRUE) / posts[-1].std(0)`.
# Which parameters sit within 1 sigma? Re-run the observation cell with a
# different noise seed and watch these z-scores scatter.
#
# **Exercise 2.** The knobs:
# 1. `GAMMA = 0.1` vs `1.0` — which zooms faster, which is riskier? Watch the
#    collection ESS.
# 2. `N_ROUNDS = 8` — do the widths keep falling? Does the readout ESS hold up?
# 3. Replace the $\gamma=1$ readout with `fm_sample(qc, so.expand(4000, K), D)`
#    and compare widths at round 1 versus round 4. You should reproduce the
#    over-sharpening described in section 4.
# 4. Replace the Wiener weight with a plain z-score
#    (`wien = 1/(s_t.std(0) + 1e-6)`) and watch what feeding the network ~50
#    channels of pure noise does to $\cos\iota$.
# 5. Set `PATIENCE = 1000` to disable early stopping. Training runs to the
#    `N_STEPS_MAX` cap, the flows overfit, and the widths get *worse* round over
#    round — the failure the validation buffer exists to catch.
#
# **Exercise 3** *(discussion)*. Watch the readout ESS fall as the target
# sharpens — by round 4 it is already an order of magnitude below round 1. In
# production the pool is 65536 rather than 8192, the networks are warm-started
# with an EMA, the PCA frame is Procrustes-stabilised, and the final readout is
# Pareto-smoothed. That last one is not optional at scale: without it the
# weights eventually collapse onto a *single* sample out of 100000, because two
# draws in that pool get a log-density ~40 nats below the rest. What you ran
# here is the same algorithm with the safety rails removed — fine for four
# rounds, not fine for fifty.
#
# ---
#
# ## 6. What a production pipeline needs on top
#
# *(work in progress — more soon)*
#
# The blue overlay above is the same algorithm run for 60 rounds and ~750k live
# simulations. Everything between it and what you just ran is **robustness, not
# ideas**: none of it changes the answer at four rounds, and all of it exists
# because something broke without it. The honest list:
#
# **Summaries**
# - *Procrustes alignment* of consecutive PCA bases, so a warm-started network
#   does not see its input frame rotate underneath it every round. (Wiener
#   weighting and the held-out validation buffer used to be on this list — both
#   are now implemented above, because they are cheap and change the answer.)
#
# **Training**
# - *EMA* over the weights, and keeping the best network rather than the latest —
#   a single diverged network otherwise poisons the buffer for every later round.
#   In one production run a network blew up to `nll = 1.7e19`, was accepted as
#   "best", and destroyed the readout.
#
# **The readout**
# - *Pareto-smoothed importance sampling* with the $\hat k$ diagnostic. Raw
#   self-normalised IS silently collapsed to ESS = 1 out of 100000 in production
#   runs; PSIS recovers ~27000 and $\hat k > 0.7$ tells you when not to trust it.
# - Enough *ODE steps* in the density: 32 steps was under-resolved and inflated
#   the reported ESS by an order of magnitude.
#
# **The loop**
# - *Adaptive buffer growth* (double when the validation NLL stalls) and a
#   principled *stopping rule*, instead of a fixed `N_ROUNDS`.
# - An occasional *exact-likelihood probe* — the one diagnostic the flows cannot
#   flatter, because the simulator does not know what the proposal is.
#
# **And the part that is not machinery at all:** getting the simulator right.
# The single largest error found while building this notebook was not in the
# inference — it was forgetting to tell lisabeta where LISA is in its orbit,
# which scrambled the polarisation content and silently distorted the
# distance–inclination posterior by 6x.
#
# ---
# *Scope: this starts from the MCMC-narrowed prior box and demonstrates the zoom
# mechanism, not a search from the full prior (that is a production-scale run;
# see arXiv:2510.13997).*

# %%
# (housekeeping — saves figures when run as a test script; no-op in colab)
if os.environ.get('TUTORIAL_SAVE_FIGS'):
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'seqnew_fig_{i:02d}.png', dpi=110,
                              bbox_inches='tight')
    print(f'saved {len(plt.get_fignums())} figures')
