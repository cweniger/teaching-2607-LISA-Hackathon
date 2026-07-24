# %% [markdown]
# # Sequential SBI on LISA data — the zoom, live
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_sequential.ipynb)
#
# The main tutorial (`tutorial_lisa_sbi.ipynb`) demonstrated the sequential
# zoom on a toy chirp and stopped at *amortized* inference for the massive
# black-hole binary, with sequential MBHB results shown on slides only. The
# reason was cost — and it evaporated: the lisabeta waveform stack simulates
# this source in ~1.5 ms, fast enough to run the **actual dynamic-SBI loop
# live**, simulating where the current posterior points, rung by rung.
#
# This notebook does exactly that:
# 1. build the LISA observation with lisabeta (as in `lisa_sims.ipynb`),
# 2. define the *same* ten-line `fm_loss` and `VelocityNet` from the tutorial,
# 3. run 4 rungs of the sequential zoom on the real 9-parameter MBHB problem —
#    PCA summaries refit every rung, ~2000 fresh simulations per rung,
#    tempered importance reweighting in between.
#
# Runtime: ~3 min on a few CPU cores; a GPU accelerates the training parts.
# Honest scope: we start from the MCMC-narrowed prior box of the production
# campaign and demonstrate the *last* zoom steps and the mechanism — not a
# search from the full prior (that is a production-scale run; see the Dynamic
# SBI paper, arXiv:2510.13997).

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
# Same source and conventions as `lisa_sims.ipynb`: the LDC1-1 (Radler)
# massive black-hole binary, one day of data at 10 s cadence, TDI A and E
# channels, whitened with the SciRD noise model. We infer **9 parameters**;
# the two gauge angles ($\varphi_0$, $\psi$) are randomized into the
# simulations and never inferred — the treatment the tutorial's toy chirp
# introduced for its phase.
#
# The prior is the MCMC-narrowed box of the production campaign: distance,
# inclination and sky are still wide open, the chirp mass is known to ~0.4%.
#
# > A note on numbers: in this simplified pipeline the source has SNR ≈ 1170.
# > The production analysis, whose conventions are carefully matched to the
# > LDC data, measures SNR ≈ 393 for the same extract; the difference is a
# > roughly constant waveform-normalization factor between the two stacks.
# > Posteriors here are correspondingly tighter than the production ones —
# > fine for demonstrating the mechanism, not for quoting physics.

# %%
T_OBS, DT = 86400.0, 10.0
N_T = int(T_OBS / DT)
freqs = np.fft.rfftfreq(N_T, d=DT)
YEAR = 31558149.8

wvf_pars = dict(minf=1e-5, maxf=0.1, timetomerger_max=(T_OBS / 2) / YEAR,
                tmax=T_OBS / YEAR, TDI='TDIAET', acc=1e-4,
                approximant='IMRPhenomD',
                LISAconst=pyresponse.LISAconstProposal,
                responseapprox='full', frozenLISA=False, TDIrescaled=False)

psd = pyLISAnoise.evaluate_AET_psd(freqs[1:], TDIT=False,
                                   LISAnoise=pyLISAnoise.LISAnoiseSciRDv1,
                                   TDIrescaled=False)   # {'freq', 'TDIA', 'TDIE'}
S_A, S_E = np.asarray(psd['TDIA']), np.asarray(psd['TDIE'])
WHITE_A = 1.0 / np.sqrt(S_A * T_OBS / 4)
WHITE_E = 1.0 / np.sqrt(S_E * T_OBS / 4)


def whiten_td(h_fd, white):
    """FD -> whitened TD with unit noise variance per sample. lisabeta uses
    the e^{+2pi i f t} Fourier convention (opposite to numpy), so conjugate
    before the inverse FFT, else time runs backwards."""
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
GAUGE_TRUE = (6.24790, 0.20445)                 # phi0, psi

rng = np.random.default_rng(0)


def sim_one(z9, phi, psi):
    """One whitened, noise-free (A, E) time series, concatenated."""
    dl, ci, lmc, eta, tc, lam, sb, a1, a2 = z9
    Mc = 10 ** lmc
    Mtot = Mc / eta ** 0.6
    m1 = 0.5 * Mtot * (1 + np.sqrt(1 - 4 * eta))
    p = {'m1': m1, 'm2': Mtot - m1, 'chi1': a1, 'chi2': a2,
         'Deltat': T_OBS / 2 + tc * YEAR, 'dist': 10 ** dl,
         'inc': np.arccos(ci), 'phi': phi, 'lambda': lam,
         'beta': np.arcsin(sb), 'psi': psi}
    h = lisa.GenerateLISATDIFreqseries_SMBH(p, freqs, **wvf_pars)[(2, 2)]
    return np.concatenate([whiten_td(h['chan1'], WHITE_A),
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
print(f'observation built, clean A+E SNR = {np.sqrt((x_clean ** 2).sum()):.0f} '
      f'(production conventions: ~393)')

tgrid = np.arange(N_T) * DT
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(tgrid / 3600, x_obs[:N_T], lw=.4, label='observed (A, whitened)')
ax.plot(tgrid / 3600, x_clean[:N_T], 'C1', lw=.8, label='hidden signal')
ax.legend(loc='upper left', fontsize=8)
ax.set(xlabel='t [hours]', ylabel='whitened amplitude')
fig.tight_layout()

# %% [markdown]
# ## 2. The inference machinery — identical to the tutorial
#
# `fm_loss`, `VelocityNet`, `fm_sample`, `fm_logprob`: copied from the main
# tutorial (Parts 2–3). The only change: `fm_logprob`'s divergence loop keeps
# the autograd graph until the *last* dimension (`d < D-1`) — the tutorial's
# 2-D version could hard-code the first.

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

# %% [markdown]
# ## 3. Compression: PCA refit on every rung
#
# As in the tutorial's toy chirp, we compress the 17280-sample data vector to
# 64 PCA coefficients — and refit the basis on every rung, because zooming
# concentrates the signal variance in ever fewer components. With 17280
# dimensions the SVD is computed via the Gram matrix of the (much smaller)
# simulation batch.

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
# The loop from the tutorial's Part 3, on the real problem. Per rung:
# 1. refit the PCA **gauges** on the current buffer,
# 2. continue training the conditional flow $q_c(\theta|s)$ and the marginal
#    flow $q_m(\theta)$ (warm-started — the same networks keep learning),
# 3. propose from a 50/50 mixture of both flows, importance-weight toward the
#    tempered target $L^\gamma \pi$ using `fm_logprob`,
# 4. keep the best 2048 (Gumbel top-k), **simulate them live**, refresh the
#    buffer.
#
# Watch the printed diagnostics: posterior width falls, the number of PCA
# components above the noise drops, and the proposal effective sample size
# (ESS) tells you how well the flows track the target.

# %%
N_RUNGS = 4          # <-- the knobs for Exercise 2
GAMMA = 0.5
N_KEEP = 2048
REFIT_PCA = True

N_BUF, N_PCA, N_PROP = 4096, 1024, 4096
LO_T = torch.tensor(PRIOR_LO, dtype=torch.float32, device=dev)
HI_T = torch.tensor(PRIOR_HI, dtype=torch.float32, device=dev)

t0 = time.time()
buf_theta = PRIOR_LO + (PRIOR_HI - PRIOR_LO) * rng.uniform(size=(N_BUF, D))
buf_x = sim_batch(buf_theta)
print(f'initial buffer: {N_BUF} live sims in {time.time() - t0:.0f} s')

qc, qm = VelocityNet(D, K).to(dev), VelocityNet(D, K).to(dev)
opt_c = torch.optim.Adam(qc.parameters(), lr=1e-3)
opt_m = torch.optim.Adam(qm.parameters(), lr=1e-3)

posts, spectra = [], []
mu = V = eigs = None
for rung in range(1, N_RUNGS + 1):
    t_r = time.time()
    # -- gauges: PCA refit on the current buffer + z-scores
    if REFIT_PCA or mu is None:
        idx = rng.choice(len(buf_theta), N_PCA, replace=False)
        mu, V, eigs = fit_pca(buf_theta[idx])
    spectra.append(eigs[:200].copy())
    s_t = torch.from_numpy(((buf_x - mu) @ V.T).astype(np.float32)).to(dev)
    th_t = torch.from_numpy(buf_theta.astype(np.float32)).to(dev)
    smu, ssd = s_t.mean(0), s_t.std(0) + 1e-6
    tmu, tsd = th_t.mean(0), th_t.std(0) + 1e-9
    w1 = zscore(th_t, tmu, tsd)
    sc = zscore(s_t, smu, ssd)
    so = zscore(torch.from_numpy(((x_obs - mu) @ V.T).astype(np.float32)
                                 ).to(dev)[None], smu, ssd)
    # -- continue training conditional and marginal flows
    n_steps = 1500 if rung == 1 else 800
    for net, opt, cond in [(qc, opt_c, sc), (qm, opt_m, torch.zeros_like(sc))]:
        for step in range(n_steps):
            i = torch.randint(0, len(w1), (256,), device=dev)
            loss = fm_loss(net, w1[i], cond[i])
            opt.zero_grad(); loss.backward(); opt.step()
    # -- propose from the mixture, weight toward L^gamma * prior
    wp = torch.cat([fm_sample(qc, so.expand(N_PROP // 2, K), D),
                    fm_sample(qm, torch.zeros(N_PROP // 2, K, device=dev), D)])
    lqc = fm_logprob(qc, wp, so.expand(N_PROP, K))
    lqm = fm_logprob(qm, wp, torch.zeros(N_PROP, K, device=dev))
    th_p = wp * tsd + tmu
    in_prior = ((th_p > LO_T) & (th_p < HI_T)).all(1)
    logw = GAMMA * (lqc - lqm) - torch.logaddexp(lqc, lqm)   # + log(flat prior)
    logw[~in_prior] = -torch.inf
    logw[~torch.isfinite(logw)] = -torch.inf
    ess = float(torch.exp(2 * torch.logsumexp(logw, 0)
                          - torch.logsumexp(2 * logw, 0)))
    # -- keep N_KEEP without replacement (Gumbel top-k), simulate LIVE
    gum = -torch.log(-torch.log(torch.rand_like(logw)))
    keep = torch.topk(logw + gum, N_KEEP).indices
    new_theta = th_p[keep].cpu().numpy().astype(np.float64)
    new_x = sim_batch(new_theta)
    buf_theta = np.concatenate([new_theta, buf_theta])[:N_BUF]
    buf_x = np.concatenate([new_x, buf_x])[:N_BUF]
    # -- posterior readout for the plot
    post = (fm_sample(qc, so.expand(4000, K), D) * tsd + tmu).cpu().numpy()
    posts.append(post)
    print(f'rung {rung} [{time.time() - t_r:4.0f} s]: '
          f'ESS {ess:4.0f}/{N_PROP}, PCA comps > noise: {(eigs > 1).sum():3d}, '
          f'std(log10_Mc) {post[:, 2].std():.1e}, '
          f'std(t_c) {post[:, 4].std() * YEAR:.0f} s')
print(f'total {time.time() - t0:.0f} s')

# %% [markdown]
# ## 5. What happened

# %%
colors = plt.cm.viridis(np.linspace(0, .9, len(posts)))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for r, (post, c) in enumerate(zip(posts, colors), 1):
    ax[0].plot(post[:, 2], post[:, 3], '.', ms=1.5, alpha=.2, color=c,
               label=f'rung {r}')
    ax[1].plot(post[:, 0], post[:, 1], '.', ms=1.5, alpha=.2, color=c)
ax[0].plot(Z_TRUE[2], Z_TRUE[3], 'r*', ms=14)
ax[1].plot(Z_TRUE[0], Z_TRUE[1], 'r*', ms=14)
ax[0].set(xlabel=r'$\log_{10} M_c$', ylabel=r'$\eta$',
          title='the zoom, rung by rung (star = truth)')
ax[1].set(xlabel=r'$\log_{10} D_L$', ylabel=r'$\cos\iota$',
          xlim=(4.3, 5.2), ylim=(-1, 1), title='distance–inclination')
ax[0].legend(markerscale=8, fontsize=8)
for r, (e, c) in enumerate(zip(spectra, colors), 1):
    ax[2].semilogy(e, color=c, label=f'rung {r}')
ax[2].axhline(1, color='r', ls='--', lw=1)
ax[2].set(xlabel='PCA component', ylabel='component SNR',
          title='compression gets easier as the zoom tightens')
ax[2].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# The same two things the toy chirp showed, now on a real gravitational wave
# with live simulations:
# 1. **the posterior tightens rung by rung** — same networks, same per-rung
#    simulation budget, only the *training distribution* moved;
# 2. **compression gets easier** — the number of PCA components above the
#    noise line drops as the buffer contracts, which is why refitting the
#    summaries (step 1 of every rung) matters.
#
# Every simulation entering the buffer after rung 1 was chosen by the current
# posterior estimate and generated by the C waveform code *while you watched*
# — this is the dynamic-SBI mechanism, not a staged re-enactment.
#
# **Exercise 1.** Truth recovery: compute
# `(post.mean(0) - Z_TRUE) / post.std(0)` for the last rung. Which parameters
# are within 1σ? Rerun the observation cell with a different noise seed and
# check how these z-scores scatter.
#
# **Exercise 2.** The knobs:
# 1. `GAMMA = 0.1` vs `1.0` — which zooms faster, which is riskier?
# 2. `REFIT_PCA = False` — how much does the frozen rung-1 basis cost you?
#    (Watch the right panel.)
# 3. `N_RUNGS = 6` — do the widths keep falling, and what happens to the ESS?
#
# **Exercise 3** *(discussion)*. The printed ESS falls as the target sharpens.
# In the production campaign the proposal pool is 262144 samples per rung, not
# 4096, and the training uses the late-time trick from the tutorial's
# Exercise 3.3 — with those, posteriors reach ~1 nat from the
# information-theoretic optimum (arXiv:2510.13997). What you ran here is the
# same algorithm at 1/64 scale.
#
# ---
# *A caveat repeated from the top: this pipeline's conventions give SNR ~1170
# where the LDC-matched production analysis measures 393, so these posteriors
# are honestly tighter than the production E1 results by roughly that factor.
# The mechanism — buffer, tempered reweighting, warm-started flows, adaptive
# summaries — is identical.*

# %%
# (housekeeping cell — saves all figures when executed as a test script;
# does nothing in an interactive colab session)
if os.environ.get('TUTORIAL_SAVE_FIGS'):
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'seq_fig_{i:02d}.png', dpi=110,
                              bbox_inches='tight')
    print(f'saved {len(plt.get_fignums())} figures')
