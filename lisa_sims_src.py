# %% [markdown]
# # Simulating LISA data yourself — the real waveform stack, live
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_sims.ipynb)
#
# The main tutorial (`tutorial_lisa_sbi.ipynb`) used **pre-simulated** training
# data for the massive black-hole binary, because the waveform stack is a
# compiled C library. But that library — **lisabeta** (Marsat & Baker,
# [arXiv:1806.10734](https://arxiv.org/abs/1806.10734)): IMRPhenomD waveforms +
# the full Fourier-domain LISA response) — ships pre-built wheels on PyPI, so
# Colab can install it in ~20 seconds. Here we do exactly that, and rebuild a
# miniature version of the tutorial's training bank from scratch:
#
# 1. pip-install lisabeta,
# 2. generate the TDI response of a merging massive black-hole binary,
# 3. add instrument noise from the LISA noise model and whiten,
# 4. produce a small (θ, summary) training set — the same object the main
#    tutorial's Part 4 trained on.
#
# *(No GPU needed — waveform generation is CPU C code.)*

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('lisabeta') is None:
    print('installing lisabeta (pre-built wheel, ~20 s) ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lisabeta'],
                   check=True)

import numpy as np
import matplotlib.pyplot as plt

import lisabeta.lisa.lisa as lisa
import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.lisa.pyLISAnoise as pyLISAnoise

print('lisabeta imported OK')

# %% [markdown]
# ## 1. One binary, one waveform
#
# Parameters of the LDC1-1 ("Radler") massive black-hole binary — the same
# source the main tutorial analyzed. lisabeta wants component masses and
# aligned spins; we start from chirp mass and mass ratio like the tutorial
# did.

# %%
LOG10_MC, ETA = 6.18864, 0.21885          # chirp mass [Msun], symmetric ratio
LOG10_DL = 4.74823                        # luminosity distance [Mpc]
COS_INC, PHI0 = -0.33939, 6.24790         # inclination, phase
LAM, SIN_BETA, PSI = 3.50910, 0.28853, 0.20445   # ecliptic sky + polarization
A1, A2 = 0.75348, 0.62159                 # aligned spin components

Mc, eta = 10 ** LOG10_MC, ETA
Mtot = Mc / eta ** 0.6
m1 = 0.5 * Mtot * (1 + np.sqrt(1 - 4 * eta))
m2 = Mtot - m1
print(f'm1 = {m1:.3e} Msun, m2 = {m2:.3e} Msun (q = {m2/m1:.2f})')

T_OBS, DT = 86400.0, 10.0                 # one day at 10 s cadence
N = int(T_OBS / DT)
freqs = np.fft.rfftfreq(N, d=DT)          # frequency grid of our segment
tgrid = np.arange(N) * DT

params = {
    'm1': m1, 'm2': m2, 'chi1': A1, 'chi2': A2,
    'Deltat': T_OBS / 2,                  # merger centered in the window
    'dist': 10 ** LOG10_DL,               # Mpc
    'inc': np.arccos(COS_INC), 'phi': PHI0,
    'lambda': LAM, 'beta': np.arcsin(SIN_BETA), 'psi': PSI,
}
YEAR = 31558149.8                         # sidereal year [s], lisabeta's unit
wvf_pars = dict(minf=1e-5, maxf=0.1,
                # start the waveform exactly at the window start: our FFT grid
                # is periodic with T_OBS, so any earlier inspiral would wrap
                # around and contaminate the segment
                timetomerger_max=(T_OBS / 2) / YEAR,
                tmax=T_OBS / YEAR,        # observation span in years
                TDI='TDIAET', acc=1e-4, approximant='IMRPhenomD',
                LISAconst=pyresponse.LISAconstProposal,
                responseapprox='full', frozenLISA=False, TDIrescaled=False)

# %%
def tdi_fd(p):
    """Frequency-domain TDI A, E channels of a binary on our rfft grid."""
    fs = lisa.GenerateLISATDIFreqseries_SMBH(p, freqs, **wvf_pars)
    h = fs[(2, 2)]                        # the (l,m)=(2,2) mode, per TDI channel
    return h['chan1'], h['chan2']         # A, E  (T carries little signal)


A_fd, E_fd = tdi_fd(params)
print(f'FD waveform computed on {len(freqs)} bins')

# %% [markdown]
# ## 2. Noise model and whitening
#
# lisabeta also ships the LISA noise model (SciRD instrumental PSDs propagated
# through the same TDI combinations). **Whitening** = dividing the
# frequency-domain data by $\sqrt{S_n(f)}$, so every frequency bin carries
# equally-important, unit-variance noise — after which "signal-to-noise" means
# what it says and the toy-model intuition from the main tutorial applies
# literally.

# %%
psd = pyLISAnoise.evaluate_AET_psd(freqs[1:], TDIT=False,
                                   LISAnoise=pyLISAnoise.LISAnoiseSciRDv1,
                                   TDIrescaled=False)   # {'freq', 'TDIA', 'TDIE'}
S_A, S_E = np.asarray(psd['TDIA']), np.asarray(psd['TDIE'])


def whiten_td(h_fd, S):
    """FD -> whitened TD, normalized so pure instrument noise has unit
    variance per sample (the factor sqrt(N/2) makes that exact for a
    hermitian spectrum whose whitened bins are unit complex normals).
    lisabeta uses the e^{+2pi i f t} Fourier convention, opposite to
    numpy's — conjugate before the inverse FFT, else time runs backwards."""
    w = np.zeros_like(h_fd)
    w[1:] = np.conj(h_fd[1:]) / np.sqrt(S * T_OBS / 4)
    return np.fft.irfft(w, n=N) * np.sqrt(N / 2)


rng = np.random.default_rng(0)


def noise_td():
    """Whitened noise is unit white Gaussian by construction."""
    return rng.standard_normal(N)


# sanity check: draw FD instrument noise from the PSD (CTFT convention:
# E|n(f)|^2 = T*S(f)/2), push it through the SAME whitening, expect std = 1
g1, g2 = rng.standard_normal((2, len(freqs) - 1))
n_fd = np.zeros(len(freqs), complex)
n_fd[1:] = np.sqrt(T_OBS * S_A / 4) * (g1 + 1j * g2)
print(f'whitened instrument noise std = {whiten_td(n_fd, S_A).std():.3f} '
      f'(should be ~1.00)')


sig_A = whiten_td(A_fd, S_A)
data_A = sig_A + noise_td()
snr = np.sqrt(sum(np.sum(whiten_td(h, S) ** 2)
                  for h, S in [(A_fd, S_A), (E_fd, S_E)]))
print(f'network SNR (A+E) of the 1-day segment: {snr:.0f}  '
      f'(the LDC-matched analysis pipeline measures ~393 for this extract; '
      f'the offset is a constant waveform-normalization difference)')

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(tgrid / 3600, data_A, lw=.4, label='whitened data (A channel)')
ax.plot(tgrid / 3600, sig_A, 'C1', lw=1, label='whitened signal')
ax.set(xlabel='t [hours]', ylabel='whitened amplitude')
ax.legend(loc='upper left')
fig.tight_layout()

# %% [markdown]
# That plot should look qualitatively like the main tutorial's "one day of
# LISA data" figure — because it is the same physics through the same
# response, just a leaner pipeline (different epoch/window conventions, so
# not bit-identical to the tutorial's pre-simulated bank).
#
# **Exercise 1.** Change the sky position (`lambda`, `beta`) and re-run: the
# signal amplitude and shape change through the LISA antenna response. Change
# `Deltat` by ±6 hours: the merger moves in the window. Change `dist`: SNR
# scales as 1/D — at what distance does the signal disappear into the noise?

# %% [markdown]
# ## 3. Play with the source
#
# So far the simulator is a black box that emits one wiggly line. Let's open
# it: regenerate the waveform while changing the source parameters and watch
# what each one does physically. First a static comparison — three chirp
# masses, same everything else, zoomed into a few hours around merger:

# %%
def clean_waveform(log10_Mc=LOG10_MC, log10_DL=LOG10_DL, cos_inc=COS_INC,
                   lam=LAM, a1=A1, a2=A2, eta=ETA):
    """Whitened clean A-channel time series for modified source parameters."""
    Mc_ = 10 ** log10_Mc
    Mtot_ = Mc_ / eta ** 0.6
    m1_ = 0.5 * Mtot_ * (1 + np.sqrt(1 - 4 * eta))
    p = dict(params, m1=m1_, m2=Mtot_ - m1_, chi1=a1, chi2=a2,
             dist=10 ** log10_DL, inc=np.arccos(cos_inc), **{'lambda': lam})
    A_fd_, E_fd_ = tdi_fd(p)
    x = whiten_td(A_fd_, S_A)
    snr_ = np.sqrt(np.sum(x ** 2) + np.sum(whiten_td(E_fd_, S_E) ** 2))
    return x, snr_


fig, ax = plt.subplots(figsize=(10, 3.4))
for lmc, c in [(6.05, 'C0'), (LOG10_MC, 'C1'), (6.35, 'C2')]:
    x, snr_ = clean_waveform(log10_Mc=lmc)
    ax.plot(tgrid / 3600, x, c, lw=.9,
            label=f'$\\log_{{10}}M_c={lmc:.2f}$  (SNR {snr_:.0f})')
ax.set(xlim=(9, 14), xlabel='t [hours]', ylabel='whitened amplitude',
       title='heavier binaries chirp lower and merge faster')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()

# %% [markdown]
# Heavier binary → lower orbital frequency at merger and fewer visible cycles
# in the band; the SNR changes too (the whole spectrum slides across the LISA
# sensitivity bucket).
#
# Now the same thing with **sliders** (works in Colab; drag and the waveform
# regenerates — each drag is a genuine call into the C waveform+response
# code, ~0.1 s):

# %%
def show_waveform(log10_Mc=LOG10_MC, log10_DL=LOG10_DL, cos_inc=COS_INC,
                  lam=LAM, t_center_h=12.0, window_h=4.0):
    x, snr_ = clean_waveform(log10_Mc, log10_DL, cos_inc, lam)
    fig, ax = plt.subplots(figsize=(9.5, 3))
    ax.plot(tgrid / 3600, x, 'C1', lw=1)
    ax.set(xlim=(t_center_h - window_h / 2, t_center_h + window_h / 2),
           xlabel='t [hours]', ylabel='whitened amplitude',
           title=f'network SNR = {snr_:.0f}')
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(show_waveform,
             log10_Mc=FloatSlider(min=5.9, max=6.5, step=0.01, value=LOG10_MC),
             log10_DL=FloatSlider(min=4.0, max=5.5, step=0.05, value=LOG10_DL),
             cos_inc=FloatSlider(min=-1.0, max=1.0, step=0.05, value=COS_INC),
             lam=FloatSlider(min=0.0, max=6.28, step=0.1, value=LAM),
             t_center_h=FloatSlider(min=1.0, max=23.0, step=0.5, value=12.0),
             window_h=FloatSlider(min=0.5, max=24.0, step=0.5, value=4.0))
except ImportError:
    show_waveform()                        # static fallback without ipywidgets

# %% [markdown]
# Things to try:
# - `cos_inc` → ±1 (face-on/off) vs 0 (edge-on): amplitude changes ~2×, and
#   this is exactly the distance–inclination degeneracy from the main
#   tutorial's Part 4 — a closer edge-on binary mimics a farther face-on one.
# - `lam` (ecliptic longitude): the amplitude modulation is the LISA antenna
#   pattern — sky information enters through the response, not the waveform.
# - `log10_DL`: pure 1/D amplitude scaling. Combined with the two above,
#   you are *feeling* the parameter degeneracies the posterior has to unravel.
# - Widen `window_h` to 24 h and lower `log10_Mc` to 5.9: the inspiral fills
#   the day.

# %% [markdown]
# ## 4. A miniature training bank
#
# Now the real thing the tutorial needed: draw parameters from the narrowed
# prior, simulate whitened noisy data, compress with PCA. This is
# `make_tutorial_simbank.py` in miniature (256 sims instead of 32768, A
# channel only for speed).

# %%
import time

PRIOR = {                                  # the tutorial's narrowed prior box
    'log10_Mc': (6.186798, 6.190189), 'eta': (0.212865, 0.223907),
    'log10_DL': (4.0, 5.5), 'cos_inc': (-1.0, 1.0),
    'phi': (0.0, 2 * np.pi), 'dt_c': (-190.0, 280.0),
    'lambda': (3.219192, 3.887023), 'sin_beta': (-0.076401, 0.512674),
    'psi': (0.0, np.pi), 'a1': (0.660765, 0.856587), 'a2': (0.344118, 0.846804),
}


def draw_params(rng):
    u = {k: rng.uniform(*v) for k, v in PRIOR.items()}
    Mc, eta = 10 ** u['log10_Mc'], u['eta']
    Mtot = Mc / eta ** 0.6
    m1 = 0.5 * Mtot * (1 + np.sqrt(1 - 4 * eta))
    return {'m1': m1, 'm2': Mtot - m1, 'chi1': u['a1'], 'chi2': u['a2'],
            'Deltat': T_OBS / 2 + u['dt_c'], 'dist': 10 ** u['log10_DL'],
            'inc': np.arccos(u['cos_inc']), 'phi': u['phi'],
            'lambda': u['lambda'], 'beta': np.arcsin(u['sin_beta']),
            'psi': u['psi']}


N_SIMS = 256
t0 = time.time()
clean = np.empty((N_SIMS, N))
for i in range(N_SIMS):
    A_fd_i, _ = tdi_fd(draw_params(rng))
    clean[i] = whiten_td(A_fd_i, S_A)
per_sim = (time.time() - t0) / N_SIMS
print(f'{N_SIMS} sims in {time.time()-t0:.0f} s  ({per_sim*1e3:.0f} ms/sim) '
      f'-> full 32768-sim bank would take ~{per_sim*32768/60:.0f} min here')

# %%
mu = clean.mean(0)
U, S, Vh = np.linalg.svd(clean - mu, full_matrices=False)
eigs = S / np.sqrt(N_SIMS - 1)
noisy_summaries = (clean + rng.standard_normal(clean.shape) - mu) @ Vh[:64].T

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].semilogy(eigs[:120]); ax[0].axhline(1, color='r', ls='--', lw=1)
ax[0].set(xlabel='PCA component', ylabel='component SNR',
          title='summary spectrum (cf. tutorial Part 4)')
ax[1].plot(noisy_summaries[:, 0], noisy_summaries[:, 1], '.', ms=3, alpha=.5)
ax[1].set(xlabel='summary $s_1$', ylabel='summary $s_2$',
          title='training summaries (what the flow conditions on)')
fig.tight_layout()

# %% [markdown]
# You now have `(θ, s)` pairs made entirely in this notebook — the same kind
# of object `mbhb_simbank.npz` contains (that bank: 32k sims, both A and E
# channels, on the exact grid/conventions of the analysis pipeline).
#
# **Exercise 2.** Increase `N_SIMS` (mind the timing printed above) and watch
# the PCA spectrum converge. How many components sit above the noise line —
# and how does that compare to the ~46 of the tutorial's full bank?
#
# **Exercise 3** *(capstone)*. Feed your home-made bank into the main
# tutorial's Part 4 cells: z-score your θ draws and summaries, train
# `VelocityNet` with `fm_loss`, and sample the posterior for the whitened
# data you built in Section 2. You have now run simulation-based inference
# on LISA data end-to-end — simulator included — in a browser tab.
#
# ---
# ### Notes
# - Install is fast because lisabeta ships pre-built wheels (cp310–cp314,
#   linux/mac). The heavier `lisa-data-challenge` package (official LDC
#   pipelines, PSDs, TDI tooling) is source-only on PyPI and needs GSL/FFTW
#   to compile (`!apt install libgsl-dev libfftw3-dev` first on Colab) — not
#   needed here.
# - The production analysis behind the main tutorial uses the same lisabeta
#   engine through the LDC `FastBHB` wrapper, with carefully matched
#   windowing/epoch conventions — see the campaign repo pointers in the README.

# %%
import os
if os.environ.get('TUTORIAL_SAVE_FIGS'):
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'sims_fig_{i:02d}.png', dpi=110,
                              bbox_inches='tight')
    print(f'saved {len(plt.get_fignums())} figures')
