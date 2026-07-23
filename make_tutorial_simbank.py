"""Generate the pre-simulated MBHB training bank for the SBI tutorial notebook
(tutorial_lisa_sbi.ipynb, Part 4).

Self-contained: builds its own PCA basis (8192 clean sims), then simulates
32768 noisy training examples at the MCMC-narrowed prior and projects them to
64 PCA coefficients.  Also packs the observation summary, truth, prior bounds,
the PCA spectrum, an example waveform pair for plotting, and a BakerMarsat
(D_L, cos iota) scatter subsample for comparison.

Output: tutorials/mbhb_simbank.npz  (~12 MB -- NOT committed to git;
run this script on a cluster with the LDC stack to regenerate).

Usage:  python tutorials/make_tutorial_simbank.py
Needs:  GPU optional (sims are the cost, ~3 min on A100), repo deps
        (src/model.py simulator + ldc/lisabeta), data/x_obs_mbmb_1d.npy.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get('DSBI_REPO', '/home/weniger/dsbi-ldc-mbhb/falcon-LDC-MBHB'))
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

import model as M                                    # noqa: E402
from plot_sbi_mcmc_posterior import load_bakermarsat_samples  # noqa: E402

RNG = np.random.default_rng(42)
N_BASIS = 8192          # clean sims for the PCA basis
N_TRAIN = 32768         # noisy training sims
K = 64                  # summary dimension
N_MCMC = 8000           # BakerMarsat scatter subsample

# MCMC-narrowed prior (same box as the standalone campaign), in the
# tutorial's parameter order: the two "interesting" parameters first.
PARAMS = [
    # (name, low, high, z_column in the 11-dim simulator convention)
    ('log10_DL',  4.0,       5.5,       2),
    ('cos_iota', -1.0,       1.0,       3),
    ('log10_Mc',  6.186798,  6.190189,  0),
    ('eta',       0.212865,  0.223907,  1),
    ('t_c_yrs',  -0.000006,  0.000009,  5),
    ('lambda',    3.219192,  3.887023,  6),
    ('sin_beta', -0.076401,  0.512674,  7),
    ('a1',        0.660765,  0.856587,  9),
    ('a2',        0.344118,  0.846804, 10),
]
# gauge angles phi0 (4) and psi (8): sampled into the waveform, not inferred
GAUGE = [(4, 0.0, 2 * np.pi), (8, 0.0, np.pi)]
Z_TRUE_11 = np.load(REPO / 'data' / 'z_true_1d.npy').ravel()


def draw_z11(n):
    """n draws from the narrowed prior in the simulator's 11-dim convention."""
    z = np.empty((n, 11))
    for name, lo, hi, col in PARAMS:
        z[:, col] = RNG.uniform(lo, hi, n)
    for col, lo, hi in GAUGE:
        z[:, col] = RNG.uniform(lo, hi, n)
    return z


def simulate(xr, z11, chunk=512):
    out = []
    for i in range(0, len(z11), chunk):
        x = np.asarray(xr.simulate_batch(len(z11[i:i + chunk]),
                                         z11[i:i + chunk].astype(np.float32)))
        out.append(x[:, :2].reshape(len(x), -1).astype(np.float32))  # A,E channels
        if (i // chunk) % 8 == 0:
            print(f'  {i + len(out[-1])}/{len(z11)}', flush=True)
    return np.concatenate(out)


def main():
    out_path = Path(__file__).resolve().parent / 'mbhb_simbank.npz'

    print(f'[1/4] PCA basis: {N_BASIS} clean sims ...', flush=True)
    xr_clean = M.XRaw(noise_scale=0.0)
    Xc = simulate(xr_clean, draw_z11(N_BASIS)).astype(np.float64)
    mu = Xc.mean(0)
    U, S, Vh = np.linalg.svd(Xc - mu, full_matrices=False)
    eigs = (S / np.sqrt(N_BASIS - 1)).astype(np.float32)   # per-component SNR
    V = Vh[:K].astype(np.float64)
    print(f'      components with SNR>1: {(eigs > 1).sum()}, '
          f'top-{K} captures {100 * (eigs[:K] ** 2).sum() / (eigs ** 2).sum():.1f}% '
          f'of ensemble SNR^2', flush=True)

    print(f'[2/4] training bank: {N_TRAIN} noisy sims ...', flush=True)
    xr_noisy = M.XRaw(noise_scale=1.0)
    z11 = draw_z11(N_TRAIN)
    Xn = simulate(xr_noisy, z11)
    s_train = ((Xn.astype(np.float64) - mu) @ V.T).astype(np.float32)
    cols = [c for _, _, _, c in PARAMS]
    z_train = z11[:, cols].astype(np.float32)

    print('[3/4] observation + extras ...', flush=True)
    x_obs = np.load(REPO / 'data' / 'x_obs_mbmb_1d.npy')[:2].astype(np.float64)
    s_obs = ((x_obs.ravel() - mu) @ V.T).astype(np.float32)
    z_true = Z_TRUE_11[cols].astype(np.float32)
    x_clean_true = np.asarray(
        xr_clean.simulate_batch(1, Z_TRUE_11[None].astype(np.float32)))[0, :2]
    mc = load_bakermarsat_samples(
        Path('/gpfs/home2/weniger/dsbi-ldc-mbhb/BakerMarsat-LDC1-1_MBHB/'
             'fullRadWNFL-2D_b_1_resampled_SSBframe_LDC.dat'))
    mc_sub = mc[RNG.choice(len(mc), N_MCMC, replace=False)][:, [2, 3]].astype(np.float32)

    print('[4/4] writing ...', flush=True)
    np.savez_compressed(
        out_path,
        z_train=z_train, s_train=s_train,
        s_obs=s_obs, z_true=z_true,
        names=np.array([n for n, _, _, _ in PARAMS]),
        bounds=np.array([[lo, hi] for _, lo, hi, _ in PARAMS], dtype=np.float32),
        pca_eigs=eigs[:256],
        x_obs=x_obs.astype(np.float32),                # (2, 5760) A,E channels
        x_clean_true=x_clean_true.astype(np.float32),  # clean signal at truth
        mcmc_dl_cosi=mc_sub,
        readme=np.array(
            'LISA SBI tutorial bank. z_train: 9 params (see names/bounds), '
            'phi0/psi randomized into the waveforms (gauge nuisances). '
            's_train: 64 PCA coefficients of noisy whitened A,E data. '
            'Data: LDC1-1 Radler MBHB, 1 day, dt=15s. Generated by '
            'make_tutorial_simbank.py.'),
    )
    print(f'wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    if '--focused' in sys.argv:
        main_focused()
        raise SystemExit
    main()


def main_focused():
    """--focused: rung-2 bank. Draw 32768 params from the E1 posterior samples
    broadened 3x (Gaussian kernel per dim, clipped to the narrowed prior box),
    project with the SAME PCA basis (regenerated deterministically, seed 42)."""
    out_path = Path(__file__).resolve().parent / 'mbhb_simbank_focused.npz'
    e1 = np.load('/gpfs/home2/weniger/dsbi-ldc-mbhb/falcon-LDC-MBHB/'
                 'standalone_tests/fm_dynamic_32k_E1.npy')   # (10000, 9)
    print('[1/3] PCA basis (identical to main bank: seed-42 clean sims) ...', flush=True)
    xr_clean = M.XRaw(noise_scale=0.0)
    Xc = simulate(xr_clean, draw_z11(N_BASIS)).astype(np.float64)
    mu = Xc.mean(0)
    _, S, Vh = np.linalg.svd(Xc - mu, full_matrices=False)
    V = Vh[:K].astype(np.float64)
    print('[2/3] focused draws + sims ...', flush=True)
    idx = RNG.choice(len(e1), N_TRAIN)
    z9 = e1[idx] + 3.0 * e1.std(0) * RNG.standard_normal((N_TRAIN, 9))
    for j, (name, lo, hi, col) in enumerate(PARAMS):
        z9[:, j] = np.clip(z9[:, j], lo, hi)
    z11 = draw_z11(N_TRAIN)                    # gauge angles randomized
    for j, (name, lo, hi, col) in enumerate(PARAMS):
        z11[:, col] = z9[:, j]
    Xn = simulate(M.XRaw(noise_scale=1.0), z11)
    s_train = ((Xn.astype(np.float64) - mu) @ V.T).astype(np.float32)
    print('[3/3] writing ...', flush=True)
    np.savez_compressed(out_path,
                        z_train=z9.astype(np.float32), s_train=s_train,
                        readme=np.array('rung-2 bank: params from E1 posterior '
                                        'broadened 3x, same PCA basis/s_obs as '
                                        'mbhb_simbank.npz'))
    print(f'wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')
