# %% [markdown]

# # From MLPs to LISA: simulation-based inference, step-by-step — *with solutions*
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_solutions.ipynb)
#
# > This is the **worked-solutions** version of the tutorial: every exercise is
# > followed by a reference solution and a discussion of what it shows. The
# > version to hand out is `tutorial_lisa_sbi.ipynb`, which has the same text
# > with empty cells in their place.
#
# | part | idea | new ingredient |
# |---|---|---|
# | 1 | fit a *function* with a neural network | MLPs, overfitting, early stopping |
# | 2 | fit a *distribution* | flow matching (FM), conditioning |
# | 3 | fit a *posterior*: feed FM pairs from a simulator | SBI, amortization |
# | 4 | a toy gravitational wave | data compression + **sequential** inference |
#
# > **Colab setup:** Runtime → Change runtime type → **T4 GPU**, then run all
# > cells top to bottom. Everything also works on CPU, just slower.
#
# > **New to PyTorch?** There is a short **FAQ at the end of this notebook**
# > answering the things that trip people up on a first read (`.detach()`,
# > `no_grad()`, why shapes are `(n, 1)`, what `state_dict` is). The official
# > references worth keeping open are the
# > [60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html),
# > [`torch.nn`](https://pytorch.org/docs/stable/nn.html) and
# > [`torch.optim`](https://pytorch.org/docs/stable/optim.html).

# %%

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {dev}' + ('' if dev == 'cuda' else '  (enable a GPU runtime for comfort!)'))

# %% [markdown]

# ---
# # Part 1 — Neural networks

# %% [markdown]

# ## The network

# %% [markdown]

# A **multi-layer perceptron** (MLP) is a program that turns an input vector
# into an output vector. The same object is called a *feed-forward network*, a
# *dense network*, or a *fully connected network*. It is the building block of
# modern machine learning: convolutional networks, transformers and the flows of
# Part 2 are all MLPs with extra structure bolted on.
#
# It is a chain of **affine maps** — matrix multiply plus constant shift — each
# followed by a **nonlinearity** $g$ applied componentwise. For an input
# $x \in \mathbb R^{d_{\rm in}}$, three hidden layers of width $H$, and an
# output in $\mathbb R^{d_{\rm out}}$:
#
# $$\begin{aligned}
#   h^{(1)} &= g\big(W^{(1)} x + b^{(1)}\big),
#     & W^{(1)} &\in \mathbb R^{H \times d_{\rm in}} \\
#   h^{(2)} &= g\big(W^{(2)} h^{(1)} + b^{(2)}\big),
#     & W^{(2)} &\in \mathbb R^{H \times H} \\
#   h^{(3)} &= g\big(W^{(3)} h^{(2)} + b^{(3)}\big),
#     & W^{(3)} &\in \mathbb R^{H \times H} \\
#   \hat y  &= W^{(4)} h^{(3)} + b^{(4)},
#     & W^{(4)} &\in \mathbb R^{d_{\rm out} \times H}
# \end{aligned}$$
#
# Compactly, $\hat y = \mathrm{MLP}_\phi(x)$, with $\phi$ collecting every
# $W^{(l)}$ and $b^{(l)}$ — the numbers training moves. Without $g$ the chain
# would collapse to one affine map; the read-out deliberately has no $g$, so
# $\hat y$ can take any value.

# %%

class MLP(nn.Module):
    """A dense feed-forward network: x (n, d_in) -> y_hat (n, d_out).

    Three hidden layers of width `hidden`, each followed componentwise by the
    nonlinearity `act`. The read-out is affine, with no `act`, so y_hat can
    take any value.

    Arguments:
      d_in: number of input features per point.
      d_out: number of output values per point.
      hidden: width of each of the three hidden layers.
      act: the nonlinearity, e.g. torch.relu or torch.selu.
    """

    def __init__(self, d_in=1, d_out=1, hidden=256, act=torch.relu):
        super().__init__()
        self.act = act                             # torch.relu, torch.selu, ...
        self.fc1 = nn.Linear(d_in, hidden)         # W1: (hidden, d_in)
        self.fc2 = nn.Linear(hidden, hidden)       # W2: (hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)       # W3: (hidden, hidden)
        self.out = nn.Linear(hidden, d_out)        # W4: (d_out, hidden)

    def forward(self, x):           # x: (n, d_in) — n points, d_in features each
        h = self.act(self.fc1(x))   # W1 @ x + b1, then g   -> (n, hidden)
        h = self.act(self.fc2(h))   # W2 @ h + b2, then g   -> (n, hidden)
        h = self.act(self.fc3(h))   # W3 @ h + b3, then g   -> (n, hidden)
        return self.out(h)          # W4 @ h + b4, no g     -> (n, d_out)

# %% [markdown]

# `nn.Linear` starts each $W$ and $b$ off at small random values, so an
# untrained network is already a valid — if useless — function. Here is what
# freshly initialized networks compute, with two different nonlinearities.
# (Deliberately narrow, `hidden=8`, so that individual kinks stay visible; at
# width 256 there are too many of them to see.)

# %%

xg = torch.linspace(-5, 5, 400)[:, None]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for a, act, nm in [(ax[0], torch.relu, 'ReLU'), (ax[1], torch.selu, 'SELU')]:
    for s in range(6):
        torch.manual_seed(s)
        a.plot(xg, MLP(1, 1, 8, act=act)(xg).detach(), lw=1.1)
    a.axvspan(-1, 1, color='k', alpha=.07)
    a.set(xlabel='x', title=f'{nm}, 6 random initializations (hidden=8)')
ax[0].set_ylabel(r'$\hat y$')
fig.tight_layout()

# %% [markdown]

# Two things to take from this:
#
# - **The interesting behaviour lives at $O(1)$.** The curves bend inside the
#   grey band and are straight and boring outside it — past its last kink a ReLU
#   network is exactly affine. So always **normalize** a network's inputs and
#   outputs to roughly mean zero and unit variance; every later part of this
#   notebook z-scores its data for this reason.
# - **Smoothness is inherited from $g$.** ReLU gives corners, SELU gives smooth
#   curves. Neither is more powerful, but depending on the problem one may be easier to train than the other.

# %% [markdown]

# ## Fitting with gradient descent

# %% [markdown]

# Given pairs $(x_i, y_i)$, assume
#
# $$y_i = y(x_i) + \epsilon_i, \qquad \epsilon_i \sim \mathcal N(0, \sigma^2)$$
#
# for an unknown function $y(x)$. The network's job is to approximate it,
# $\mathrm{MLP}_\phi(x) \approx y(x)$, and we fit it by **maximum likelihood**:
# make the data we observed as probable as possible under the model.
#
# Gaussian noise gives each observation the density
#
# $$p(y_i \,|\, x_i, \phi, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}
#   \exp\!\left[-\frac{\big(y_i - \mathrm{MLP}_\phi(x_i)\big)^2}{2\sigma^2}\right],$$
#
# and the $N$ of them multiply. Taking the log (products underflow) and flipping
# the sign (optimizers minimize) leaves the **negative log-likelihood**, averaged
# over the data and dropping the constant $\tfrac12\log 2\pi$:
#
# $$\underbrace{\mathcal L(\phi, \sigma)}_{\text{NLL}}
#   = \frac{\mathrm{MSE}(\phi)}{2\sigma^2} + \log\sigma ,
#   \qquad \mathrm{MSE}(\phi) = \frac{1}{N} \sum_{i=1}^{N}
#   \big(\mathrm{MLP}_\phi(x_i) - y_i\big)^2 .$$
#
# The **mean squared error** is the only $\phi$-dependent part, so at fixed
# $\sigma$ minimizing the NLL is minimizing the MSE. What $\sigma$ adds is a
# scale: errors measured in units of the claimed uncertainty, with $\log\sigma$
# preventing $\sigma \to 0$. Minimizing over $\sigma$ as well gives
# $\hat\sigma^2 = \mathrm{MSE}_{\rm train}$, which we plug in each epoch rather
# than fit. (Lower NLL is better; unlike an MSE it can go negative.)
#
# We minimize by **gradient descent**,
#
# $$\phi_{k+1} = \phi_k - \eta\, \nabla_\phi \mathcal L(\phi_k),$$
#
# from the random $\phi_0$ above, with the **learning rate** $\eta$ setting the
# step size. One pass over all $N$ examples is an *epoch*.
#
# ```text
# ALGORITHM  fit(net, train, validation, eta, patience)
# ──────────────────────────────────────────────────────────────
# phi ← all trainable parameters of net (every W and b)
# opt ← Adam(phi, learning rate eta)
#
# repeat for each epoch:
#     y_pred ← net(x_train)                    # forward pass
#     MSE    ← mean (y_pred − y_train)²         # scalar loss
#     g      ← ∂MSE/∂phi                        # backward pass
#     phi    ← phi − eta·g                      # update, in place
#
#     sigma  ← sqrt(MSE_train)                  # plug-in noise level
#     L_val  ← MSE_val/(2 sigma²) + log sigma   # validation NLL
#     if L_val is the best so far: remember phi
#     if no improvement for `patience` epochs: stop
#
# return the remembered phi                     # "early stopping"
# ──────────────────────────────────────────────────────────────
# ```
#
# The last two steps are **early stopping**: keep the parameters from the best
# validation epoch, not the last one. It makes the number of epochs a knob you
# no longer have to tune — pass something large and let `patience` decide.

# %%

def fit(net, x, y, x_val, y_val, lr=1e-4, patience=300, epochs=100_000,
        rewind=True):
    """Train on MSE; monitor the Gaussian negative log-likelihood; stop early.

    The NLL is minus the log probability the model assigns to the data, using
    sigma^2 = mean squared TRAINING residual. Lower is better; it can go
    negative.

    Arguments:
      net: the network to train, modified in place.
      x, y: training inputs (n, d_in) and targets (n, d_out).
      x_val, y_val: held-out inputs and targets, used only for monitoring.
      lr: Adam learning rate.
      patience: stop after this many epochs with no new best validation NLL.
      epochs: hard cap, normally never reached because patience fires first.
      rewind: keep the best epoch's weights (True) or the last epoch's (False).

    Returns:
      (hist, best_ep): the per-epoch table of (training NLL, validation NLL,
      sigma) as an array, and the index of the epoch whose weights we kept.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)   # holds pointers to phi
    hist, best, best_ep, snap = [], np.inf, 0, None

    for ep in range(epochs):
        mse = ((net(x) - y) ** 2).mean()               # forward pass + loss
        opt.zero_grad()                                # PyTorch accumulates grads
        mse.backward()                                 # fills p.grad for every p
        opt.step()                                     # one gradient step

        with torch.no_grad():                          # monitoring only
            sig2 = ((net(x) - y) ** 2).mean()          # plug-in sigma^2
            mse_val = ((net(x_val) - y_val) ** 2).mean()
            nll = (0.5 + 0.5 * sig2.log()).item()      # train NLL = 0.5 + log sigma
            nll_val = (0.5 * mse_val / sig2 + 0.5 * sig2.log()).item()
        hist.append((nll, nll_val, sig2.sqrt().item()))

        if nll_val < best:                             # new best: snapshot phi
            best, best_ep = nll_val, ep
            snap = copy.deepcopy(net.state_dict())
        if ep - best_ep > patience:                    # stalled: stop
            break

    if rewind:
        net.load_state_dict(snap)                      # rewind to the best epoch
        print(f'stopped at epoch {ep + 1}; kept epoch {best_ep + 1} '
              f'(validation NLL {best:.2f})')
    else:                                              # keep the final weights
        best_ep = ep
        print(f'stopped at epoch {ep + 1}; kept the LAST epoch '
              f'(validation NLL {hist[-1][1]:.2f})')
    return np.array(hist), best_ep

# %% [markdown]

# ## Example: measuring a frequency

# %% [markdown]

# The input need not be a single number: `MLP(30, 1)` maps a 30-dimensional
# vector to a scalar. As an example we take a prototypical parameter-estimation
# problem — measuring the frequency of a noisy sine.
#
# The signal is sampled at 30 fixed times and buried in noise,
#
# $$x_j = \sin(2\pi\,\nu\,t_j) + 0.3\,\varepsilon_j, \qquad
#   t_j = \tfrac{j}{29},\quad j = 0 \dots 29, \qquad
#   \nu \sim U(1, 5)\ \text{cycles},$$
#
# with $\varepsilon_j \sim \mathcal N(0, 1)$. The network sees the 30 samples
# $x$ and must return $\nu$. The frequencies stay well below this grid's Nyquist
# limit of 15 cycles, so nothing here is ambiguous in principle.
#
# One sentence worth keeping in mind: a network trained on squared error
# converges to the conditional mean $\mathbb E[\nu \,|\, x]$, so its output is a
# **posterior mean** and the estimated $\hat\sigma$ is a **posterior width**. This is already
# simulation-based inference, just a very simplified form of it (Gaussian posterior, homoscedastic noise).

# %%

N_GRID, NU_LO, NU_HI = 30, 1.0, 5.0
tj = torch.linspace(0, 1, N_GRID)
SPAN = NU_HI - NU_LO


def sine_data(n, sigma=0.3, seed=0, random_phase=False, nu_hi=NU_HI):
    """Simulate noisy samples of a sine, and the frequency to recover from them.

    Arguments:
      n: how many examples to draw.
      sigma: standard deviation of the noise added to each sample.
      seed: torch seed, so the training and held-out sets can differ.
      random_phase: give each example a uniform random phase instead of zero.
      nu_hi: top of the frequency band; the bottom is always NU_LO.

    Returns:
      (x, nu): the noisy samples x (n, 30), and the target frequency (n, 1)
      *rescaled* to [0, 1] as (nu - NU_LO) / span rather than left in cycles.
      Networks train much better when inputs and targets are O(1), so that the
      initial weights produce outputs of roughly the right size; multiply by
      span and add NU_LO to read anything back in cycles.
    """
    torch.manual_seed(seed)
    span = nu_hi - NU_LO
    nu = NU_LO + span * torch.rand(n, 1)                     # nu ~ U(1, nu_hi)
    phi = torch.rand(n, 1) * 2 * np.pi if random_phase else torch.zeros(n, 1)
    x = torch.sin(2 * np.pi * nu * tj + phi) + sigma * torch.randn(n, N_GRID)
    return x, (nu - NU_LO) / span


x_tr, nu_tr = sine_data(300, seed=0)                # training set
x_va, nu_va = sine_data(2000, seed=1)               # held-out set

fig, ax = plt.subplots(1, 3, figsize=(13, 2.7), sharey=True)
for a, i in zip(ax, range(3)):
    nu_i = NU_LO + SPAN * nu_tr[i, 0]
    a.plot(tj, torch.sin(2 * np.pi * nu_i * tj), 'k--', lw=1, label='hidden signal')
    a.plot(tj, x_tr[i], 'C0o-', ms=4, lw=.8, label='what the network sees')
    a.set(xlabel='t', title=fr'$\nu$ = {nu_i:.2f} cycles')
ax[0].set_ylabel('x'); ax[0].legend(fontsize=8)
fig.tight_layout()

# %%

# Two helpers, so that every fit below can be read out and plotted the same way.

def evaluate(net, hist, best_ep, x_val, nu_val, nu_hi=NU_HI):
    """Read a finished fit out in cycles, undoing the [0, 1] target rescaling.

    Arguments:
      net: the trained network.
      hist, best_ep: the two values returned by fit().
      x_val, nu_val: the held-out set to evaluate on.
      nu_hi: top of the frequency band the fit used.

    Returns:
      (nu_true, nu_est, rmse, sigma_hat), all in cycles. rmse is the
      estimator's *real* scatter on held-out data; sigma_hat is the error bar
      the model *claims* -- one number for every input, taken from the plug-in
      sigma^2 (the mean squared training residual) at the epoch we kept.
    """
    span = nu_hi - NU_LO
    sigma_hat = hist[best_ep, 2] * span                        # claimed, in cycles
    with torch.no_grad():
        nu_est = (NU_LO + span * net(x_val)).squeeze()
    nu_true = (NU_LO + span * nu_val).squeeze()
    rmse = (nu_est - nu_true).pow(2).mean().sqrt().item()       # real, in cycles
    return nu_true, nu_est, rmse, sigma_hat


def plot_fit(hist, best_ep, nu_true, nu_est, sigma_hat, nu_hi=NU_HI, label=None):
    """Draw the standard two-panel summary of a finished fit.

    Left: estimate against truth, with the band the model claims. Right: the
    two negative log-likelihood curves and the epoch we kept.

    Arguments:
      hist, best_ep: the two values returned by fit().
      nu_true, nu_est: truth and estimate in cycles, from evaluate().
      sigma_hat: the error bar the model claims, in cycles, from evaluate().
      nu_hi: top of the frequency band, for the axis limits.
      label: optional figure title.
    """
    rmse = (nu_est - nu_true).pow(2).mean().sqrt()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].fill_between([NU_LO, nu_hi], [NU_LO - sigma_hat, nu_hi - sigma_hat],
                       [NU_LO + sigma_hat, nu_hi + sigma_hat], color='C0',
                       alpha=.18,
                       label=fr'model claims $\pm\hat\sigma$ = {sigma_hat:.2f} cycles')
    ax[0].plot([NU_LO, nu_hi], [NU_LO, nu_hi], 'k--', lw=1)
    ax[0].plot(nu_true, nu_est, 'C0.', ms=2, alpha=.35, label='held-out data')
    ax[0].set(xlabel=r'true $\nu$ [cycles]', ylabel=r'estimated $\nu$ [cycles]',
              aspect='equal', title=f'held-out RMSE {rmse:.3f} cycles')
    ax[0].legend(fontsize=8, loc='upper left')

    lo = hist[:, 1].min()                            # frame on the validation dip
    ax[1].plot(hist[:, 0], lw=1, label='training NLL')
    ax[1].plot(hist[:, 1], lw=1, label='validation NLL')
    ax[1].axvline(best_ep, color='k', ls=':', lw=1, label='best epoch (kept)')
    ax[1].set(xlabel='epoch', ylabel='negative log-likelihood',
              ylim=(lo - 1.8, lo + 2.5))
    ax[1].legend(fontsize=8)
    if label:
        fig.suptitle(label, fontsize=10)
    fig.tight_layout()

# %%

freq_net = MLP(N_GRID, 1, 256)                      # 30 numbers in, 1 out
hist, best_ep = fit(freq_net, x_tr, nu_tr, x_va, nu_va)

nu_true, nu_est, rmse, sigma_hat = evaluate(freq_net, hist, best_ep, x_va, nu_va)
resid = nu_est - nu_true

# %%

plot_fit(hist, best_ep, nu_true, nu_est, sigma_hat)

# %%

# RMSE = the real scatter on held-out data; sigma_hat = the error bar claimed.
print('actual scatter, in bands of frequency:')
for lo in range(1, 5):
    m = (nu_true >= lo) & (nu_true < lo + 1)
    print(f'  nu = {lo}-{lo + 1} cycles:  RMSE {resid[m].pow(2).mean().sqrt():.3f}'
          f'   (model claims {sigma_hat:.3f})')

# %% [markdown]

# **Overfitting, unmistakably.** Remember that the NLL is minus the log
# probability the model assigns to the data, so down is better and negative is
# fine. The training NLL falls without limit, because
# $\hat\sigma$ tracks training residuals and those keep shrinking as the network
# memorizes its 300 examples. The validation NLL bottoms out and then leaves the
# top of the panel: a network starts overfitting noise, and achieves precision on data it has
# already seen that it cannot keep on unseen data. Early stopping keeps the network
# in the range where the network residuals on training data are comparable to network residuals on held-out data, 
# and the model remains realistic regarding its own precision.
#
# Note: The band is the model's
# $\hat\sigma$, identical at every frequency, while the printout shows the real
# scatter roughly doubling across the band — 30 samples cover fewer points per
# cycle as $\nu$ grows. One $\sigma$ for all inputs cannot account for the frequency dependence of the measurement uncertainty.
#
# What we want is a distribution over $\nu$ that **depends on the data**: wide
# here, narrow there, and in general one that can be curved and multi-modal. Approaches to this problem will be discussed in Part 2.

# %% [markdown]

# ### Exercise 1

# %% [markdown]

# `experiment` below re-runs everything with whatever you change. Try:
#
# 1. **`rewind=False`** — keep the *last* epoch instead of the best one, i.e.
#    switch early stopping off. What happens to $\hat\sigma$ compared with the
#    real error? (And note what `patience` alone does: since we always rewind to
#    the best epoch, raising it only burns compute.)
# 2. **`width`** — 8, then 1024. Does the widest network do worst?
# 3. **`n_train`** — 100, then 2000. Which improves, the RMSE or the honesty of
#    $\hat\sigma$?
# 4. **`nu_hi`** — push the top of the band to 12 cycles (Nyquist is 15). Where
#    does the estimator fail first, and does $\hat\sigma$ notice?
# 5. **`random_phase=True`** — now the same frequency can produce completely
#    different data, and the network has to become phase-invariant on its own.
#    (Part 4 treats the phase of its chirp exactly this way: a parameter you
#    must marginalize over but never infer.)

# %%

def experiment(n_train=300, width=256, patience=300, sigma=0.3,
               nu_hi=NU_HI, random_phase=False, rewind=True, plot=True,
               label=None):
    """Re-run the whole frequency fit with one setting changed, and plot it.

    Draws a fresh training set and the usual 2000-example held-out set (a
    different seed, so it is genuinely unseen), builds an MLP of the given
    width, fits it with early stopping, then draws the same two panels as above.

    Arguments:
      n_train: size of the training set.
      width: hidden width of the MLP.
      patience: passed to fit(); since we always rewind to the best epoch,
        raising it only burns compute.
      sigma: noise level in the simulated data.
      nu_hi: top of the frequency band.
      random_phase: randomize the phase, so the network has to become
        phase-invariant on its own.
      rewind: keep the best epoch (True) or the last one (False).
      plot: draw the two panels, or only print the numbers.
      label: optional figure title.

    Returns:
      (rmse, sigma_hat) in cycles: the real scatter of the estimate on held-out
      data, and the single error bar the model claims. Their ratio is how
      honest the fit is -- 1.0 would be perfectly calibrated, larger means
      over-confident.
    """
    xt, nt = sine_data(n_train, sigma, seed=0, random_phase=random_phase,
                       nu_hi=nu_hi)                # nu_hi is passed through, so
    xv, nv = sine_data(2000, sigma, seed=1, random_phase=random_phase,
                       nu_hi=nu_hi)                # no global state is touched
    net = MLP(N_GRID, 1, width)
    h, bep = fit(net, xt, nt, xv, nv, patience=patience, rewind=rewind)
    nu_t, nu_e, rmse, sig = evaluate(net, h, bep, xv, nv, nu_hi=nu_hi)
    print(f'  held-out RMSE {rmse:.3f} cycles,  model claims '
          f'sigma_hat {sig:.3f}  -> off by x{rmse / sig:.2f}')
    if plot:
        plot_fit(h, bep, nu_t, nu_e, sig, nu_hi=nu_hi, label=label)
    return rmse, sig


# %%

# your code goes here
#
# Call experiment() a few times, changing one argument at a time, and note the
# RMSE and the claimed sigma each run reports.
experiment()


# %%

# @title Reference solution { display-mode: "form" }
freq_runs = {
    'baseline':          dict(),
    'no early stopping': dict(rewind=False),
    'width = 8':         dict(width=8),
    'width = 1024':      dict(width=1024),
    'n_train = 100':     dict(n_train=100),
    'n_train = 2000':    dict(n_train=2000),
    'nu up to 12':       dict(nu_hi=12.0),
    'random phase':      dict(random_phase=True),
}
freq_results = {}
for name, kw in freq_runs.items():
    print(f'{name}:')                              # panels for the two that
    plot = name in ('baseline', 'no early stopping')   # make the point; the rest
    freq_results[name] = experiment(label=name, plot=plot, **kw)  # go in the table

print(f'\n{"setting":18s} {"RMSE":>7s} {"claims":>8s} {"ratio":>7s}')
for name, (r, sg) in freq_results.items():
    print(f'{name:18s} {r:7.3f} {sg:8.3f} {r / sg:7.2f}')

# %% [markdown]

# **What you should see.** Switching early stopping off is the instructive one,
# and not in the way you might expect: the RMSE actually *improves* slightly
# (0.25 → 0.21 cycles), because more training does sharpen the point estimate.
# What breaks is the error bar — $\hat\sigma$ falls to 0.05 cycles, so the model
# is wrong about its own precision by a factor of four. A model can get *more*
# accurate and much *less* honest at the same time, and only held-out data tells
# you which is happening. `patience` on its own does nothing to the result: we
# always rewind to the best epoch, so raising it only spends more epochs finding
# the same network.
#
# `width=8` underfits, with too few kinks to represent the map from 30 samples to
# a frequency; `width=1024` is *no worse* than the default, because early
# stopping is doing the regularizing rather than the architecture. More data
# improves accuracy a lot (the RMSE roughly halves from 300 to 2000 examples) but
# **not** calibration — the ratio stays near 1.4 throughout, because what is
# wrong is not the amount of data but the *shape* of the model: one global
# $\sigma$ cannot describe an error that depends on $\nu$. Pushing the band to
# 12 cycles makes that worse still, and randomising the phase costs real accuracy,
# since the network must discover an invariance to a parameter it is never asked
# about.

# %% [markdown]

# ---
# # Part 2 — Modeling distributions with flow matching

# %% [markdown]

# Part 1 fitted a *function* — one number per input. Now we fit a **conditional
# distribution**: given a condition $c$, produce samples of $\theta$. The lecture
# covered why this is hard (a density must be non-negative and integrate to one,
# and that normalization is a *global* constraint, so an MLP's output cannot
# simply be declared a density) and why flow matching sidesteps it. Here we build
# the thing.
#
# Notation for the rest of the notebook: $\theta$ is the quantity whose
# distribution we want, $c$ is whatever we condition on, and $\phi$ are the
# learnable network parameters. The goal is
#
# $$q_\phi(\theta \,|\, c) \;\approx\; p(\theta \,|\, c),$$
#
# learned from example pairs $(\theta, c)$ — and *nothing else*. We never
# evaluate $p$, only sample from it.

# %% [markdown]

# ## Flow matching

# %% [markdown]

# Build the sampler out of a **flow**. Start from a unit Gaussian and move the
# points continuously until they are distributed like $p(\theta|c)$. The motion
# is described by a **velocity field** $v_\phi(\theta, t \,|\, c)$ — an MLP
# exactly like Part 1's, taking a position $\theta$, a time $t \in [0,1]$ and the
# condition $c$ — so "training the generative model" means fitting that velocity
# field. The price is that sampling costs an ODE solve rather than one forward
# pass.

# %% [markdown]

# ### The mechanics, in three equations

# %% [markdown]

# Write $\theta_0$ for a point drawn from the Gaussian at $t=0$ and $\theta_1$
# for a data point at $t=1$.
#
# **1. Training.** Draw a pair $(\theta_1, c)$ from the data, a noise point
# $\theta_0 \sim \mathcal N(0, I)$ and a random time $t$. Place yourself on the
# straight line between $\theta_0$ and $\theta_1$ at time $t$, and regress the
# velocity onto the direction from one to the other:
#
# $$\mathcal L(\phi) = \mathbb E_{t,\, \theta_0,\, (\theta_1, c)}
#   \Big[\;\big\| \, v_\phi\big(
#   \underbrace{(1-t)\,\theta_0 + t\,\theta_1}_{\textstyle \theta_t},
#   \; t \,\big|\, c\big) \; - \; (\theta_1 - \theta_0) \, \big\|^2 \;\Big],
#   \qquad t \sim U(0,1).$$
#
# Note what is *absent*: no integration, no sampling from the model, no density,
# no Jacobian. It is a plain regression loss — Part 1's `fit` with a fancier
# target.
#
# **2. Sampling.** Draw a noise point and integrate the learned velocity field
# from $t=0$ to $t=1$:
#
# $$\theta(0) = \theta_0 \sim \mathcal N(0, I), \qquad
#   \frac{\mathrm d \theta}{\mathrm d t} = v_\phi\big(\theta(t),\, t \,|\, c\big),
#   \qquad \theta(1) \sim q_\phi(\cdot \,|\, c) .$$
#
# We integrate with plain Euler steps,
# $\theta \mathrel{+}= v_\phi(\theta, t \,|\, c)\,\Delta t$.
#
# **3. Evaluation.** If you also need the *density* of a point (we will, in
# Part 4), integrate the same ODE **backwards** from $\theta_1$ while
# accumulating the divergence of the velocity field:
#
# $$\log q_\phi(\theta_1 \,|\, c) = \log \mathcal N\big(\theta(0);\, 0, I\big)
#   \; - \; \int_0^1 \nabla \!\cdot\! v_\phi\big(\theta(t),\, t \,|\, c\big)\,
#   \mathrm d t .$$
#
# *Why* regressing onto straight lines between unrelated random pairs yields a
# flow that transports $\mathcal N(0,I)$ to $p$ is genuinely non-obvious, and we
# will not derive it — see Lipman et al. (arXiv:2210.02747), Liu et al.
# (arXiv:2209.03003) and Albergo & Vanden-Eijnden (arXiv:2209.15571). Note that
# the requirements on a density hold here *by construction*: samples come out of
# an ODE, so they are genuine draws, and $q_\phi$ is normalized because the flow
# only ever transports an already-normalized Gaussian.

# %% [markdown]

# ## Example: a spiral we can rotate and rescale

# %% [markdown]

# Our example: a spiral, with the condition $c = (\varphi, s)$ rotating it by
# $\varphi$ and scaling it by $s$,
#
# $$\varphi \sim U(0, 2\pi), \qquad s \sim U(0.6, 1.4).$$

# %%

def target_spiral(phase, scale):
    """Draw samples on a rotated, scaled spiral.

    Arguments:
      phase: the rotation angle (n, 1).
      scale: the radial scale (n, 1).

    Returns:
      Samples (n, 2), with a little Gaussian thickness added to the arm.
    """
    a = 3 * np.pi * torch.rand_like(phase).sqrt()     # angle along the arm
    r = 0.45 * a * scale                              # radius grows with angle
    ang = a + phase                                   # the rotation
    return (torch.cat([r * ang.cos(), r * ang.sin()], 1)
            + 0.12 * torch.randn(len(phase), 2, device=phase.device))


def encode(phase, scale):
    """Build the conditioning vector for the spiral.

    The phase goes in as cos/sin rather than as the angle itself, so that phase
    0 and phase 2*pi are literally the same input; feeding the raw angle would
    ask the network to learn that the ends of its input range coincide, and
    would leave a visible seam there.

    Arguments:
      phase: the rotation angle (n, 1).
      scale: the radial scale (n, 1).

    Returns:
      (cos phase, sin phase, scale) stacked into (n, 3).
    """
    return torch.cat([phase.cos(), phase.sin(), scale], 1)


N_DATA = 60000
phase_tr = 2 * np.pi * torch.rand(N_DATA, 1, device=dev)
scale_tr = 0.6 + 0.8 * torch.rand(N_DATA, 1, device=dev)
theta_tr = target_spiral(phase_tr, scale_tr)          # the samples we learn from
cond_tr = encode(phase_tr, scale_tr)                  # what we condition on

fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for a, (ph, sc) in zip(ax, [(0.0, 1.0), (2.0, 1.0), (4.0, 1.0), (0.0, 1.4)]):
    p = torch.full((3000, 1), ph, device=dev)
    s = torch.full((3000, 1), sc, device=dev)
    t = target_spiral(p, s).cpu()
    a.plot(t[:, 0], t[:, 1], 'k.', ms=1, alpha=.3)
    a.set(title=fr'$\varphi$ = {ph}, $s$ = {sc}', xlabel=r'$\theta_1$',
          aspect='equal', xlim=(-7, 7), ylim=(-7, 7))
ax[0].set_ylabel(r'$\theta_2$')
fig.suptitle('four members of the target family', y=1.02)
fig.tight_layout()

# %% [markdown]

# ## The implementation

# %% [markdown]

# Four short functions, used unchanged for the rest of the notebook.

# %%

def mlp(d_in, d_out, hidden, layers):
    """Generic MLP helper: R^d_in -> R^d_out through `layers` hidden blocks.

    The same idea as Part 1's MLP class, but with the input/output dimensions
    and the depth configurable rather than fixed at three hidden layers.

    Arguments:
      d_in, d_out: input and output dimensions.
      hidden: width of every hidden layer.
      layers: how many hidden blocks (Linear + ReLU) to stack.

    Returns:
      An nn.Sequential ending in an affine read-out with no nonlinearity.
    """
    mods, d = [], d_in                              # mods: the layer list so far
    for _ in range(layers):                         # one hidden block per layer
        mods += [nn.Linear(d, hidden), nn.ReLU()]   # affine map, then ReLU
        d = hidden                                  # next block takes `hidden` in
    return nn.Sequential(*mods, nn.Linear(d, d_out))  # read-out, no nonlinearity


def fm_loss(net, th1, cond):
    """Equation 1: the flow-matching objective. Reused in Parts 3 and 4.

    Arguments:
      net: the velocity field being trained.
      th1: the data points theta_1 (n, d_theta).
      cond: what they are conditioned on (n, d_cond).

    Returns:
      One scalar loss, ready to call .backward() on.
    """
    th0 = torch.randn_like(th1)                     # theta_0 ~ N(0, I): noise point
    t = torch.rand(len(th1), 1, device=th1.device)   # t ~ U(0, 1), one per example
    tht = (1 - t) * th0 + t * th1                   # theta_t on the straight line
    v = net(tht, t, cond)                           # velocity the network predicts
    return ((v - (th1 - th0)) ** 2).mean()          # regress onto theta_1 - theta_0


class VelocityNet(nn.Module):
    """Velocity field v(theta, t | cond): an MLP with a Fourier embedding of t.

    Calling it maps th (n, d_theta), t (n, 1) and cond (n, d_cond) to a
    velocity (n, d_theta).

    Arguments:
      d_theta: dimension of theta, and so of the velocity.
      d_cond: how many numbers we condition on.
      hidden: width of every hidden layer.
      layers: how many hidden blocks to stack.
    """

    def __init__(self, d_theta, d_cond, hidden=128, layers=3):
        super().__init__()
        self.freqs = torch.tensor([1., 2., 4., 8.])           # time-embedding freqs
        # inputs: theta (d_theta) + time embedding (9) + conditioning (d_cond)
        self.net = mlp(d_theta + 9 + d_cond, d_theta, hidden, layers)

    def forward(self, th, t, cond):
        ft = 2 * np.pi * t * self.freqs.to(t.device)          # (n, 4) scaled times
        temb = torch.cat([t, ft.sin(), ft.cos()], 1)          # (n, 9): t and 4 sin/cos
        # a raw scalar t is hard for an MLP to resolve finely; sin/cos of several
        # frequencies gives it a basis in which sharp t-dependence is easy
        return self.net(torch.cat([th, temb, cond], 1))       # -> (n, d_theta)


@torch.no_grad()                                    # sampling never needs gradients
def fm_sample(net, cond, d_theta, steps=64):
    """Equation 2: Euler-integrate dtheta/dt = v from t=0 (noise) to t=1.

    Arguments:
      net: the trained velocity field.
      cond: one conditioning row per sample wanted (n, d_cond).
      d_theta: dimension of theta.
      steps: how many Euler steps; more means a more faithful ODE solve.

    Returns:
      Samples (n, d_theta), drawn from q_phi(. | cond).
    """
    n = len(cond)                                   # one sample per conditioning row
    th = torch.randn(n, d_theta, device=cond.device)          # theta(0) ~ N(0, I)
    for i in range(steps):
        t = torch.full((n, 1), (i + 0.5) / steps, device=cond.device)  # midpoint
        th = th + net(th, t, cond) / steps          # Euler step: theta += v * dt
    return th                                       # theta(1) ~ q_phi(. | cond)

# %%

def train_fm(net, th1, cond, steps=3000, batch=512, lr=1e-3, log=True):
    """Minimize fm_loss by Adam -- the same loop as Part 1's fit().

    Arguments:
      net: the velocity field, trained in place.
      th1: the data to learn (n, d_theta).
      cond: the matching conditioning rows (n, d_cond).
      steps: how many Adam steps to take.
      batch: minibatch size, drawn fresh each step.
      lr: Adam learning rate.
      log: print the loss every 1000 steps.

    Returns:
      The same net, now trained.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)      # holds pointers to phi
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(th1), (batch,), device=th1.device)  # minibatch
        loss = fm_loss(net, th1[i], cond[i])             # forward
        opt.zero_grad(); loss.backward(); opt.step()     # zero, backward, step
        if log and (step + 1) % 1000 == 0:
            print(f'  step {step + 1}/{steps}  loss {loss.item():.3f}  '
                  f'[{time.time() - t0:.0f}s]')
    return net

# %% [markdown]

# ## Train it

# %%

snet = VelocityNet(2, d_cond=3).to(dev)             # theta is 2-D, cond is 3 numbers
train_fm(snet, theta_tr, cond_tr, steps=6000)

# %% [markdown]

# Now ask the trained network for particular conditions. The bottom-right panel
# requests a **scale of 1.9**, well outside the $[0.6, 1.4]$ it was trained on.

# %%

REQUESTS = [(0.0, 1.0), (2.0, 1.0), (4.0, 1.0),
            (1.0, 0.7), (1.0, 1.3), (1.0, 1.9)]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for a, (ph, sc) in zip(ax.ravel(), REQUESTS):
    p = torch.full((4000, 1), ph, device=dev)
    s = torch.full((4000, 1), sc, device=dev)
    truth = target_spiral(p, s).cpu()
    got = fm_sample(snet, encode(p, s), 2).cpu()
    a.plot(truth[:, 0], truth[:, 1], 'k.', ms=1, alpha=.18, label='target')
    a.plot(got[:, 0], got[:, 1], 'C0.', ms=1, alpha=.3, label='flow matching')
    outside = '  (EXTRAPOLATED)' if not 0.6 <= sc <= 1.4 else ''
    a.set(title=fr'$\varphi$ = {ph}, $s$ = {sc}{outside}', aspect='equal',
          xlim=(-9, 9), ylim=(-9, 9))
ax[0, 0].legend(markerscale=8, fontsize=8)
fig.suptitle('one network, six requested conditions', y=1.0)
fig.tight_layout()

# %% [markdown]

# **This is amortization.** The training data has a finite number of labeled spirals as examples,
# yet the network reproduces any member of the
# family on request — it learned the whole map $c \mapsto p(\theta|c)$ from a 
# handful of examples. That property is the entire reason this machinery is worth
# building for inference, where $c$ will be *data* and $\theta$ the parameters we
# want.
#
# Note that this approach generally stops working outside the training range: the extrapolated panel gets
# the rough size wrong and the structure blurred. Amortization **interpolates**; it
# does not extrapolate.

# %% [markdown]

# ### Exercise 2

# %% [markdown]

# Write your own conditional target and fit it. All you have to supply is a
# function `my_target(c)` that takes a **single float** $c \in [0,1]$ and returns
# **one** two-dimensional sample. Plain numpy is fine — no tensors, no batching,
# no device. `study_conditional(my_target)` does everything else: it draws 40000
# training samples at random $c$, fits a flow, and plots the flow against your
# target side by side at four values of $c$.
#
# The starting point in the cell below is a Gaussian blob whose centre slides
# with $c$:
#
# ```python
# def my_target(c):                      # c is a float in [0, 1]
#     return np.array([4 * c - 2, 0.0]) + 0.3 * np.random.randn(2)
# ```
#
# 1. **Your family.** Replace it with something more interesting. Ideas: two
#    blobs whose separation is $c$; a ring whose thickness is $c$; a shape that
#    changes its number of modes with $c$; letters morphing into each other.
# 2. **Unseen conditions.** Ask for conditions outside the training range, e.g.
#    `study_conditional(my_target, requests=[0.0, 0.5, 1.0, 1.4])`. Does your
#    family extrapolate as badly as the spiral's scale did? Families where $c$
#    only rescales things often survive; ones where $c$ changes the *structure*
#    do not.
# 3. **The ODE knob.** Pass `sample_steps=1, 4, 16`. How many Euler steps do you
#    need? What does `sample_steps=1` correspond to geometrically?
# 4. **Training budget.** Halve `steps` (the number of Adam steps), or drop
#    `n_data` to 5000. Which part of the picture degrades first — the shape, or
#    its dependence on $c$?

# %%

def study_conditional(my_target, n_data=40000, steps=4000,
                      requests=(0.0, 0.33, 0.67, 1.0), sample_steps=64):
    """Fit a conditional flow to my_target and plot it against the target.

    Arguments:
      my_target: takes one float c in [0, 1] and returns one 2-D sample (any
        array-like of length two -- a numpy array, a list, a tuple). We call it
        once per example in a plain Python loop, which costs a second or two
        for 40000 samples and is nothing next to the training that follows.
      n_data: how many training samples to draw.
      steps: how many Adam steps to take.
      requests: the values of c to plot; anything outside [0, 1] is
        extrapolation, and is flagged as such in the panel title.
      sample_steps: Euler steps used when sampling the fitted flow.

    Returns:
      The trained VelocityNet.
    """
    def draw(c):
        """Call my_target for every row of c (n, 1) -> samples (n, 2)."""
        th = np.stack([np.asarray(my_target(cv), dtype=float)
                       for cv in c.flatten().tolist()])
        return torch.tensor(th, dtype=torch.float32, device=dev)

    c = torch.rand(n_data, 1, device=dev)          # training conditions ~ U(0, 1)
    th = draw(c)
    net = VelocityNet(2, d_cond=1).to(dev)
    train_fm(net, th, c, steps=steps, log=False)

    lim = 1.15 * th.abs().max().item()             # one shared scale for all panels
    fig, ax = plt.subplots(1, len(requests), figsize=(3.4 * len(requests), 3.6))
    for a, cv in zip(np.atleast_1d(ax), requests):
        cc = torch.full((3000, 1), float(cv), device=dev)
        tr, got = draw(cc).cpu(), fm_sample(net, cc, 2, steps=sample_steps).cpu()
        a.plot(tr[:, 0], tr[:, 1], 'k.', ms=1, alpha=.2, label='target')
        a.plot(got[:, 0], got[:, 1], 'C0.', ms=1, alpha=.3, label='flow')
        tag = '' if 0.0 <= cv <= 1.0 else '  (EXTRAP.)'
        a.set(title=f'c = {cv:.2f}{tag}', aspect='equal',
              xlim=(-lim, lim), ylim=(-lim, lim))
    np.atleast_1d(ax)[0].legend(markerscale=8, fontsize=8)
    fig.tight_layout()
    return net

# %%

# your code goes here
#
# my_target takes one float c in [0, 1] and returns one 2-D sample. Below: a
# Gaussian blob whose centre slides with c. Replace it.
def my_target(c):
    """A Gaussian blob whose centre slides with c: float in [0, 1] -> one 2-D sample."""
    return np.array([4 * c - 2, 0.0]) + 0.3 * np.random.randn(2)


study_conditional(my_target)

# %%

# @title Reference solution { display-mode: "form" }
def my_target(c):                                   # noqa: F811
    """A banana that rotates with c: structure changes, so extrapolation fails."""
    u = np.random.uniform(-3, 3)                              # along the arc
    v = 0.3 * u ** 2 - 1.2 + 0.25 * np.random.randn()         # bend it
    ang = 2 * np.pi * c                                       # the rotation
    return [u * np.cos(ang) - v * np.sin(ang),
            u * np.sin(ang) + v * np.cos(ang)]


study_conditional(my_target, requests=[0.0, 0.25, 0.5, 1.4])

# %% [markdown]

# **What to notice.** The rotating banana is learned cleanly inside its range,
# and the last panel — $c = 1.4$, well past the training range of $[0,1]$ — is
# wrong in a revealing way: a rotation is not a rescaling, so there is nothing
# for the network to extrapolate along and it falls back on something smooth and
# incorrect. Compare that with the spiral's scale, which extrapolated tolerably.
# Whether amortization survives outside its training range depends on whether $c$
# changes the *structure* of the distribution or merely its size.

# %% [markdown]

# ---
# # Part 3 — From generative models to inference: SBI
#
# We now arrive at a modern way of doing simulation-based inference (SBI) with
# neural networks: **take conditional flow matching and feed it pairs from a
# simulator.**
#
# In Part 2 the pairs $(\theta_1, c)$ were the outputs and inputs of a stochastic
# simulator — your `my_target` — and the network learned to sample
# $p(\theta_1 \,|\, c)$. For inference we manufacture the pairs differently: draw
# parameters from a prior, push them through a simulator, and let the *data* play
# the role of the condition,
#
# $$\theta_i \sim p(\theta), \qquad x_i \sim p(x \,|\, \theta_i), \qquad c = x .$$
#
# Each pair is a draw from the joint distribution, and the joint factorizes two
# ways — which is all Bayes' theorem is:
#
# $$\underbrace{p(x \,|\, \theta)\,p(\theta)}_{\textstyle \text{what we can sample}}
#   \;=\; p(\theta, x) \;=\;
#   \underbrace{p(\theta \,|\, x)\,p(x)}_{\textstyle \text{what we want}}
#   \qquad\Longleftrightarrow\qquad
#   p(\theta \,|\, x) = \frac{p(x \,|\, \theta)\,p(\theta)}{p(x)} .$$
#
# We can only sample the left-hand side; a network trained on those samples hands
# us the right-hand one, $q_\phi(\theta \,|\, x) \approx p(\theta \,|\, x)$. No
# likelihood evaluation, no MCMC, no theorem applied by hand — it is enforced
# purely by where the training pairs come from. And because the model is
# amortized in $c = x$, one training run gives the posterior for *any*
# observation.
#
# ## The simulator: throwing a ball
#
# A ball is launched from the ground at angle $\alpha$ with speed $v$ on a flat
# planet, and we measure only **where it lands**:
#
# $$x = \frac{v^2}{g}\,\sin(2\alpha) + \sigma\,\varepsilon, \qquad
#   \varepsilon \sim \mathcal N(0, 1).$$
#
# We want $\theta = (v, \alpha)$ out of that single number, so the problem is
# degenerate by construction — it is
# whatever the curve $v^2 \sin(2\alpha) = \text{const}$ happens to look like: a
# **curved ridge**, folded back on itself at $45°$, because a throw at $60°$ and a
# throw at $30°$ land in exactly the same place. The two throws below are
# indistinguishable from the landing point alone.

# %%

G = 9.81
SIGMA_X = 0.4                                  # measurement error on the range [m]
V_LO, V_HI = 8.0, 12.0                         # the prior box: v [m/s] ...
A_LO, A_HI = 0.15, np.pi / 2 - 0.15            # ... and alpha [rad]


def ball_prior():
    """Draw one theta = (v, alpha), uniform in the prior box."""
    return [np.random.uniform(V_LO, V_HI), np.random.uniform(A_LO, A_HI)]


def ball_sim(theta, n_throws=1):
    """Simulate one throw, measuring only where the ball lands.

    Arguments:
      theta: the parameters (v, alpha), speed in m/s and angle in rad.
      n_throws: average this many repeats of the same throw, which shrinks the
        measurement error to sigma / sqrt(n_throws). The physics itself is the
        single line for r.

    Returns:
      [range], a one-element list, since x is one-dimensional here.
    """
    v, alpha = theta
    r = v ** 2 / G * np.sin(2 * alpha)                       # where it lands
    return [r + SIGMA_X * np.random.randn() / np.sqrt(n_throws)]


def ball_path(theta, n=200):
    """Compute a flight path, for the picture only.

    Arguments:
      theta: the parameters (v, alpha).
      n: how many points to sample along the trajectory.

    Returns:
      (n, 2) array of (x, y) positions, from launch to landing.
    """
    v, alpha = theta
    t = np.linspace(0, 1, n) * 2 * v * np.sin(alpha) / G
    return np.stack([v * np.cos(alpha) * t,
                     v * np.sin(alpha) * t - 0.5 * G * t ** 2], -1)


def ridge(x_obs, v):
    """The degeneracy: the angles that land a ball of speed v at range x_obs.

    Arguments:
      x_obs: the observed range, a scalar.
      v: the speeds to solve at, an array.

    Returns:
      (a, pi/2 - a): the line drive and the lob, nan at speeds too slow to
      reach x_obs at all.
    """
    s = G * x_obs / v ** 2                     # = sin(2 alpha); > 1 is unreachable
    a = 0.5 * np.arcsin(np.where(s > 1, np.nan, s))
    return a, np.pi / 2 - a                    # the line drive and the lob


def plot_ridge(ax, x_obs, label=None):
    """Draw the degeneracy curve on a (v, alpha) plane, and set up the axes.

    Arguments:
      ax: the axes to draw on.
      x_obs: the observed range, a scalar.
      label: legend label for the curve, or None to leave it unlabelled.
    """
    v = np.linspace(V_LO, V_HI, 200)
    for i, a in enumerate(ridge(x_obs, v)):
        ax.plot(v, np.degrees(a), 'k--', lw=1, label=label if i == 0 else None)
    ax.set(xlabel='v [m/s]', ylabel=r'$\alpha$ [deg]', xlim=(V_LO, V_HI),
           ylim=(np.degrees(A_LO), np.degrees(A_HI)))


THETA_BALL = [10.4, 0.62]                       # v = 10.4 m/s, alpha = 36 deg
THETA_BALL_TWIN = [10.4, np.pi / 2 - 0.62]      # the 54 deg lob that lands there too

fig, ax = plt.subplots(figsize=(6.5, 3.0))
for th, c in zip([THETA_BALL, THETA_BALL_TWIN], ['C0', 'C3']):
    p = ball_path(th)
    ax.plot(p[:, 0], p[:, 1], c, lw=1.5,
            label=fr'$v$ = {th[0]:.1f} m/s, '
                  fr'$\alpha$ = {np.degrees(th[1]):.0f}$^\circ$')
ax.axvline(ball_path(THETA_BALL)[-1, 0], color='k', ls=':', lw=1)
ax.set(xlabel='distance [m]', ylabel='height [m]', ylim=(0, None))
ax.legend(fontsize=8, title='same landing point', title_fontsize=8)
fig.tight_layout()

# %% [markdown]

# ## Train — with nothing new to write
#
# `VelocityNet`, `fm_loss`, `train_fm` and `fm_sample` are the Part 2
# functions, untouched. The only difference is that `cond` is now the output of
# a simulator, so two short wrappers are all we add:
#
# - `train_sbi(prior, simulator)` simulates a training set and fits
#   $q_\phi(\theta \,|\, x)$, returning the trained network;
# - `sample_posterior(net, x_obs)` draws from that network for one observation.
#
# You supply the *prior* and the *simulator*, one draw at a time and in plain
# numpy — exactly like `my_target` in Part 2. Everything else, including the
# dimensions of $\theta$ and $x$, is worked out from what they return.
#
# We do it twice: once with a **single** throw, and once with the mean of
# **twenty** throws (which shrinks the noise on the measurement to
# $\sigma/\sqrt{20}$).

# %%

def train_sbi(prior, simulator, n_sim=40000, steps=3000, hidden=128, layers=3):
    """Fit q(theta | x) from a prior and a simulator.

    The dimensions of theta and x are worked out from one trial draw, so
    nothing here has to be told them.

    Arguments:
      prior: called with no arguments, returns one theta vector (any
        array-like of length d_theta).
      simulator: called with one theta, returns one x vector (length d_x).
        Both are plain Python/numpy -- no tensors, no batching, no device. We
        call them once per example in a loop, which costs a second or two for
        the default 40000 pairs and is nothing next to the training that
        follows.
      n_sim: how many (theta, x) pairs to simulate.
      steps: how many Adam steps to take.
      hidden, layers: size of the velocity network.

    Returns:
      The trained VelocityNet, carrying the z-scoring constants as buffers so
      that sample_posterior can undo them.
    """
    theta_np = np.stack([np.asarray(prior(), dtype=float) for _ in range(n_sim)])
    x_np = np.stack([np.asarray(simulator(th), dtype=float) for th in theta_np])
    theta = torch.tensor(theta_np, dtype=torch.float32, device=dev)  # (n, d_theta)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)          # (n, d_x)
    print(f'  {n_sim} pairs simulated: theta is {theta.shape[1]}-D, '
          f'x is {x.shape[1]}-D')

    net = VelocityNet(theta.shape[1], x.shape[1], hidden, layers).to(dev)
    # The z-scoring constants are part of the trained model, so keep them ON the
    # net: registered buffers travel with .to(dev) and with state_dict(), and
    # sample_posterior can then undo the scaling without being handed anything.
    net.register_buffer('t_mu', theta.mean(0)); net.register_buffer('t_sd', theta.std(0))
    net.register_buffer('x_mu', x.mean(0)); net.register_buffer('x_sd', x.std(0))

    train_fm(net, (theta - net.t_mu) / net.t_sd,      # z-score both sides, so
             (x - net.x_mu) / net.x_sd, steps=steps)  # the net sees O(1) numbers
    return net


def sample_posterior(net, x_obs, n_samples=6000):
    """Draw posterior samples for one observation.

    Arguments:
      net: a network returned by train_sbi.
      x_obs: the observation, one x vector of length d_x.
      n_samples: how many posterior samples to draw.

    Returns:
      (n_samples, d_theta) numpy array of draws from q(theta | x_obs).
    """
    xo = torch.tensor(np.asarray(x_obs, dtype=float), dtype=torch.float32, device=dev)
    cond = ((xo - net.x_mu) / net.x_sd).expand(n_samples, len(net.x_mu))
    theta = fm_sample(net, cond, len(net.t_mu)) * net.t_sd + net.t_mu  # un-z-score
    return theta.cpu().numpy()


def plot_ball_posterior(ax, post, x_obs, truth=None, title=None,
                        ridge_label=None, post_label=None):
    """Draw a posterior scatter on the (v, alpha) plane, over the degeneracy curve.

    Arguments:
      ax: the axes to draw on.
      post: posterior samples (n, 2), as returned by sample_posterior.
      x_obs: the observation those samples belong to.
      truth: the true (v, alpha) to mark with a star, or None to leave it out.
      title: optional axes title.
      ridge_label, post_label: legend labels, or None to leave them out.
    """
    plot_ridge(ax, np.asarray(x_obs, dtype=float)[0], label=ridge_label)
    ax.plot(post[:, 0], np.degrees(post[:, 1]), 'C0.', ms=1.5, alpha=.25,
            label=post_label)
    if truth is not None:
        ax.plot(truth[0], np.degrees(truth[1]), 'r*', ms=15)
    if title:
        ax.set_title(title)

# %%

print('one throw:')
net_1 = train_sbi(ball_prior, ball_sim)
print('twenty throws:')
net_20 = train_sbi(ball_prior, lambda th: ball_sim(th, n_throws=20))

# %%

x_obs_1 = ball_sim(THETA_BALL)                        # our observation ...
x_obs_20 = ball_sim(THETA_BALL, n_throws=20)          # ... and a better one

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for a, net, xo, nt in [(ax[0], net_1, x_obs_1, 1), (ax[1], net_20, x_obs_20, 20)]:
    plot_ball_posterior(a, sample_posterior(net, xo), xo, truth=THETA_BALL,
                        title=f'{nt} throw{"s" if nt > 1 else ""}:  x = {xo[0]:.2f} m')
ax[1].set_ylabel('')
fig.tight_layout()

# %% [markdown]

# **What the posterior looks like.** A curved ridge lying along the
# dashed line — every $(v, \alpha)$ that lands the ball where we saw it — folded
# back on itself about $45°$, where the two arms meet at the slowest speed that
# can reach this far, $v = \sqrt{g\,x}$. Read it along a vertical line and it is
# **bimodal**: any speed above that minimum has two viable angles, one lob and
# one line drive.  The dashed curve is drawn from the
# formula only so that we can see it got the shape right.
#
# **What more data does, and does not do.** Twenty throws instead of one
# shrinks the measurement noise by $\sqrt{20}$ and the arms become correspondingly
# thinner. But they do not merge, and the second mode does not go away.
#
# **Exercise 3.**
# 1. **Amortization.** `net_1` already covers *every* $x$, not just ours. Call
#    `sample_posterior(net_1, [x])` for two or three different observed ranges,
#    with no retraining — try a near-maximal range
#    ($x \approx v^2/g \approx 11$ m). What happens to the two modes as the
#    range approaches the largest achievable one?
# 2. **Break the degeneracy.** Write a simulator that returns *two* numbers, the
#    range **and** the time of flight $2 v \sin\alpha / g$, and hand it to
#    `train_sbi` — the extra dimension is picked up on its own. What happens to
#    the second mode? Why?

# %%

# your code goes here


# %%

# @title Reference solution { display-mode: "form" }
# 1: amortization -- the SAME network, three observations, no retraining.
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
for a, xo in zip(ax, [[6.0], [9.5], [11.5]]):
    plot_ball_posterior(a, sample_posterior(net_1, xo), xo, title=f'x = {xo[0]} m')
for a in ax[1:]:
    a.set_ylabel('')
fig.tight_layout()

# %%

# 2: adding the time of flight kills the second mode. Only the simulator changes.
def ball_sim2(theta):
    """Simulate one throw, measuring the range AND the time of flight.

    Arguments:
      theta: the parameters (v, alpha).

    Returns:
      [range, time of flight] -- x has two components now, which is all
      train_sbi needs to notice the change.
    """
    v, alpha = theta
    return [v ** 2 / G * np.sin(2 * alpha) + SIGMA_X * np.random.randn(),
            2 * v * np.sin(alpha) / G + 0.02 * np.random.randn()]


net_2obs = train_sbi(ball_prior, ball_sim2, steps=6000)   # small posterior: train longer
x_obs2 = ball_sim2(THETA_BALL)
post2 = sample_posterior(net_2obs, x_obs2)

fig, ax = plt.subplots(figsize=(5.8, 4.6))
plot_ball_posterior(ax, post2, x_obs2, truth=THETA_BALL, ridge_label='range alone',
                    post_label='posterior from both')
# the other observable has its own degeneracy: T = 2 v sin(alpha) / g = const
vv = np.linspace(V_LO, V_HI, 200)
sa = G * x_obs2[1] / (2 * vv)                             # = sin(alpha)
ax.plot(vv, np.degrees(np.arcsin(np.where(sa > 1, np.nan, sa))), 'k:', lw=1.4,
        label='time of flight alone')
ax.legend(fontsize=8, markerscale=6, loc='upper right')
ax.set_title('two observables: the posterior sits where the curves cross')
fig.tight_layout()

# %% [markdown]

# **Answers.** (1) As the observed range approaches the maximum achievable one,
# $v^2/g$ at $\alpha = 45°$, the two arms squeeze together and merge: only
# throws near $45°$ can reach that far, so the ambiguity disappears and the
# posterior becomes a single blob pinned to the corner of the prior. (2) The
# time of flight depends on $\sin\alpha$, not $\sin 2\alpha$, and is therefore
# *not* symmetric about $45°$ — one extra number breaks the reflection and one
# mode dies. Each observable has its own degeneracy curve, and the posterior
# collapses onto where the two cross; note that it does **not** follow either
# curve, and that it lies closer to the time-of-flight one, which is the sharper
# measurement here. Choosing what to measure is inference design, and it beats
# any amount of network tuning.

# %% [markdown]

# ---
# # Part 4 — A toy gravitational wave: compression + sequential zoom
#
# Now let us look at a long noisy time series containing a
# chirp,
# $$d(t) = A\sin\!\big(2\pi(f_0 t + \tfrac12 \dot f t^2) + \varphi\big)
#          + n(t), \qquad n\sim\mathcal N(0,1),$$
# with 1024 samples. We infer here two parameters, $(f_0, \dot f)$; the phase $\varphi$ is a
# **nuisance parameter** — we randomize it in training and never infer it
# (exactly how the real LISA analysis treats its gauge angles).
#
# Two new problems appear:
# 1. $x$ has 1024 dimensions — we need to **compress** before conditioning. We do that here classically with PCA.
# 2. With many cycles the likelihood is razor-sharp: when trained on a small
#    number of simulations, an amortized net trained
#    from the wide prior turns out too blurry. We fix that by **zooming in sequentially** (this is sequential SBI).  Sequential SBI
#    leads to results that are not amortized but focused on the region of interest, reducing the number of required simulator calls.

# %%

N_T = 1024
tgrid = torch.linspace(0, 1, N_T, device=dev)
PRIOR_LO = torch.tensor([40., 0.], device=dev)      # f0 [cycles], fdot
PRIOR_HI = torch.tensor([80., 40.], device=dev)
AMP = 1.0
THETA_CHIRP = torch.tensor([55.3, 17.8], device=dev)


def chirp_sim(theta, noise=1.0, phi=None):
    """Simulate noisy chirps.

    Arguments:
      theta: the parameters (f0, fdot) per row, (n, 2).
      noise: standard deviation of the white noise added; 0 gives clean signals.
      phi: the nuisance phase (n, 1), randomized uniformly when left as None.
        Randomizing it in training is how the network learns to marginalize
        over a parameter we never infer.

    Returns:
      The data (n, 1024).
    """
    if phi is None:
        phi = torch.rand(len(theta), 1, device=theta.device) * 2 * np.pi
    phase = 2 * np.pi * (theta[:, :1] * tgrid + 0.5 * theta[:, 1:2] * tgrid ** 2)
    return AMP * torch.sin(phase + phi) + noise * torch.randn(len(theta), N_T,
                                                              device=theta.device)


torch.manual_seed(7)
x_obs_chirp = chirp_sim(THETA_CHIRP[None], phi=torch.tensor([[2.1]], device=dev))
snr = AMP * np.sqrt(N_T / 2)
print(f'signal-to-noise ratio ~ {snr:.0f}')

fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(tgrid.cpu(), x_obs_chirp[0].cpu(), lw=.5, label='observed (signal+noise)')
ax.plot(tgrid.cpu(), chirp_sim(THETA_CHIRP[None], noise=0,
                               phi=torch.tensor([[2.1]], device=dev))[0].cpu(),
        'C1', lw=1, label='hidden signal')
ax.legend(loc='upper right'); ax.set(xlabel='t', ylabel='d(t)')
fig.tight_layout()

# %% [markdown]

# ## Step 1: compression with PCA
#
# We can't feed 1024 numbers into the conditioning — most of them are noise.
# To isolate the signal, we simulate *clean* signals from the prior, and find the directions in which
# they actually vary (principal component analysis = an SVD). The singular
# values tell us each component's signal-to-noise; we keep the top $K$.

# %%

def fit_pca(theta_bank, K=64):
    """Find the directions in which CLEAN signals from theta_bank actually vary.

    Arguments:
      theta_bank: the parameters to simulate signals for, (n, 2).
      K: how many principal directions to keep.

    Returns:
      (mu, V, eigs): the mean waveform (N_T,), the top-K principal directions
      V (K, N_T) to project onto, and the singular values rescaled into a
      per-component signal-to-noise -- all of them, not just K, since the
      spectrum plot wants the tail.
    """
    clean = chirp_sim(theta_bank, noise=0.0)
    mu = clean.mean(0)
    U, S, Vh = torch.linalg.svd(clean - mu, full_matrices=False)
    eigs = S / np.sqrt(len(clean) - 1)              # per-component SNR
    return mu, Vh[:K], eigs


def draw_prior(n):
    """The prior: n draws of theta = (f0, fdot), uniform in the box -> (n, 2)."""
    return PRIOR_LO + (PRIOR_HI - PRIOR_LO) * torch.rand(n, 2, device=dev)


N_BANK, K_PCA = 32768, 64          # amortized-run budget and summary count
torch.manual_seed(1)
theta_bank = draw_prior(N_BANK)                 # one bank: PCA basis and training
mu0, V0, eigs0 = fit_pca(theta_bank, K=K_PCA)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.semilogy(eigs0.cpu()[:200])
ax.axhline(1, color='r', ls='--', lw=1, label='noise level')
ax.axvline(K_PCA, color='k', ls=':', lw=1, label=f'K = {K_PCA}')
ax.set(xlabel='PCA component', ylabel='component SNR',
       title='wide prior: signal variance spread over MANY components')
ax.legend()
fig.tight_layout()

# %% [markdown]

# Note how *flat* that spectrum is: at the wide prior, chirps with different
# $(f_0,\dot f)$ are nearly orthogonal waveforms, so no small linear basis
# captures them all (for the real MBHB prior it takes ~2000 components!).
# Keep this plot in mind — it will look completely different after zooming.
#
# ## Step 2: amortized SBI from the wide prior

# %%

def summarize(x, mu, V):
    """Compress data onto the PCA basis.

    Arguments:
      x: the data to compress, (n, N_T).
      mu, V: the mean waveform and basis returned by fit_pca.

    Returns:
      The summaries (n, K) that the flow conditions on, instead of all N_T
      numbers.
    """
    return (x - mu) @ V.T


x_bank = chirp_sim(theta_bank)                  # add noise: this is the training data

s_bank = summarize(x_bank, mu0, V0)
s_mu, s_sd = s_bank.mean(0), s_bank.std(0) + 1e-6
th_mu, th_sd = theta_bank.mean(0), theta_bank.std(0)

cnet = VelocityNet(2, K_PCA).to(dev)
train_fm(cnet, (theta_bank - th_mu) / th_sd,     # z-score both sides, as always
         (s_bank - s_mu) / s_sd, steps=6000)

s_obs = (summarize(x_obs_chirp, mu0, V0) - s_mu) / s_sd
post0 = fm_sample(cnet, s_obs.expand(4000, K_PCA), 2) * th_sd + th_mu

# %%

def chirp_true_logpost(f0g, fdg, x_obs):
    """Evaluate the exact posterior on a grid, for comparison with the flow.

    The nuisance phase is marginalized out: the signal is linear in
    (cos phi, sin phi), so a two-basis matched filter does it.

    Arguments:
      f0g, fdg: the grid axes in f0 and fdot.
      x_obs: the observed data (1, 1024).

    Returns:
      The unnormalized log-posterior on the grid, (len(f0g), len(fdg)).
    """
    F0, FD = torch.meshgrid(f0g, fdg, indexing='ij')
    th = torch.stack([F0.ravel(), FD.ravel()], 1)
    phase = 2 * np.pi * (th[:, :1] * tgrid + 0.5 * th[:, 1:2] * tgrid ** 2)
    bs, bc = AMP * torch.sin(phase), AMP * torch.cos(phase)   # phi=0 and phi=pi/2 bases
    d = x_obs[0]
    # logL(phi) = d.h - |h|^2/2 with h = bs cos(phi) + bc sin(phi);
    # marginalize phi on a fine grid (analytic Bessel form exists; grid is clearer)
    phis = torch.linspace(0, 2 * np.pi, 64, device=dev)[:, None, None]
    h_d = (bs @ d) * phis.cos()[:, :, 0] + (bc @ d) * phis.sin()[:, :, 0]
    hh = ((bs ** 2).sum(1) * phis.cos()[:, :, 0] ** 2
          + (bc ** 2).sum(1) * phis.sin()[:, :, 0] ** 2
          + 2 * (bs * bc).sum(1) * phis.cos()[:, :, 0] * phis.sin()[:, :, 0])
    logL = h_d - 0.5 * hh
    return torch.logsumexp(logL, 0).reshape(len(f0g), len(fdg))


F0_LO, F0_HI = 54.95, 55.80        # a few sigma around the exact posterior
FD_LO, FD_HI = 16.85, 18.45
f0g = torch.linspace(F0_LO, F0_HI, 160, device=dev)
fdg = torch.linspace(FD_LO, FD_HI, 160, device=dev)
lp = chirp_true_logpost(f0g, fdg, x_obs_chirp).cpu()

fig, ax = plt.subplots(figsize=(5.5, 4.4))
ax.plot(post0.cpu()[:, 0], post0.cpu()[:, 1], 'C0.', ms=2, alpha=.3,
        label='amortized posterior')
p = (lp - lp.max()).exp()
# the exact posterior is far too small to see at this scale -- box it instead
ax.add_patch(plt.Rectangle((F0_LO, FD_LO), F0_HI - F0_LO, FD_HI - FD_LO,
                           fill=False, ec='k', lw=1.5,
                           label='exact posterior (inside this box)'))
ax.plot(*THETA_CHIRP.cpu(), 'r*', ms=14, label='truth')
ax.set(xlabel=r'$f_0$', ylabel=r'$\dot f$', xlim=(48, 64), ylim=(8, 28))
ax.legend(fontsize=8)
ax.set_title('amortized: roughly the right place, far too blurry')
fig.tight_layout()

# %% [markdown]

# The network found the right region, and it does contain the truth — but it is
# **enormously** wider than the true posterior, which fits entirely inside that
# little black box (we can compute it exactly here, a luxury the real problem
# doesn't offer). The blur is about a factor of ten in each direction.
#
# It is also *unreliable*: re-run this with a different seed and the amortized
# posterior sometimes drifts a couple of its own widths off the truth, or misses
# it altogether. Neither the blur nor the wandering is a network-capacity
# problem. Count the training samples that land inside those contours:

# %%

inside = ((theta_bank[:, 0] > F0_LO) & (theta_bank[:, 0] < F0_HI)
          & (theta_bank[:, 1] > FD_LO) & (theta_bank[:, 1] < FD_HI))
print(f'training samples in the posterior neighbourhood: {inside.sum().item()} / {len(theta_bank)}')

# %% [markdown]

# **Sample starvation:** the posterior occupies a tiny fraction of the prior
# volume, so a couple of dozen of the 32768 training examples land where the
# answer lives — and the network is effectively interpolating between them.
# That is
# why the result is both broad and jumpy from run to run. More capacity cannot
# fix having no data. The fix is to *move the training
# distribution*: simulate where the current posterior estimate points,
# retrain, repeat — each **round** zooms further in. The training buffer
# converges to a **tempered posterior** $\propto L^\gamma \pi$ ($\gamma<1$
# keeps it a bit wider than the posterior for safety; see the Dynamic SBI
# paper, arXiv:2510.13997).
#
# To decide which proposed samples to keep we need importance weights, i.e.
# the *density* of the flow — obtained by integrating the ODE backwards while
# accumulating its divergence.

# %%

def fm_logprob(net, w1, cond, steps=64):
    """Equation 3: evaluate log q(w1 | cond) by integrating the ODE backwards.

    Exact rather than estimated: the divergence is accumulated with one
    autograd call per dimension of theta.

    Arguments:
      net: the trained velocity field.
      w1: the points to evaluate the density at (n, d_theta).
      cond: the matching conditioning rows (n, d_cond).
      steps: how many Euler steps in the reverse solve.

    Returns:
      log q(w1 | cond), shape (n,).
    """
    w = w1.clone()
    n_dim = w1.shape[1]
    logdet = torch.zeros(len(w1), device=w1.device)
    for i in range(steps):
        t = torch.full((len(w1), 1), 1 - (i + 0.5) / steps, device=w1.device)
        with torch.enable_grad():
            wg = w.requires_grad_(True)
            v = net(wg, t, cond)
            # keep the graph alive for every dimension but the last one
            div = sum(torch.autograd.grad(v[:, d].sum(), wg,
                                          retain_graph=(d < n_dim - 1))[0][:, d]
                      for d in range(n_dim))
        w = (w - v / steps).detach()
        logdet = logdet - div.detach() / steps
    base = (-0.5 * (w ** 2).sum(1) - 0.5 * w.shape[1] * np.log(2 * np.pi))
    return base + logdet

# %% [markdown]

# ### How each round proposes
#
# Draw from a 50/50 mixture of the two flows, $q_c(\theta \,|\, s_{\rm obs})$ and
# $q_m(\theta)$ — the conditional zooms in, the marginal keeps a safety net — and
# reweight those draws toward the **tempered** posterior $L^\gamma \pi$. The
# likelihood is never evaluated: both flows are trained on the same buffer, so
# $q_c \propto L\,q_m$ and the *ratio* is the likelihood. Up to constants,
#
# $$\log w \;=\; \gamma\,\big(\log q_c - \log q_m\big) \;-\; \log\big(q_c + q_m\big),$$
#
# set to $-\infty$ outside the prior box. The top `n_keep` weights — jittered by
# Gumbel noise, so this draws rather than takes the mode — are simulated and
# become the next buffer. $\gamma < 1$ keeps that buffer a little wider than the
# posterior itself.

# %%

def sequential_chirp(n_rounds=10, gamma=0.3, n_keep=1024, refit_pca=True,
                     loss_fn=fm_loss, verbose=True, x_obs=x_obs_chirp):
    """Zoom in on the posterior by moving the training distribution, round by round.

    Each round: refit the PCA basis and the z-scores on the current buffer,
    continue training two warm-started flows -- the conditional q_c(theta|s)
    and the marginal q_m(theta) -- propose from a 50/50 mixture of the two,
    weight the proposals toward the tempered posterior L^gamma * prior (the
    ratio q_c/q_m stands in for the likelihood), keep n_keep of them without
    replacement, simulate those, and refresh the buffer.

    Arguments:
      n_rounds: how many zoom rounds to run.
      gamma: the tempering exponent. Larger zooms faster but leaves less
        safety margin if an early round's estimate excludes the truth.
      n_keep: how many proposals survive each round and get simulated.
      refit_pca: refit the compression basis every round, or freeze round 1's.
      loss_fn: the training objective, exposed so it can be swapped out.
      verbose: print the buffer and posterior widths each round.
      x_obs: the observation to zoom in on, (1, 1024).

    Returns:
      (posts, spectra): the posterior samples read out at the end of each
      round, and each round's PCA singular-value spectrum.
    """
    torch.manual_seed(1)
    buf_theta = draw_prior(4096)
    buf_x = chirp_sim(buf_theta)
    posts, spectra = [], []
    # warm-started nets: keep training the SAME networks across rounds (this is
    # what production codes do; retraining from scratch each round underfits)
    qc, qm = VelocityNet(2, K_PCA).to(dev), VelocityNet(2, K_PCA).to(dev)
    opt_c = torch.optim.Adam(qc.parameters(), lr=2e-3)
    opt_m = torch.optim.Adam(qm.parameters(), lr=2e-3)
    for round_ in range(1, n_rounds + 1):
        # -- gauges: PCA refit on the CURRENT buffer scale + z-scores
        if refit_pca or round_ == 1:
            mu, V, eigs = fit_pca(buf_theta)
        spectra.append(eigs.cpu())
        s = summarize(buf_x, mu, V)
        smu, ssd = s.mean(0), s.std(0) + 1e-6
        tmu, tsd = buf_theta.mean(0), buf_theta.std(0)
        w1 = (buf_theta - tmu) / tsd                     # z-scored parameters
        sc = (s - smu) / ssd                             # z-scored summaries
        so = (summarize(x_obs, mu, V) - smu) / ssd       # ... and the observation
        # -- continue training conditional q_c(theta|s) and marginal q_m(theta)
        for net, opt, cond in [(qc, opt_c, sc), (qm, opt_m, torch.zeros_like(sc))]:
            for step in range(500):
                i = torch.randint(0, len(w1), (256,), device=dev)
                loss = loss_fn(net, w1[i], cond[i])
                opt.zero_grad(); loss.backward(); opt.step()
        # -- propose from a 50/50 mixture, weight toward L^gamma * prior
        n_prop = 4096
        wp = torch.cat([fm_sample(qc, so.expand(n_prop // 2, K_PCA), 2),
                        fm_sample(qm, torch.zeros(n_prop // 2, K_PCA, device=dev), 2)])
        lqc = fm_logprob(qc, wp, so.expand(n_prop, K_PCA))
        lqm = fm_logprob(qm, wp, torch.zeros(n_prop, K_PCA, device=dev))
        th_p = wp * tsd + tmu
        in_prior = ((th_p > PRIOR_LO) & (th_p < PRIOR_HI)).all(1)
        logw = gamma * (lqc - lqm) - torch.logaddexp(lqc, lqm)   # + log(flat prior)
        logw[~in_prior] = -torch.inf
        logw[~torch.isfinite(logw)] = -torch.inf                 # numerical guard
        # -- keep n_keep WITHOUT replacement (Gumbel top-k), simulate, refresh buffer
        gum = -torch.log(-torch.log(torch.rand_like(logw)))
        keep = torch.topk(logw + gum, n_keep).indices
        new_theta = th_p[keep]
        buf_theta = torch.cat([new_theta, buf_theta])[:4096]
        buf_x = torch.cat([chirp_sim(new_theta), buf_x])[:4096]
        # -- posterior readout at gamma=1 for the plot
        post = fm_sample(qc, so.expand(4000, K_PCA), 2) * tsd + tmu
        posts.append(post.cpu())
        if verbose:
            print(f'round {round_}: buffer f0 std {buf_theta[:, 0].std():.3f}, '
                  f'posterior f0 std {post[:, 0].std():.3f}')
    return posts, spectra


posts, spectra = sequential_chirp()

# %%

fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
colors = plt.cm.viridis(np.linspace(0, .9, len(posts)))
for r, (post, c) in enumerate(zip(posts, colors), 1):
    for a in ax[:2]:
        a.plot(post[:, 0], post[:, 1], '.', ms=1.5, alpha=.25, color=c,
               label=f'round {r}' if a is ax[0] else None)
for a in ax[:2]:
    a.contour(f0g.cpu(), fdg.cpu(), p.T, levels=[0.011, 0.61], colors='k',
              linewidths=1)
    a.plot(*THETA_CHIRP.cpu(), 'r*', ms=14)
    a.set(xlabel=r'$f_0$', ylabel=r'$\dot f$')
ax[0].set(xlim=(40, 80), ylim=(0, 40), title='the zoom trajectory')
ax[1].set(xlim=(F0_LO, F0_HI), ylim=(FD_LO, FD_HI),
          title='late rounds vs exact posterior (black)')
ax[0].legend(markerscale=8, fontsize=8)
for r, (e, c) in enumerate(zip(spectra, colors), 1):
    ax[2].semilogy(e[:200], color=c, label=f'round {r}')
ax[2].axhline(1, color='r', ls='--', lw=1)
ax[2].set(xlabel='PCA component', ylabel='component SNR',
          title='compression gets easier as the prior shrinks')
ax[2].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]

# Two things happened at once:
# 1. **The posterior tightened** toward the true (black) contours, round by
#    round — same network size, same per-round simulation budget; only the
#    *training distribution* moved.
# 2. **Compression became easy**: at the zoomed prior a handful of PCA
#    components carry all the signal (right panel) — the flat wide-prior
#    spectrum steepened dramatically. This is why adaptive summaries
#    (refitting the basis as you zoom) matter for the real problem.
#
# **One honest caveat.** Compare the late rounds against the black contours
# carefully: the final posterior is a little *narrower* than the exact one
# (roughly half the width, in our runs). Some of that is the flow's own mild
# over-confidence, but most of
# it is structural. The buffer converges to $L^\gamma\pi$, and we then train
# $q_c(\theta|s)$ **on that buffer** — so the likelihood enters twice, once
# through the buffer and once through the conditioning, and the readout behaves
# like $L^{1+\gamma}\pi$ rather than $L\pi$. Correcting it means importance
# reweighting the readout by $L^{-\gamma}$, which the ratio
# $\log q_c - \log q_m$ already gives us. This here is a simple example, but a
# complete implementation would do exactly that;
# we leave it out to keep the loop readable.
#
# **Exercise 4.**
# 1. Run with `gamma=0.1` and `gamma=1.0`. Which converges faster? Which is
#    riskier? (Think: what happens if an early, imperfect posterior estimate
#    excludes the truth — can a later round recover?)
# 2. Run with `refit_pca=False` (freeze the round-1 basis). How does that change
#    the zoom, and does it change it in the direction you expected?
# 3. **Lower the SNR.** Set `AMP = 0.5` and re-run everything from the top of
#    Part 4. With a weaker signal the likelihood grows competitive secondary
#    maxima, and the zoom sometimes locks onto one and shrinks around it
#    confidently — in our tests, roughly one run in three. This is *the*
#    failure mode of sequential inference: it is not that the posterior is
#    wide, it is that it is narrow and wrong. What would you monitor to catch
#    it without knowing the answer?

# %%

# your code goes here


# %%

# @title Reference solution to 4.1 and 4.2 { display-mode: "form" }
# Three more runs at the same budget as the one above (which we reuse as the
# baseline). Everything is compared on the width of the f0 posterior, since that
# is what the zoom is supposed to shrink.
GAMMA, N_R = 0.3, len(posts)

zoom_runs = {f'baseline (gamma = {GAMMA})': posts}
for name, kw in [('gamma = 0.1', dict(gamma=0.1)),
                 ('gamma = 1.0', dict(gamma=1.0)),
                 ('frozen PCA basis', dict(refit_pca=False))]:
    zoom_runs[name], _ = sequential_chirp(n_rounds=N_R, verbose=False, **kw)

# the exact posterior width in f0, from the grid we already computed (p, f0g)
pw = np.asarray(p) / np.asarray(p).sum()
f0n = f0g.cpu().numpy()[:, None]
f0_exact = np.sqrt((pw * (f0n - (pw * f0n).sum()) ** 2).sum())

print(f'\n{"run":26s} {"f0 width":>9s} {"vs exact":>9s} {"truth at":>12s}')
for name, ps in zoom_runs.items():
    w = ps[-1][:, 0].std().item()
    off = (THETA_CHIRP[0].item() - ps[-1][:, 0].mean().item()) / w
    print(f'{name:26s} {w:9.4f} {w / f0_exact:8.2f}x {off:+8.1f} sigma')

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for name, ps in zoom_runs.items():
    ax[0].semilogy(range(1, N_R + 1), [q[:, 0].std().item() for q in ps], 'o-',
                   ms=4, label=name)
    ax[1].plot(ps[-1][:, 0], ps[-1][:, 1], '.', ms=1.5, alpha=.2, label=name)
ax[0].axhline(f0_exact, color='k', ls='--', lw=1, label='exact posterior')
ax[0].set(xlabel='round', ylabel=r'posterior width in $f_0$',
          title='how fast each variant zooms')
ax[0].legend(fontsize=8)
ax[1].contour(f0g.cpu(), fdg.cpu(), p.T, levels=[0.011, 0.61], colors='k',
              linewidths=1)
ax[1].plot(*THETA_CHIRP.cpu(), 'r*', ms=14)
ax[1].set(xlabel=r'$f_0$', ylabel=r'$\dot f$', xlim=(F0_LO, F0_HI),
          ylim=(FD_LO, FD_HI), title=f'round {N_R} against the exact posterior')
ax[1].legend(markerscale=8, fontsize=8)
fig.tight_layout()

# %%

# @title Reference solution to 4.3 { display-mode: "form" }
# The failure to catch is "narrow and wrong", so the monitor has to ask whether
# the parameters we settled on can still reproduce the data. Maximize the
# likelihood over the phase (a matched filter) and look at the residual: it
# should be consistent with pure noise, chi^2 per sample = 1.

def max_logl(theta, x_obs=x_obs_chirp, n_phi=64):
    """Score parameters by log L with the nuisance phase maximized out.

    This is a matched filter, and it is the monitor Exercise 4.3 asks for: it
    needs no knowledge of the true answer.

    Arguments:
      theta: the parameters to score, (n, 2).
      x_obs: the observed data (1, 1024).
      n_phi: how many phases to try when maximizing.

    Returns:
      The maximized log-likelihood of each theta, (n,).
    """
    ph = 2 * np.pi * (theta[:, :1] * tgrid + 0.5 * theta[:, 1:2] * tgrid ** 2)
    phis = torch.linspace(0, 2 * np.pi, n_phi, device=dev)[:, None, None]
    h = AMP * torch.sin(ph + phis)                    # (n_phi, n, N_T) templates
    return (h @ x_obs[0] - 0.5 * (h ** 2).sum(-1)).max(0).values


d2 = (x_obs_chirp[0] ** 2).sum()                      # chi^2 of doing nothing
for name, th in [('final posterior', posts[-1][:200].to(dev)),
                 ('random prior draws', draw_prior(200))]:
    chi2 = (d2 - 2 * max_logl(th).max()) / N_T
    print(f'best residual chi2 per sample, {name:19s} {chi2:.3f}')

# %% [markdown]

# **Answers.** (1) $\gamma$ buys speed at the price of safety, and the run above
# shows both ends of that. $\gamma = 1$ zooms fastest — the buffer *is* the
# posterior estimate, so every simulation goes where the answer seems to be —
# and after ten rounds it is already *narrower* than the exact posterior. There
# is no safety margin left: if an early round's estimate had excluded the truth,
# the proposal would have no samples out there and no later round could pull it
# back. $\gamma = 0.1$ shows the opposite failure. It barely zooms at all
# (the width stalls near 1 rather than reaching 0.3), and because the flow never
# gets training data at the scale of the likelihood, the readout ends up both
# broad *and* mis-centred — in our run the truth sits about $2.5\sigma$ out. Slow
# is not the same as safe.
#
# (2) Freezing the basis does **not** slow this problem down — it is three times
# *further* along after ten rounds than the adaptive default (width 0.09 against
# 0.31), and lands within a few percent of the exact width. Refitting redefines
# the conditioning vector
# every round, so the warm-started network has to relearn the map from summaries
# to parameters while it is also chasing a moving buffer, and 500 steps per round
# is not enough to do both. The adaptive basis is still the better summary in
# information terms (that is what the steepening spectrum shows), but its
# advantage only pays off when the per-round training budget is large enough to
# absorb the change — or, as production codes do, when the basis is aligned
# across rounds instead of refitted from scratch. Worth knowing that a good idea
# can cost more than it returns at a small budget.
#
# (3) Ask whether the parameters you settled on can still reproduce the data.
# Maximizing the likelihood over the nuisance phase and looking at the residual
# gives $\chi^2$ per sample $\approx 1.04$ at the final posterior against
# $1.2$–$1.4$ at random prior draws: the fit is consistent with pure noise, so
# the zoom landed on a real solution. A posterior that had locked onto a
# secondary maximum would keep shrinking while this number stayed high — narrow
# *and* wrong, caught without knowing the answer. The other standard monitors are
# the effective sample size of the importance weights (a collapse means the
# proposal has stopped covering the target) and simply re-running with a
# different seed to see whether the answers agree.

# %% [markdown]

# ---
# # Where next: the real thing, live
#
# You now have every piece of a production simulation-based-inference
# pipeline: a density model that can represent awkward shapes (Part 2), a way
# to turn it into a posterior by feeding it simulator output (Part 3), and the
# compression plus sequential zoom that make it work when the posterior is a
# needle in the prior's haystack (Part 4).
#
# The companion notebook **`lisa_mbhb_first_steps.ipynb`** runs exactly this on
# real LISA data: the LDC1-1 (Radler) massive black-hole binary, nine
# parameters, with the waveforms simulated *live* by `lisabeta` inside the zoom
# loop. It uses the same `fm_loss`, the same `VelocityNet`, the same
# `fm_sample` and `fm_logprob` you have here, and it runs the loop for four
# rounds — a minute or so per round on a Colab T4.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_mbhb_first_steps.ipynb)
#
# ---
# ## Where to go from here
#
# - **Dynamic (sequential) SBI:** Alvey, Lyu, Weniger et al., arXiv:2510.13997
#   — the tempered-buffer mechanism you used in Part 4, at production scale.
# - **Flow matching:** Lipman et al. 2022 (arXiv:2210.02747); for GW posteriors
#   Dax et al. (arXiv:2305.17161).
# - **LISA Data Challenges:** https://lisa-ldc.lal.in2p3.fr — Radler is LDC1;
#   the reference posterior above is the Baker & Marsat submission
#   (arXiv:2003.00357 describes their method).
# - All code from this notebook + the production experiments behind the quoted
#   numbers: this repository and the `standalone_tests/` folder of github.com/lvhf123/dsbi-ldc-mbhb.
#
# ---
# ## PyTorch FAQ
#
# The things that most often interrupt a first read of this notebook.
#
# **Why is every tensor shaped `(n, 1)` and not just `(n,)`?** PyTorch layers
# treat the first axis as the batch (one row per example) and the rest as
# features. `nn.Linear(1, H)` wants one feature per row, so a batch of $n$
# scalars is `(n, 1)`. `x[:, None]` adds that axis.
#
# **What does `.detach()` do?** It returns the same numbers with the link to the
# computational graph cut, so autograd will not track them. You need it before
# handing a tensor to matplotlib or numpy; without it you either get an error or
# keep the whole graph alive.
#
# **`.detach()` vs `torch.no_grad()` vs `.item()`?** `no_grad()` is a block that
# stops the graph being *built* at all — use it for evaluation and
# book-keeping. `.detach()` cuts one existing tensor loose. `.item()` pulls a
# single number out of a one-element tensor as a plain Python float.
#
# **Why `opt.zero_grad()` every step?** `backward()` *adds* into `p.grad` rather
# than overwriting it. Forget to zero and you are stepping on the sum of all
# gradients so far — the most common PyTorch bug.
#
# **What is `state_dict()`?** A dictionary of the network's tensors, so
# `copy.deepcopy(net.state_dict())` is a snapshot of the weights and
# `net.load_state_dict(snap)` puts them back. That is all early stopping needs.
#
# **What does `nn.Module` give me?** Assigning `self.fc1 = nn.Linear(...)`
# registers that layer's tensors, so they appear in `net.parameters()` and move
# with `.to(dev)`. Calling `net(x)` runs `forward(x)` with the machinery around
# it.
#
# **Why `net(x)` and not `net.forward(x)`?** Both run the same code, but the
# call form also runs the hooks and mode handling PyTorch wraps around it.
# Always use `net(x)`.
#
# **What is Adam actually doing beyond `phi ← phi − eta·g`?** It keeps a running
# average of each parameter's gradient and its square, and scales that
# parameter's step by them — so parameters with consistently small gradients
# still move. Same loop, per-parameter step size.
#
# **CPU or GPU?** Tensors and models live on a device and must match. `dev` is
# set in the first cell; `.to(dev)` moves things. Part 1 is small enough to stay
# on the CPU; later parts put everything on the GPU.
#
# **What does `torch.manual_seed` fix, exactly?** The global random stream used
# by `torch.rand`, `torch.randn` *and* weight initialization. Calling it before
# building a network makes that network's random starting point reproducible.

# %%

# (housekeeping cell — saves all figures when this notebook is executed as a
# test script; does nothing in an interactive colab session)
if os.environ.get('TUTORIAL_SAVE_FIGS'):
    for i in plt.get_fignums():
        plt.figure(i).savefig(f'tutorial_solutions_fig_{i:02d}.png', dpi=110,
                              bbox_inches='tight')
    print(f'saved {len(plt.get_fignums())} figures')

# %% [markdown]

# *Generated for the LISA SBI tutorial, 2026-07-27. Built and battle-tested on
# the LDC1-1 MBHB analysis campaign of July 2026.*
