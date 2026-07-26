# %% [markdown]
# # From MLPs to LISA: simulation-based inference, hands-on
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_lisa_sbi.ipynb)
#
# **LISA tutorial — approximately 95 minutes.**
#
# | part | idea | new ingredient |
# |---|---|---|
# | 1 | fit a *function* with a neural network | MLPs, overfitting, early stopping |
# | 2 | fit a *distribution* | flow matching (FM), conditioning |
# | 3 | fit a *posterior*: feed FM pairs from a simulator | SBI, amortization |
# | 4 | a toy gravitational wave | data compression + **sequential** inference |
#
# Each part fixes the visible failure of the one before. All code is plain
# PyTorch, and the companion notebook `lisa_sequential_new.ipynb` runs the 
# same algorithms on real LISA data.
#
# > **Colab setup:** Runtime → Change runtime type → **T4 GPU**, then run all
# > cells top to bottom. Everything also works on CPU, just slower.
#
# > **Collapsing sections:** headings sit in their own cells, so clicking the
# > triangle next to one folds it away. Anything titled ***Aside*** is safe to
# > skip on a first read — collapse them all and you have the core path.
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
# # Part 1 — Neural networks  *(~30 min)*

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
    """A dense feed-forward network from R^d_in to R^d_out."""

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
#   curves. Neither is more powerful; they fail differently.

# %% [markdown]
# ### Aside — how the initial values are chosen

# %% [markdown]
# By default `nn.Linear` draws each weight and bias uniformly from
# $\pm 1/\sqrt{n_{\rm in}}$, where the **fan-in** $n_{\rm in}$ is the number of
# inputs to that layer. Wider layers therefore start with smaller weights, which
# keeps the scale of a layer's output roughly independent of its width — without
# it, stacking wide layers would make activations grow or vanish geometrically
# with depth.

# %% [markdown]
# ## Fitting with gradient descent

# %% [markdown]
# Given pairs $(x_i, y_i)$ we want the network to reproduce them. The obvious
# measure of mismatch is the **mean squared error**,
#
# $$\mathrm{MSE}(\phi) = \frac{1}{N} \sum_{i=1}^{N}
#   \big(\mathrm{MLP}_\phi(x_i) - y_i\big)^2 ,$$
#
# minimized over all of $\phi$ at once by **gradient descent** — repeatedly
# stepping in the direction of steepest descent,
#
# $$\phi_{k+1} = \phi_k - \eta\, \nabla_\phi \mathcal L(\phi_k),
#   \qquad k = 0, 1, 2, \dots$$
#
# from the random $\phi_0$ above, with the **learning rate** $\eta$ setting the
# step size. Each iteration $k$ over all $N$ examples is one *epoch*.
#
# **We monitor something better than the MSE.** An MSE says how large the errors
# are, not whether the network's *confidence* is justified. So give the model a
# noise level $\sigma$ — $y_i \sim \mathcal N(\mathrm{MLP}_\phi(x_i), \sigma^2)$ —
# and track its negative log-likelihood,
#
# $$\mathcal L = \frac{\mathrm{MSE}}{2\sigma^2} \;+\; \log\sigma ,
#   \qquad \hat\sigma^2 = \mathrm{MSE}_{\rm train},$$
#
# a probability density rather than a score. The best $\sigma$ for a given
# network is just its root-mean-square training residual, so we plug that in
# each epoch instead of fitting it.
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
def fit(net, x, y, x_val, y_val, lr=1e-4, patience=300, epochs=100_000):
    """Train on MSE; monitor the Gaussian NLL with a plug-in sigma; stop early."""
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

    net.load_state_dict(snap)                          # rewind to the best epoch
    print(f'stopped at epoch {ep + 1}; best validation NLL {best:.2f} '
          f'at epoch {best_ep + 1}')
    return np.array(hist), best_ep

# %% [markdown]
# ### Aside — why the plug-in $\sigma$ costs nothing

# %% [markdown]
# Two conveniences hide in $\hat\sigma^2 = \mathrm{MSE}_{\rm train}$. First, it
# is the exact maximum-likelihood $\sigma$ for the current network, so no
# optimization is involved. Second, $\sigma$ drops out of
# $\nabla_\phi \mathcal L$ entirely — minimizing the NLL over the network *is*
# minimizing the MSE — so the gradient step is untouched and $\hat\sigma$ is
# pure diagnostics. You could instead make $\sigma$ a learned `nn.Parameter`, but
# then it lags the network by however long gradient descent takes to drag it
# down, and the overfitting signal arrives late.

# %% [markdown]
# ### Aside — what Adam adds

# %% [markdown]
# We use **Adam** rather than the bare update above. It keeps a running average
# of each parameter's gradient and of its square, and scales that parameter's
# step by them, so parameters with persistently small gradients still move. The
# loop structure is unchanged; only the step size becomes per-parameter and
# adaptive.

# %% [markdown]
# ## Example: measuring a frequency

# %% [markdown]
# Nothing in the network cared that the input was one number: `MLP(30, 1)` maps a
# 30-dimensional vector to a scalar just as happily. That is the shape of every
# problem in the rest of this notebook — **a lot of data in, few parameters
# out** — so we start there.
#
# A sine of unknown frequency $\nu$ is sampled at 30 fixed times and buried in
# noise:
#
# $$d_j = \sin(2\pi\,\nu\,t_j) + 0.3\,\varepsilon_j, \qquad
#   t_j = \tfrac{j}{29},\quad j = 0 \dots 29, \qquad
#   \nu \sim U(1, 5)\ \text{cycles}.$$
#
# The network sees the 30 noisy samples and must return $\nu$. We hand it the
# rescaled target $(\nu-1)/4 \in [0,1]$, for the $O(1)$ reason above. The
# frequencies stay well below this grid's Nyquist limit of 15 cycles, so nothing
# here is ambiguous in principle.

# %%
N_GRID, NU_LO, NU_HI = 30, 1.0, 5.0
tj = torch.linspace(0, 1, N_GRID)
SPAN = NU_HI - NU_LO


def sine_data(n, sigma=0.3, seed=0, random_phase=False):
    """n examples: d (n, 30) noisy samples, and the rescaled frequency (n, 1)."""
    torch.manual_seed(seed)
    nu = NU_LO + SPAN * torch.rand(n, 1)                     # nu ~ U(1, 5)
    phi = torch.rand(n, 1) * 2 * np.pi if random_phase else torch.zeros(n, 1)
    d = torch.sin(2 * np.pi * nu * tj + phi) + sigma * torch.randn(n, N_GRID)
    return d, (nu - NU_LO) / SPAN


d_tr, nu_tr = sine_data(300, seed=0)                # training set
d_va, nu_va = sine_data(2000, seed=1)               # held-out set

fig, ax = plt.subplots(1, 3, figsize=(13, 2.7), sharey=True)
for a, i in zip(ax, range(3)):
    nu_i = NU_LO + SPAN * nu_tr[i, 0]
    a.plot(tj, torch.sin(2 * np.pi * nu_i * tj), 'k--', lw=1, label='hidden signal')
    a.plot(tj, d_tr[i], 'C0o-', ms=4, lw=.8, label='what the network sees')
    a.set(xlabel='t', title=fr'$\nu$ = {nu_i:.2f} cycles')
ax[0].set_ylabel('d'); ax[0].legend(fontsize=8)
fig.tight_layout()

# %%
freq_net = MLP(N_GRID, 1, 256)                      # 30 numbers in, 1 out
hist, best_ep = fit(freq_net, d_tr, nu_tr, d_va, nu_va)

sigma_hat = hist[best_ep, 2] * SPAN                 # plug-in sigma, in cycles
with torch.no_grad():
    nu_est = (NU_LO + SPAN * freq_net(d_va)).squeeze()
nu_true = (NU_LO + SPAN * nu_va).squeeze()
resid = nu_est - nu_true

# %%
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].fill_between([NU_LO, NU_HI], [NU_LO - sigma_hat, NU_HI - sigma_hat],
                   [NU_LO + sigma_hat, NU_HI + sigma_hat], color='C0', alpha=.18,
                   label=fr'model: $\pm\hat\sigma$ = {sigma_hat:.2f} cycles')
ax[0].plot([NU_LO, NU_HI], [NU_LO, NU_HI], 'k--', lw=1)
ax[0].plot(nu_true, nu_est, 'C0.', ms=2, alpha=.35, label='held-out data')
ax[0].set(xlabel=r'true $\nu$ [cycles]', ylabel=r'estimated $\nu$ [cycles]',
          aspect='equal', title=f'RMSE {resid.pow(2).mean().sqrt():.3f} cycles')
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(hist[:, 0], lw=1, label='training NLL')
ax[1].plot(hist[:, 1], lw=1, label='validation NLL')
ax[1].axvline(best_ep, color='k', ls=':', lw=1, label='best epoch (kept)')
ax[1].set(xlabel='epoch', ylabel='negative log-likelihood', ylim=(-3.2, 1.0))
ax[1].legend(fontsize=8)
fig.tight_layout()

# %%
print('actual scatter, in bands of frequency:')
for lo in range(1, 5):
    m = (nu_true >= lo) & (nu_true < lo + 1)
    print(f'  nu = {lo}-{lo + 1} cycles:  RMSE {resid[m].pow(2).mean().sqrt():.3f}'
          f'   (model claims {sigma_hat:.3f})')

# %% [markdown]
# **Overfitting, unmistakably.** The training NLL falls without limit, because
# $\hat\sigma$ tracks training residuals and those keep shrinking as the network
# memorizes its 300 examples. The validation NLL bottoms out and then leaves the
# top of the panel: a network claiming precision it achieves only on data it has
# already seen is a terrible model for data it has not. Early stopping keeps the
# one network on that curve worth having.
#
# **The error bar is one number; the error is not.** The band is the model's
# $\hat\sigma$, identical at every frequency, while the printout shows the real
# scatter roughly doubling across the band — 30 samples cover fewer points per
# cycle as $\nu$ grows. One $\sigma$ for all inputs is the wrong *shape* of
# answer.
#
# What we want is a distribution over $\nu$ that **depends on the data**: wide
# here, narrow there, and in general curved and multi-modal. That is Part 2.

# %% [markdown]
# ### Exercise 1

# %% [markdown]
# `experiment` below re-runs everything with whatever you change. Try:
#
# 1. **`patience`** — 20, then 3000. How much accuracy does stopping too early
#    cost, and how bad does the over-confidence get if you stop too late?
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
               nu_hi=NU_HI, random_phase=False):
    """Re-run the frequency fit with different settings; returns (RMSE, sigma_hat)."""
    global NU_HI, SPAN
    NU_HI, SPAN = nu_hi, nu_hi - NU_LO             # let the band be changed
    dt, nt = sine_data(n_train, sigma, seed=0, random_phase=random_phase)
    dv, nv = sine_data(2000, sigma, seed=1, random_phase=random_phase)
    net = MLP(N_GRID, 1, width)
    h, bep = fit(net, dt, nt, dv, nv, patience=patience)
    with torch.no_grad():
        r = (SPAN * (net(dv) - nv)).squeeze()
    rmse, sig = r.pow(2).mean().sqrt().item(), h[bep, 2] * SPAN
    print(f'  RMSE {rmse:.3f} cycles,  model claims {sig:.3f}  '
          f'-> off by x{rmse / sig:.2f}')
    NU_HI, SPAN = 5.0, 4.0                         # restore the defaults
    return rmse, sig


experiment()                                        # <-- change one thing at a time

# %% [markdown]
# ---
# # Part 2 — Modeling distributions with flow matching  *(~25 min)*
#
# Part 1 fitted a *function*: one number $y$ for each input $x$. Now we fit a
# **distribution**.
#
# From here on $\theta$ denotes the quantity whose distribution we care about
# — in Parts 3 to 5 it will be the physical parameters we want to infer. We are
# handed samples from an unknown target density and want a model that
# reproduces it:
#
# $$q_\phi(\theta) \;\approx\; p(\theta),$$
#
# where $p(\theta)$ is the **target** we only ever see samples from, and
# $q_\phi(\theta)$ is our **model**, with $\phi$ the learnable network
# parameters — the same $\phi$ that gradient descent moved in Part 1.
#
# ## Why this is the central problem — and why it is hard
#
# Essentially all of statistics is a statement about probability
# distributions: a prior, a likelihood, a posterior, a predictive
# distribution, an evidence integral. So being able to *represent an
# arbitrary probability density* — and compute with it — is not one technique
# among many, it is the core capability. Everything after this cell is an
# application of it.
#
# The obvious move would be to do what we did in Part 1: let an MLP output
# $q_\phi(\theta)$ directly. It does not work, because a density is a far more
# constrained object than a function. We need a $q_\phi(\theta)$ that is
#
# - **non-negative** everywhere, $q_\phi(\theta) \ge 0$;
# - **normalized**, $\int q_\phi(\theta)\,\mathrm d\theta = 1$;
# - typically **smooth**, since the physics behind it usually is.
#
# and, crucially, we need *two operations* on it:
#
# - **evaluate**: given $\theta$, return $q_\phi(\theta)$ (for likelihoods,
#   weights, model comparison);
# - **sample**: produce fresh draws $\theta \sim q_\phi$ (for error bars,
#   corner plots, propagating uncertainty).
#
# Ideally both are **fast**, because we will call them millions of times.
#
# Non-negativity is easy — exponentiate. Normalization is the hard one: it is a
# *global* constraint, so you cannot simply declare an MLP's output to be a
# density and train it, since changing the network anywhere changes the
# integral everywhere. The classical escapes each give something up. A Gaussian
# satisfies everything and is trivially fast, but can represent almost nothing.
# An unnormalized energy-based model $q_\phi \propto e^{-E_\phi(\theta)}$ is
# arbitrarily flexible, but you no longer know the normalization and sampling
# needs MCMC — slow, and re-run from scratch for every new problem. Normalizing
# flows give you both operations exactly, but only by restricting the
# architecture to invertible layers with tractable Jacobians.
#
# ## Flow matching
#
# Flow matching is the current answer: an unrestricted network, exact sampling,
# a computable density, and a training objective that is plain regression.
#
# The idea is to build the sampler out of a **flow**. Start from an easy
# distribution (a unit Gaussian) and move the points continuously until they
# are distributed like $p$. The motion is described by a **velocity field**
# $v_\phi(\theta, t)$ — an MLP exactly like Part 1's, taking a position
# $\theta$ and a time $t \in [0,1]$ — and "training the generative model" means
# fitting that velocity field. The price is that sampling costs an ODE solve
# rather than a single forward pass.
#
# ### The mechanics, in three equations
#
# Write $\theta_0$ for a point drawn from the Gaussian at $t = 0$ and
# $\theta_1$ for a data point at $t = 1$.
#
# **1. Training.** Pick a random time $t$, a random noise point
# $\theta_0 \sim \mathcal N(0, I)$ and a random data sample
# $\theta_1 \sim p$. Place yourself on the straight line between the two at
# time $t$, and regress the velocity onto the direction from $\theta_0$ to
# $\theta_1$:
#
# $$\mathcal L(\phi) = \mathbb E_{t,\, \theta_0,\, \theta_1}
#   \Big[\;\big\| \, v_\phi\big(
#   \underbrace{(1-t)\,\theta_0 + t\,\theta_1}_{\textstyle \theta_t},
#   \; t\big) \; - \; (\theta_1 - \theta_0) \, \big\|^2 \;\Big],
#   \qquad t \sim U(0,1).$$
#
# Note what is *absent*: no integration, no sampling from the model, no
# density, no Jacobian. It is a plain regression loss — Part 1's `fit` with a
# fancier target.
#
# **2. Sampling.** Draw a noise point and integrate the learned velocity field
# from $t = 0$ to $t = 1$:
#
# $$\theta(0) = \theta_0 \sim \mathcal N(0, I), \qquad
#   \frac{\mathrm d \theta}{\mathrm d t} = v_\phi\big(\theta(t),\, t\big),
#   \qquad \theta(1) \sim q_\phi \;\approx\; p .$$
#
# We integrate with plain Euler steps,
# $\theta \mathrel{+}= v_\phi(\theta,t)\,\Delta t$.
#
# **3. Evaluation.** If you also need the *density* of a point (we will, in
# Part 4), integrate the same ODE **backwards** from $\theta_1$ while
# accumulating the divergence of the velocity field:
#
# $$\log q_\phi(\theta_1) = \log \mathcal N\big(\theta(0);\, 0, I\big)
#   \; - \; \int_0^1 \nabla \!\cdot\! v_\phi\big(\theta(t),\, t\big)\,
#   \mathrm d t .$$
#
# *Why* regressing onto straight lines between unrelated random pairs produces
# a velocity field whose flow transports $\mathcal N(0,I)$ to $p$ is genuinely
# non-obvious, and we will not derive it here — see Lipman et al.
# (arXiv:2210.02747), Liu et al. (arXiv:2209.03003) and Albergo &
# Vanden-Eijnden (arXiv:2209.15571). For our purposes it is a black box with
# three knobs, and the three equations above are all of them. Note that the
# three requirements of the box above are met *by construction*: samples come
# out of an ODE so they are genuine draws, and the density is normalized
# because the flow only ever transports an already-normalized Gaussian.

# %% [markdown]
# ## Two target distributions
#
# Two densities to aim at, both shaped like things you genuinely meet in GW
# parameter estimation: a **banana** (a curved degeneracy between two
# parameters) and a **spiral** (multi-modal along a curve). Neither is
# remotely Gaussian.

# %%
def target_banana(n):
    """n samples of a curved 'banana' density -> (n, 2)."""
    t1 = 6 * torch.rand(n, device=dev) - 3                  # spread along the arc
    t2 = 0.3 * torch.rand(n, device=dev).mul(2).sub(1) \
        + 0.3 * t1 ** 2 - 1.2                               # bend it
    return torch.stack([t1, t2 + 0.15 * torch.randn(n, device=dev)], 1)


def target_spiral(n):
    """n samples along an Archimedean spiral -> (n, 2)."""
    a = 3 * np.pi * torch.rand(n, 1, device=dev).sqrt()   # angle
    r = 0.45 * a                                          # radius grows with angle
    c = torch.cat([r * a.cos(), r * a.sin()], 1)
    return c + 0.12 * torch.randn(n, 2, device=dev)       # thicken the arm


th_banana = target_banana(20000)
th_spiral = target_spiral(20000)

fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
for a, th, ttl in [(ax[0], th_banana, 'target: banana'),
                   (ax[1], th_spiral, 'target: spiral')]:
    a.plot(th[:4000, 0].cpu(), th[:4000, 1].cpu(), 'k.', ms=1, alpha=.3)
    a.set(title=ttl, xlabel=r'$\theta_1$', ylabel=r'$\theta_2$', aspect='equal')
fig.tight_layout()

# %% [markdown]
# ## The implementation
#
# Four short functions. They are used unchanged for the rest of the notebook,
# including the companion LISA notebook. `cond` is the conditioning input; leave
# it `None` for now (we use it in the second half of this part).

# %%
# Generic MLP helper for the rest of the notebook: same idea as Part 1's MLP
# class, but with configurable input/output dimensions and depth.
def mlp(d_in, d_out, hidden, layers):
    mods, d = [], d_in                              # mods: the layer list so far
    for _ in range(layers):                         # one hidden block per layer
        mods += [nn.Linear(d, hidden), nn.ReLU()]   # affine map, then ReLU
        d = hidden                                  # next block takes `hidden` in
    return nn.Sequential(*mods, nn.Linear(d, d_out))  # read-out, no nonlinearity


def fm_loss(net, th1, cond=None):
    """Equation 1: the flow-matching objective. Reused in Parts 3, 4 and 5."""
    th0 = torch.randn_like(th1)                     # theta_0 ~ N(0, I): noise point
    t = torch.rand(len(th1), 1, device=th1.device)  # t ~ U(0, 1), one per example
    tht = (1 - t) * th0 + t * th1                   # theta_t on the straight line
    v = net(tht, t, cond)                           # velocity the network predicts
    return ((v - (th1 - th0)) ** 2).mean()          # regress it onto theta_1 - theta_0


class VelocityNet(nn.Module):
    """Velocity field v(theta, t | cond): an MLP with a Fourier embedding of t."""

    def __init__(self, d_theta, d_cond=0, hidden=128, layers=3):
        super().__init__()
        self.freqs = torch.tensor([1., 2., 4., 8.])          # time-embedding freqs
        # inputs: theta (d_theta) + time embedding (9) + conditioning (d_cond)
        self.net = mlp(d_theta + 9 + d_cond, d_theta, hidden, layers)

    def forward(self, th, t, cond=None):
        ft = 2 * np.pi * t * self.freqs.to(t.device)         # (n, 4) scaled times
        temb = torch.cat([t, ft.sin(), ft.cos()], 1)         # (n, 9): t and 4 sin/cos
        # a raw scalar t is hard for an MLP to resolve finely; sin/cos of several
        # frequencies gives it a basis in which sharp t-dependence is easy
        parts = [th, temb] if cond is None else [th, temb, cond]
        return self.net(torch.cat(parts, 1))                 # -> (n, d_theta)


@torch.no_grad()                                    # sampling never needs gradients
def fm_sample(net, cond, d_theta, steps=64, n=None, return_path=False):
    """Equation 2: Euler-integrate dtheta/dt = v from t=0 (noise) to t=1."""
    n = len(cond) if cond is not None else n        # one sample per conditioning row
    device = cond.device if cond is not None else next(net.parameters()).device
    th = torch.randn(n, d_theta, device=device)     # theta(0) ~ N(0, I)
    path = [th.clone()]                             # keep the route, for plotting
    for i in range(steps):
        t = torch.full((n, 1), (i + 0.5) / steps, device=device)  # midpoint time
        th = th + net(th, t, cond) / steps          # Euler step: theta += v * dt
        path.append(th.clone())
    return (th, torch.stack(path)) if return_path else th        # theta(1) ~ q_phi

# %%
def train_fm(net, th1, cond=None, steps=3000, batch=512, lr=1e-3, log=True):
    """Minimize fm_loss by Adam -- the same loop as Part 1's fit()."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)      # holds pointers to phi
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(th1), (batch,), device=th1.device)  # minibatch
        loss = fm_loss(net, th1[i], None if cond is None else cond[i])  # forward
        opt.zero_grad(); loss.backward(); opt.step()     # zero, backward, step
        if log and (step + 1) % 1000 == 0:
            print(f'  step {step + 1}/{steps}  loss {loss.item():.3f}  '
                  f'[{time.time() - t0:.0f}s]')
    return net

# %% [markdown]
# ## Train it on the spiral

# %%
snet = VelocityNet(2).to(dev)                       # d_cond = 0: unconditional
train_fm(snet, th_spiral)

samp, path = fm_sample(snet, None, 2, n=4000, return_path=True)

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4))
# (a) target vs model samples
ax[0].plot(th_spiral[:4000, 0].cpu(), th_spiral[:4000, 1].cpu(), 'k.', ms=1,
           alpha=.15, label='target')
ax[0].plot(samp[:, 0].cpu(), samp[:, 1].cpu(), 'C0.', ms=1, alpha=.3,
           label='flow-matching samples')
ax[0].set(title='the model learned the spiral', xlabel=r'$\theta_1$', ylabel=r'$\theta_2$')
ax[0].legend(markerscale=8, fontsize=8); ax[0].set_aspect('equal')
# (b) where each sample came from, and the route it took
p = path[:, :60].cpu()
ax[1].plot(p[:, :, 0], p[:, :, 1], ':', color='gray', lw=.7)
ax[1].plot(p[0, :, 0], p[0, :, 1], 'C2o', ms=4, label=r'base sample $\theta(0)$')
ax[1].plot(p[-1, :, 0], p[-1, :, 1], 'C0o', ms=4, label=r'final sample $\theta(1)$')
ax[1].set(title='60 trajectories of the learned flow', xlabel=r'$\theta_1$')
ax[1].legend(fontsize=8); ax[1].set_aspect('equal')
fig.tight_layout()

# %% [markdown]
# **Reading the panels.**
# - *Left:* the model reproduces the spiral, arms and gaps included — from a
#   plain regression loss and 3000 Adam steps.
# - *Right:* every final sample traces back to one Gaussian base point. Note
#   the routes are **curved**, even though training only ever used *straight*
#   lines between random pairs $(\theta_0, \theta_1)$ — the network learns the *average*
#   velocity over all pairs passing through a point, and the resulting flow
#   bends. Do not expect a trajectory to connect the pair it was trained on.
#
# **Exercise 2a — your own distribution.** Fill in `my_target` below. That is
# the *only* thing you write: the cell after it already trains a flow on
# whatever your sampler returns and plots the result against it. Ideas: two
# moons, a checkerboard, a ring, a mixture of a few Gaussians, your initials.
# (Return the samples in a random order — the plots show the first 4000.)

# %%
# TODO — your code here: return (n, 2) samples from a distribution you invent.
def my_target(n):
    raise NotImplementedError('write your own target sampler')


# %%
# @title Reference solution { display-mode: "form" }
def my_target(n):                                   # noqa: F811
    """Two moons."""
    a = np.pi * torch.rand(n // 2, 1, device=dev)   # half circle
    top = torch.cat([a.cos(), a.sin()], 1)
    bot = torch.cat([1 - a.cos(), -a.sin() + 0.4], 1)
    th = 1.6 * torch.cat([top, bot]) + 0.09 * torch.randn(2 * (n // 2), 2, device=dev)
    return th[torch.randperm(len(th), device=dev)]  # shuffle: plots take th[:4000]


# %%
# Provided: fit a flow to whatever my_target returns, and plot both.
th_mine = my_target(20000)
mynet = VelocityNet(2).to(dev)
train_fm(mynet, th_mine, log=False)
samp_mine = fm_sample(mynet, None, 2, n=4000)

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.plot(th_mine[:4000, 0].cpu(), th_mine[:4000, 1].cpu(), 'k.', ms=1, alpha=.15,
        label='target')
ax.plot(samp_mine[:, 0].cpu(), samp_mine[:, 1].cpu(), 'C0.', ms=1, alpha=.3,
        label='flow-matching samples')
ax.set(xlabel=r'$\theta_1$', ylabel=r'$\theta_2$', title='your own distribution')
ax.legend(markerscale=8, fontsize=8); ax.set_aspect('equal')
fig.tight_layout()

# %% [markdown]
# ## Conditional flow matching
#
# One more ingredient and we are done. Very often we do not want *one*
# distribution but a *family* of them, indexed by some input $c$ — that is a
# **conditional** density $q(\theta \,|\, c)$. The change is minimal: feed $c$ to
# the velocity field alongside $\theta$ and $t$,
#
# $$\mathcal L(\phi) = \mathbb E_{t,\, \theta_0,\, (\theta_1, c)}
#   \Big[\;\big\| \, v_\phi\big((1-t)\,\theta_0 + t\,\theta_1,
#   \; t \,\big|\, c\big) \; - \; (\theta_1 - \theta_0) \, \big\|^2 \;\Big],$$
#
# where the pairs $(\theta_1, c)$ are drawn **jointly**: each training sample comes
# with the $c$ it belongs to. Sampling is unchanged except that you say which
# $c$ you want. In code, that is the `cond` argument we have been passing as
# `None` — nothing else moves.
#
# Demonstration: rings of varying radius. The condition $c$ is the radius,
# the target $q(\theta\,|\,c)$ is a ring of that radius.

# %%
def target_ring(radius):
    """radius: (n, 1) -> (n, 2) points on a ring of that radius."""
    a = 2 * np.pi * torch.rand_like(radius)
    return (torch.cat([radius * a.cos(), radius * a.sin()], 1)
            + 0.06 * torch.randn(len(radius), 2, device=radius.device))


c_train = 0.5 + 2.0 * torch.rand(40000, 1, device=dev)   # radii in [0.5, 2.5]
th_ring = target_ring(c_train)

rnet = VelocityNet(2, d_cond=1).to(dev)                  # <-- the only change
train_fm(rnet, th_ring, c_train, steps=4000)

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
ax[0].plot(th_ring[:6000, 0].cpu(), th_ring[:6000, 1].cpu(), 'k.', ms=1, alpha=.2)
ax[0].set(title=r'training data: all radii mixed together', xlabel=r'$\theta_1$',
          ylabel=r'$\theta_2$')
for r, col in zip([0.7, 1.3, 1.9, 2.4], ['C0', 'C1', 'C2', 'C3']):
    c = torch.full((1500, 1), r, device=dev)
    s = fm_sample(rnet, c, 2).cpu()
    ax[1].plot(s[:, 0], s[:, 1], '.', color=col, ms=1.5, alpha=.5, label=f'c = {r}')
ax[1].set(title='one network, four requested radii', xlabel=r'$\theta_1$')
ax[1].legend(markerscale=6, fontsize=8)
for a in ax:
    a.set_aspect('equal'); a.set(xlim=(-3, 3), ylim=(-3, 3))
fig.tight_layout()

# %% [markdown]
# The training set (left) is a filled disc — no individual ring is visible in
# it. Yet asking the trained network for $c = 0.7$ or $c = 2.4$ returns a
# clean ring of exactly that radius (right). The network did not memorize four
# rings; it learned the *whole family* $q(\theta\,|\,c)$ at once, which is why a
# radius it never saw during training works just as well. That property is
# called **amortization**, and it is the entire reason this machinery is
# interesting for inference.
#
# **Exercise 2b.**
# 1. **Interpolation.** Ask for a radius outside the training range, e.g.
#    $c = 3.5$. Does the network extrapolate sensibly? (It has no reason to.)
# 2. **The ODE knob.** Redo `fm_sample` with `steps=1, 4, 16`. How many Euler
#    steps do you need before the rings stop being distorted? What does
#    `steps=1` correspond to geometrically?
# 3. **Your own family.** Make the condition control something else — the
#    opening angle of an arc, the separation of two blobs, the number of
#    modes — and check that unseen conditions behave.
# 4. *(bonus)* In `fm_loss`, replace `t = torch.rand(...)` by
#    `t = torch.rand(...) ** 0.5`, which spends more training time near
#    $t = 1$ where the sharp structure forms. Do few-step samples get cleaner?
#    Remember this trick — it returns at the end of Part 4.

# %%
# TODO — your code here (questions 1, 2 and 4; question 3 is open-ended).


# %%
# @title Reference solution { display-mode: "form" }
# 1 + 2: extrapolation beyond the training range, and the ODE step count.
fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
for a, r in zip(ax[:2], [2.4, 3.5]):
    s = fm_sample(rnet, torch.full((1500, 1), r, device=dev), 2).cpu()
    a.plot(s[:, 0], s[:, 1], 'C0.', ms=1.5, alpha=.5)
    a.set(title=f'c = {r}' + ('  (in prior)' if r < 2.5 else '  (EXTRAPOLATED)'))
for a, st in zip(ax[2:], [1, 4, 16]):
    s = fm_sample(rnet, torch.full((1500, 1), 1.9, device=dev), 2, steps=st).cpu()
    a.plot(s[:, 0], s[:, 1], 'C1.', ms=1.5, alpha=.5)
    a.set(title=f'c = 1.9, steps = {st}')
for a in ax:
    a.set_aspect('equal'); a.set(xlim=(-4, 4), ylim=(-4, 4), xlabel=r'$\theta_1$')
fig.tight_layout()

# 4: late-time weighting of t. Same net size, same budget, only t changes.
def fm_loss_late(net, w1, cond=None):
    w0 = torch.randn_like(w1)
    t = torch.rand(len(w1), 1, device=w1.device) ** 0.5   # <-- pushes t toward 1
    wt = (1 - t) * w0 + t * w1
    return ((net(wt, t, cond) - (w1 - w0)) ** 2).mean()


rnet_late = VelocityNet(2, d_cond=1).to(dev)
opt = torch.optim.Adam(rnet_late.parameters(), lr=1e-3)
for step in range(4000):
    i = torch.randint(0, len(th_ring), (512,), device=dev)
    loss = fm_loss_late(rnet_late, th_ring[i], c_train[i])
    opt.zero_grad(); loss.backward(); opt.step()

fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
for a, st in zip(ax, [1, 4, 16]):
    s = fm_sample(rnet_late, torch.full((1500, 1), 1.9, device=dev), 2, steps=st).cpu()
    a.plot(s[:, 0], s[:, 1], 'C2.', ms=1.5, alpha=.5)
    a.set(title=f'late-t training, steps = {st}', aspect='equal',
          xlim=(-4, 4), ylim=(-4, 4), xlabel=r'$\theta_1$')
fig.tight_layout()

# %% [markdown]
# **Answers.** (1) Try it before assuming: here the network extrapolates to
# $c = 3.5$ rather well, returning a clean ring of about the right radius. It
# gets away with it because the family depends on $c$ in the simplest possible
# way — the shape is fixed and only its scale moves — so the smooth
# interpolation the network learned inside $[0.5, 2.5]$ happens to keep working
# outside. Do not generalize from this: nothing *forces* it, and for a family
# whose shape changes qualitatively with $c$ (say the number of modes)
# extrapolation fails outright. Beyond the prior you are trusting the network,
# not the data. (2) `steps=1` is a single Euler step, so the sample is just
# $\theta_0 + v_\phi(\theta_0, \tfrac12)$ — one smooth displacement of the
# Gaussian,
# which cannot open a hole in the middle, so you get a fuzzy disc instead of a
# ring. It sharpens fast and is essentially converged by ~16 steps. (4) With
# $t$ pushed toward 1 the
# few-step samples are visibly tighter, because the network spends its
# capacity where the shape actually forms. This is a one-line change that buys
# real accuracy — at LISA scale it moved our posteriors from "16 nats too wide"
# to "1 nat from optimal".

# %% [markdown]
# ---
# # Part 3 — From generative models to inference: SBI  *(~15 min)*
#
# Here is the whole idea of simulation-based inference, in one sentence:
# **take conditional flow matching and feed it pairs from a simulator.**
#
# In Part 2 the pairs $(w_1, c)$ were ring points and their radius. Now let
# $w_1 = \theta$ (the parameters we want to infer) and $c = x$ (the data we
# observe), and manufacture the pairs like this:
#
# $$\theta_i \sim p(\theta) \quad \text{(prior)}, \qquad
#   x_i \sim p(x \,|\, \theta_i) \quad \text{(simulator)} .$$
#
# Those $(\theta_i, x_i)$ are samples from the joint $p(x\,|\,\theta)\,p(\theta)$,
# which we know how to sample *forwards*. Train the conditional model on them
# and it learns the *other* factorization of the same joint — the **posterior**
# $q_\phi(\theta \,|\, x) \approx p(\theta \,|\, x)$. No likelihood evaluation,
# no MCMC, no Bayes' theorem applied by hand: the theorem is enforced purely by
# where the training pairs come from. And because the model is amortized in
# $c = x$ (the rings), one training run gives the posterior for *any*
# observation.
#
# ## The simulator: throwing a ball
#
# A ball is launched from the ground at angle $\alpha$ with speed $v$ on a flat
# planet, and we measure only **where it lands**:
#
# $$r(v, \alpha) = \frac{v^2}{g}\,\sin(2\alpha), \qquad
#   x = r(v, \alpha) + \sigma\,\varepsilon, \quad \varepsilon\sim\mathcal N(0,1).$$
#
# We want $\theta = (v, \alpha)$ from $x$. One number in, two parameters out —
# so this problem is *degenerate by construction*, and the shape of the
# degeneracy is not something we invented: it is whatever the curve
# $v^2\sin(2\alpha) = \text{const}$ happens to look like. Two features fall out
# of the physics for free:
#
# - **a curved ridge**, because a slower ball thrown at a better angle lands in
#   the same place as a faster ball thrown at a worse one;
# - **bimodality**, because $\sin(2\alpha)$ is symmetric about $45°$ — a lob at
#   $60°$ and a line drive at $30°$ are indistinguishable from the landing
#   point alone.
#
# This is the everyday reality of GW parameter estimation in miniature. (The
# distance–inclination degeneracy of a real black-hole binary is exactly this:
# a face-on binary far away looks like an edge-on binary nearby.)

# %%
G = 9.81
SIGMA_X = 0.4                                # measurement noise on the range [m]
BALL_LO = torch.tensor([8.0, 0.15], device=dev)      # v [m/s], alpha [rad]
BALL_HI = torch.tensor([12.0, np.pi / 2 - 0.15], device=dev)


def ball_range(theta):
    """Noise-free landing position r(v, alpha) -> (n,)."""
    v, alpha = theta[:, 0], theta[:, 1]
    return v ** 2 / G * torch.sin(2 * alpha)


def ball_sim(theta, n_throws=1):
    """Simulator: theta (n,2) -> mean of n_throws measured landings, (n,1)."""
    r = ball_range(theta)[:, None]
    noise = SIGMA_X * torch.randn(len(theta), n_throws, device=theta.device)
    return r + noise.mean(1, keepdim=True)   # noise on the mean: sigma/sqrt(n)


def draw_ball_prior(n):
    return BALL_LO + (BALL_HI - BALL_LO) * torch.rand(n, 2, device=dev)


THETA_TRUE = torch.tensor([[10.4, 0.62]], device=dev)   # v = 10.4 m/s, alpha = 36 deg
print(f'true parameters:  v = {THETA_TRUE[0, 0]:.1f} m/s, '
      f'alpha = {np.degrees(THETA_TRUE[0, 1].item()):.0f} deg')
print(f'noise-free range: r = {ball_range(THETA_TRUE).item():.2f} m')

# %% [markdown]
# The reference posterior is analytic here (uniform prior times a Gaussian
# likelihood on one number), so we can check the network against the truth.

# %%
def ball_true_logpost(vg, ag, x_obs, n_throws=1):
    """Exact log-posterior on a (v, alpha) grid, up to a constant."""
    V, A = torch.meshgrid(vg, ag, indexing='ij')
    r = V ** 2 / G * torch.sin(2 * A)
    return -0.5 * (x_obs - r) ** 2 / (SIGMA_X ** 2 / n_throws)


vg = torch.linspace(BALL_LO[0], BALL_HI[0], 300, device=dev)
ag = torch.linspace(BALL_LO[1], BALL_HI[1], 300, device=dev)


def plot_ball_truth(ax, x_obs, n_throws=1):
    lp = ball_true_logpost(vg, ag, x_obs, n_throws).cpu()
    p = np.exp(lp - lp.max())
    ax.contour(vg.cpu(), np.degrees(ag.cpu()), p.T, levels=[0.011, 0.14, 0.61],
               colors='k', linewidths=1)             # 3/2/1 sigma of a Gaussian
    ax.set(xlabel='v [m/s]', ylabel=r'$\alpha$ [deg]',
           xlim=(8, 12), ylim=(np.degrees(0.15), 90 - np.degrees(0.15)))

# %% [markdown]
# ## Train — with nothing new to write
#
# `VelocityNet`, `fm_loss`, `train_fm` and `fm_sample` are the Part 2
# functions, untouched. The only difference is that `cond` is now the output of
# a simulator. We z-score both sides first, which is standard practice: the
# flow works best when everything it sees is $O(1)$.
#
# We do it twice: once with a **single** throw, and once with the mean of
# **twenty** throws (which shrinks the noise on the measurement to
# $\sigma/\sqrt{20}$).

# %%
def zscore(a, mean, std):
    return (a - mean) / std


def run_ball_sbi(n_throws, n_train=40000, steps=3000):
    """Simulate a training set, fit q(theta | x), return posterior samples."""
    theta = draw_ball_prior(n_train)                  # theta_i ~ prior
    x = ball_sim(theta, n_throws)                     # x_i ~ p(x | theta_i)
    tmu, tsd = theta.mean(0), theta.std(0)
    xmu, xsd = x.mean(0), x.std(0)

    net = VelocityNet(2, d_cond=1).to(dev)            # w = theta (2), cond = x (1)
    train_fm(net, zscore(theta, tmu, tsd), zscore(x, xmu, xsd), steps=steps)

    x_obs = ball_sim(THETA_TRUE, n_throws)            # our observation
    so = zscore(x_obs, xmu, xsd).expand(6000, 1)
    post = fm_sample(net, so, 2) * tsd + tmu
    return post.cpu(), x_obs.item()


print('one throw:')
post_1, x_obs_1 = run_ball_sbi(n_throws=1)
print('twenty throws:')
post_20, x_obs_20 = run_ball_sbi(n_throws=20)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for a, post, xo, nt in [(ax[0], post_1, x_obs_1, 1), (ax[1], post_20, x_obs_20, 20)]:
    plot_ball_truth(a, xo, nt)
    a.plot(post[:, 0], np.degrees(post[:, 1]), 'C0.', ms=1.5, alpha=.25)
    a.plot(THETA_TRUE[0, 0].cpu(), np.degrees(THETA_TRUE[0, 1].cpu()), 'r*', ms=15)
    a.set_title(f'{nt} throw{"s" if nt > 1 else ""}:  x = {xo:.2f} m')
ax[1].set_ylabel('')
fig.tight_layout()

# %% [markdown]
# **What the posterior looks like.** Not a blob. A curved ridge that traces
# every $(v, \alpha)$ landing the ball where we saw it, folded back on itself
# about $45°$: the two arms meet at the slowest speed that can reach this far,
# $v = \sqrt{g\,x}$, thrown at exactly $45°$. Read it along a vertical line and
# it is **bimodal** — any speed above that minimum has two viable angles, one
# lob and one line drive. The network found all of this from a regression loss,
# having never been shown the range formula. Black contours: the exact
# posterior.
#
# **What more data does, and does not do.** Twenty throws instead of one
# shrinks the measurement noise by $\sqrt{20}$ and the arms become correspondingly
# thinner. But they do not merge, and the second mode does not go away: no
# amount of measuring *where* the ball lands can tell a $30°$ throw from a
# $60°$ one. That is a **structural** degeneracy, and the only cures are a
# different measurement (time of flight, apex height) or extra prior
# information. Statistical error shrinks with data; degeneracy does not.
#
# **Exercise 3.**
# 1. **Amortization.** The trained network covers *every* $x$, not just ours.
#    Use `run_ball_sbi` to get a network, then sample the posterior for two or
#    three different observed ranges without retraining — try a near-maximal
#    range ($x \approx v^2/g \approx 11$ m). What happens to the two modes as
#    the range approaches the largest achievable one?
# 2. **Break the degeneracy.** Change `ball_sim` to return *two* numbers, the
#    range **and** the time of flight $2 v \sin\alpha / g$, and set
#    `d_cond = 2`. What happens to the second mode? Why?
# 3. **Coverage.** Draw 200 parameter pairs from the prior, simulate one
#    observation each, sample 200 posterior draws per observation, and record
#    the fraction of posterior samples with $v$ below the true $v$. If the
#    posterior is calibrated, those fractions should be *uniform* on $[0,1]$ —
#    plot the histogram. This is the standard SBI validation test, and it needs
#    no reference posterior at all.

# %%
# TODO — your code here.


# %%
# @title Reference solution { display-mode: "form" }
# 1: amortization over the observed range, in one trained network.
theta_b = draw_ball_prior(40000)
x_b = ball_sim(theta_b, 1)
tmu_b, tsd_b = theta_b.mean(0), theta_b.std(0)
xmu_b, xsd_b = x_b.mean(0), x_b.std(0)
net_b = VelocityNet(2, d_cond=1).to(dev)
train_fm(net_b, zscore(theta_b, tmu_b, tsd_b), zscore(x_b, xmu_b, xsd_b),
         steps=6000, log=False)      # longer: question 3 needs an accurate posterior

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
for a, xo in zip(ax, [6.0, 9.5, 11.5]):
    xt = torch.tensor([[xo]], device=dev)
    post = (fm_sample(net_b, zscore(xt, xmu_b, xsd_b).expand(6000, 1), 2)
            * tsd_b + tmu_b).cpu()
    plot_ball_truth(a, xo)
    a.plot(post[:, 0], np.degrees(post[:, 1]), 'C0.', ms=1.5, alpha=.25)
    a.set_title(f'x = {xo} m')
for a in ax[1:]:
    a.set_ylabel('')
fig.tight_layout()

# 2: adding the time of flight kills the second mode.
def ball_sim2(theta, n_throws=1):
    """Range AND time of flight -> (n, 2)."""
    v, alpha = theta[:, 0], theta[:, 1]
    m = torch.stack([v ** 2 / G * torch.sin(2 * alpha),      # range
                     2 * v * torch.sin(alpha) / G], 1)       # time of flight
    sd = torch.tensor([SIGMA_X, 0.02], device=theta.device)  # per-channel noise
    return m + sd * torch.randn(len(theta), 2, device=theta.device) / np.sqrt(n_throws)


x_b2 = ball_sim2(theta_b, 1)
x2mu, x2sd = x_b2.mean(0), x_b2.std(0)
net_b2 = VelocityNet(2, d_cond=2).to(dev)
train_fm(net_b2, zscore(theta_b, tmu_b, tsd_b), zscore(x_b2, x2mu, x2sd), log=False)

x_obs2 = ball_sim2(THETA_TRUE, 1)
post2 = (fm_sample(net_b2, zscore(x_obs2, x2mu, x2sd).expand(6000, 2), 2)
         * tsd_b + tmu_b).cpu()

# 3: coverage -- the rank of the truth among posterior samples should be uniform.
N_SIM, N_POST = 1000, 100
theta_c = draw_ball_prior(N_SIM)
x_c = ball_sim(theta_c, 1)
post_c = fm_sample(net_b, zscore(x_c, xmu_b, xsd_b).repeat_interleave(N_POST, 0), 2)
post_c = (post_c * tsd_b + tmu_b).reshape(N_SIM, N_POST, 2)
ranks = (post_c[:, :, 0] < theta_c[:, None, 0]).float().mean(1).cpu()

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
plot_ball_truth(ax[0], x_obs2[0, 0].item())
ax[0].plot(post2[:, 0], np.degrees(post2[:, 1]), 'C2.', ms=1.5, alpha=.25)
ax[0].plot(THETA_TRUE[0, 0].cpu(), np.degrees(THETA_TRUE[0, 1].cpu()), 'r*', ms=15)
ax[0].set_title('range + time of flight: one mode left')
ax[1].hist(ranks, bins=10, range=(0, 1), color='C0', alpha=.8)
ax[1].axhline(N_SIM / 10, color='r', ls='--', lw=1, label='uniform expectation')
ax[1].set(xlabel=r'fraction of posterior samples below true $v$',
          ylabel='count', title='coverage: flat is calibrated')
ax[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# **Answers.** (1) As the observed range approaches the maximum achievable one,
# $v^2/g$ at $\alpha = 45°$, the two arms squeeze together and merge: only
# throws near $45°$ can reach that far, so the ambiguity disappears and the
# posterior becomes a single blob pinned to the corner of the prior. (2) The
# time of flight depends on $\sin\alpha$, not $\sin 2\alpha$, and is therefore
# *not* symmetric about $45°$ — one extra number breaks the reflection and one
# mode dies. Choosing what to measure is inference design, and it beats any
# amount of network tuning. (3) For an exact posterior the histogram is flat by
# construction, so its *shape* diagnoses the failure: a slope means a biased
# posterior, a hump in the middle means one that is too wide, and excess in
# both end bins — which is what you should see here, mildly — means one that is
# slightly too *narrow*, i.e. over-confident, so the truth lands in the tails
# more often than it should. That is a few-percent error, invisible in the
# contour plot above, and it needs no reference posterior to detect. Which is
# why this is *the* validation tool once the problems get real and no exact
# answer exists to compare against.

# %% [markdown]
# ---
# # Part 4 — A toy gravitational wave: compression + sequential zoom  *(~30 min)*
#
# Now the data looks like *our* data: a long noisy time series containing a
# chirp,
# $$d(t) = A\sin\!\big(2\pi(f_0 t + \tfrac12 \dot f t^2) + \varphi\big)
#          + n(t), \qquad n\sim\mathcal N(0,1),$$
# with 1024 samples. We infer $(f_0, \dot f)$; the phase $\varphi$ is a
# **nuisance parameter** — we randomize it in training and never infer it
# (exactly how the real LISA analysis treats its gauge angles).
#
# Two new problems appear:
# 1. $x$ has 1024 dimensions — we need to **compress** before conditioning;
# 2. with many cycles the likelihood is razor-sharp: an amortized net trained
#    from the wide prior turns out too blurry. We fix that by **zooming in
#    sequentially** (this is dynamic SBI).

# %%
N_T = 1024
tgrid = torch.linspace(0, 1, N_T, device=dev)
PRIOR_LO = torch.tensor([40., 0.], device=dev)      # f0 [cycles], fdot
PRIOR_HI = torch.tensor([80., 40.], device=dev)
AMP = 1.0
THETA_TRUE = torch.tensor([55.3, 17.8], device=dev)


def chirp_sim(theta, noise=1.0, phi=None):
    """theta: (n,2) -> data (n, 1024). phi randomized unless given."""
    if phi is None:
        phi = torch.rand(len(theta), 1, device=theta.device) * 2 * np.pi
    phase = 2 * np.pi * (theta[:, :1] * tgrid + 0.5 * theta[:, 1:2] * tgrid ** 2)
    return AMP * torch.sin(phase + phi) + noise * torch.randn(len(theta), N_T,
                                                              device=theta.device)


torch.manual_seed(7)
x_obs_chirp = chirp_sim(THETA_TRUE[None], phi=torch.tensor([[2.1]], device=dev))
snr = AMP * np.sqrt(N_T / 2)
print(f'signal-to-noise ratio ~ {snr:.0f}')

fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(tgrid.cpu(), x_obs_chirp[0].cpu(), lw=.5, label='observed (signal+noise)')
ax.plot(tgrid.cpu(), chirp_sim(THETA_TRUE[None], noise=0,
                               phi=torch.tensor([[2.1]], device=dev))[0].cpu(),
        'C1', lw=1, label='hidden signal')
ax.legend(loc='upper right'); ax.set(xlabel='t', ylabel='d(t)')
fig.tight_layout()

# %% [markdown]
# ## Step 1: compression with PCA
#
# We can't feed 1024 numbers into the conditioning — most of them are noise.
# Simulate *clean* signals from the prior, and find the directions in which
# they actually vary (principal component analysis = an SVD). The singular
# values tell us each component's signal-to-noise; we keep the top $K$.

# %%
def fit_pca(theta_bank, K=64):
    clean = chirp_sim(theta_bank, noise=0.0)
    mu = clean.mean(0)
    U, S, Vh = torch.linalg.svd(clean - mu, full_matrices=False)
    eigs = S / np.sqrt(len(clean) - 1)              # per-component SNR
    return mu, Vh[:K], eigs


def draw_prior(n):
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
    return (x - mu) @ V.T


def zscore(a, mean, std):
    return (a - mean) / std


x_bank = chirp_sim(theta_bank)                  # add noise: this is the training data

s_bank = summarize(x_bank, mu0, V0)
s_mu, s_sd = s_bank.mean(0), s_bank.std(0) + 1e-6
th_mu, th_sd = theta_bank.mean(0), theta_bank.std(0)

cnet = VelocityNet(2, K_PCA).to(dev)
train_fm(cnet, zscore(theta_bank, th_mu, th_sd),
         zscore(s_bank, s_mu, s_sd), steps=5000)

s_obs = zscore(summarize(x_obs_chirp, mu0, V0), s_mu, s_sd)
post0 = fm_sample(cnet, s_obs.expand(4000, K_PCA), 2) * th_sd + th_mu

# %%
def chirp_true_logpost(f0g, fdg, x_obs):
    """Exact posterior on a grid, phase marginalized analytically:
    the signal is linear in (cos phi, sin phi) -> 2-basis matched filter."""
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
p = np.exp(lp - lp.max())
# the exact posterior is far too small to see at this scale -- box it instead
ax.add_patch(plt.Rectangle((F0_LO, FD_LO), F0_HI - F0_LO, FD_HI - FD_LO,
                           fill=False, ec='k', lw=1.5,
                           label='exact posterior (inside this box)'))
ax.plot(*THETA_TRUE.cpu(), 'r*', ms=14, label='truth')
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
# accumulating its divergence (this is the one "advanced" cell; it is exactly
# what production codes like `falcon` do under the hood).

# %%
def fm_logprob(net, w1, cond, steps=64):
    """log q(w1|cond) via reverse ODE + divergence (exact, per-dimension autograd)."""
    w = w1.clone()
    logdet = torch.zeros(len(w1), device=w1.device)
    for i in range(steps):
        t = torch.full((len(w1), 1), 1 - (i + 0.5) / steps, device=w1.device)
        with torch.enable_grad():
            wg = w.requires_grad_(True)
            v = net(wg, t, cond)
            div = sum(torch.autograd.grad(v[:, d].sum(), wg, retain_graph=(d == 0))[0][:, d]
                      for d in range(w1.shape[1]))
        w = (w - v / steps).detach()
        logdet = logdet - div.detach() / steps
    base = (-0.5 * (w ** 2).sum(1) - 0.5 * w.shape[1] * np.log(2 * np.pi))
    return base + logdet

# %%
def sequential_chirp(n_rounds=8, gamma=0.5, n_keep=2048, refit_pca=True,
                     loss_fn=fm_loss, verbose=True):
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
        w1 = zscore(buf_theta, tmu, tsd)
        sc = zscore(s, smu, ssd)
        so = zscore(summarize(x_obs_chirp, mu, V), smu, ssd)
        # -- continue training conditional q_c(theta|s) and marginal q_m(theta)
        for net, opt, cond in [(qc, opt_c, sc), (qm, opt_m, torch.zeros_like(sc))]:
            for step in range(800):
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
    a.plot(*THETA_TRUE.cpu(), 'r*', ms=14)
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
# over-confidence, which you already met in Part 3's coverage test, but most of
# it is structural. The buffer converges to $L^\gamma\pi$, and we then train
# $q_c(\theta|s)$ **on that buffer** — so the likelihood enters twice, once
# through the buffer and once through the conditioning, and the readout behaves
# like $L^{1+\gamma}\pi$ rather than $L\pi$. Correcting it means importance
# reweighting the readout by $L^{-\gamma}$, which the ratio
# $\log q_c - \log q_m$ already gives us. Production codes do exactly this;
# we leave it out to keep the loop readable.
#
# **Exercise 4.**
# 1. Run with `gamma=0.1` and `gamma=1.0`. Which converges faster? Which is
#    riskier? (Think: what happens if an early, imperfect posterior estimate
#    excludes the truth — can a later round recover?)
# 2. Run with `refit_pca=False` (freeze the round-1 basis). How much slower is
#    the zoom? Look at the right panel to see why.
# 3. **Lower the SNR.** Set `AMP = 0.5` and re-run everything from the top of
#    Part 4. With a weaker signal the likelihood grows competitive secondary
#    maxima, and the zoom sometimes locks onto one and shrinks around it
#    confidently — in our tests, roughly one run in three. This is *the*
#    failure mode of sequential inference: it is not that the posterior is
#    wide, it is that it is narrow and wrong. What would you monitor to catch
#    it without knowing the answer?
# 4. **Fix the over-confidence.** Implement the $L^{-\gamma}$ reweighting
#    described above: draw from `qc`, evaluate `fm_logprob` under `qc` and
#    `qm`, and resample with weights $\exp(-\gamma(\log q_c - \log q_m))$.
#    Does the final width move toward the black contours?
# 5. *(bonus, from Exercise 2b.4)* Define `fm_loss_late` with
#    `t = torch.rand(...) ** (1/8)` for half the batch and pass it as
#    `loss_fn=`. At LISA scale this one-line change moved our posteriors from
#    "16 nats too wide" to "1 nat from mathematically optimal".

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
# The companion notebook **`lisa_sequential_new.ipynb`** runs exactly this on
# real LISA data: the LDC1-1 (Radler) massive black-hole binary, nine
# parameters, with the waveforms simulated *live* by `lisabeta` inside the zoom
# loop. It uses the same `fm_loss`, the same `VelocityNet`, the same
# `fm_sample` and `fm_logprob` you have here, and it runs the loop for four
# rounds — a minute or so per round on a Colab T4.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_sequential_new.ipynb)
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
        plt.figure(i).savefig(f'tutorial_fig_{i:02d}.png', dpi=110,
                              bbox_inches='tight')
    print(f'saved {len(plt.get_fignums())} figures')

# %% [markdown]
# *Generated for the LISA SBI tutorial, 2026-07-27. Built and battle-tested on
# the LDC1-1 MBHB analysis campaign of July 2026.*
