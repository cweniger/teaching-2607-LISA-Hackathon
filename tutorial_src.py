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
# PyTorch, and the companion notebook `lisa_sequential_new.ipynb` runs the very
# same functions on real LISA data.
#
# > **Colab setup:** Runtime → Change runtime type → **T4 GPU**, then run all
# > cells top to bottom. Everything also works on CPU, just slower.

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
# # Part 1 — Neural networks: fitting functions  *(~25 min)*
#
# ## The network
#
# Before any statistics, we need the object everything else is built from.
#
# A **multi-layer perceptron** (MLP) is a computer program that turns an input
# vector into an output vector. The same object is called a *feed-forward
# neural network*, a *dense network*, or a *fully connected network* — four
# names, one thing. It is the centrepiece of modern machine learning:
# convolutional networks, transformers, and the flows we build in Part 2 are
# all MLPs with extra structure bolted on, and every one of them still has
# ordinary dense layers inside.
#
# The program is a chain of **affine maps**, each followed by an elementwise
# **nonlinearity** $g$. For an input $x \in \mathbb R^{d_{\rm in}}$, three
# hidden layers of width $H$, and an output in $\mathbb R^{d_{\rm out}}$:
#
# $$\begin{aligned}
#   h^{(1)} &= g\big(W^{(1)} x + b^{(1)}\big),
#     & W^{(1)} &\in \mathbb R^{H \times d_{\rm in}}, & b^{(1)} &\in \mathbb R^{H} \\
#   h^{(2)} &= g\big(W^{(2)} h^{(1)} + b^{(2)}\big),
#     & W^{(2)} &\in \mathbb R^{H \times H}, & b^{(2)} &\in \mathbb R^{H} \\
#   h^{(3)} &= g\big(W^{(3)} h^{(2)} + b^{(3)}\big),
#     & W^{(3)} &\in \mathbb R^{H \times H}, & b^{(3)} &\in \mathbb R^{H} \\
#   \hat y  &= W^{(4)} h^{(3)} + b^{(4)},
#     & W^{(4)} &\in \mathbb R^{d_{\rm out} \times H},
#     & b^{(4)} &\in \mathbb R^{d_{\rm out}}
# \end{aligned}$$
#
# Written compactly, $\hat y = \mathrm{MLP}_\phi(x)$, where the **parameters**
# $\phi = \{W^{(1)}, b^{(1)}, \dots, W^{(4)}, b^{(4)}\}$ are all the numbers
# the program contains. Two details that matter:
#
# - **The nonlinearity is what makes it more than one big matrix.** A chain of
#   affine maps is itself an affine map, so without $g$ the whole network would
#   collapse to a single $Wx + b$. The default is
#   $g(z) = \mathrm{ReLU}(z) = \max(z, 0)$, which builds piecewise-linear
#   outputs with visible kinks; smooth choices like `torch.tanh`,
#   `nn.functional.gelu` or `torch.selu` build smooth outputs instead. The
#   read-out has *no* nonlinearity, so $\hat y$ is free to take any value.
# - **Initialization.** `nn.Linear` fills each $W$ and $b$ with small random
#   numbers (uniform in $\pm 1/\sqrt{\text{fan-in}}$ by default), so an
#   untrained network is already a perfectly valid — just useless — function.
#   Training is nothing more than moving those numbers.

# %%
class MLP(nn.Module):
    """A dense feed-forward network from R^d_in to R^d_out."""

    def __init__(self, d_in=1, d_out=1, hidden=256, act=torch.relu):
        super().__init__()
        self.act = act                             # torch.relu, torch.tanh, ...
        self.fc1 = nn.Linear(d_in, hidden)         # W1: (hidden, d_in)
        self.fc2 = nn.Linear(hidden, hidden)       # W2: (hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)       # W3: (hidden, hidden)
        self.out = nn.Linear(hidden, d_out)        # W4: (d_out, hidden)

    def forward(self, x):           # x: (n, d_in) — n points, d_in features each
        h = self.act(self.fc1(x))   # W1 @ x + b1, then g   -> (n, hidden)
        h = self.act(self.fc2(h))   # W2 @ h + b2, then g   -> (n, hidden)
        h = self.act(self.fc3(h))   # W3 @ h + b3, then g   -> (n, hidden)
        y = self.out(h)             # W4 @ h + b4, no g     -> (n, d_out)
        return y

# %% [markdown]
# Here is what three freshly initialized networks compute, before any
# training. Random parameters give random functions — that is the raw material
# gradient descent will shape.

# %%
xg = torch.linspace(-1, 1, 400)[:, None]        # a grid of 400 inputs, shape (400, 1)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
for s in range(3):
    torch.manual_seed(s)
    ax.plot(xg, MLP(1, 1, hidden=256)(xg).detach(), lw=1.2, label=f'seed {s}')
ax.set(xlabel='x', ylabel=r'$\hat y$', title='untrained networks are random functions')
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Fitting with gradient descent
#
# Given input–output pairs $(x_i, y_i)$, we want the network to reproduce
# them. Quantify the mismatch with the **mean squared error**,
#
# $$\mathcal L(\phi) = \frac{1}{N} \sum_{i=1}^{N}
#   \big\|\,\mathrm{MLP}_\phi(x_i) - y_i\,\big\|^2 ,$$
#
# and minimize it over all weights and biases at once by **gradient descent**
# — an iterative update that repeatedly steps in the direction of steepest
# descent,
#
# $$\phi_{k+1} = \phi_k - \eta\, \nabla_\phi \mathcal L(\phi_k),
#   \qquad k = 0, 1, 2, \dots$$
#
# starting from the random initialization $\phi_0$, with **learning rate**
# $\eta$ setting the step size. Each iteration $k$ is one *epoch* below:
#
# ```text
# ALGORITHM  fit(net, {(x_i, y_i)}, epochs, eta)
# ──────────────────────────────────────────────────────────────
# phi ← all trainable parameters of net (every W and b)
# opt ← Adam(phi, learning rate eta)      # opt holds pointers to phi
#
# for epoch = 1 … epochs:
#     y_pred_i ← net(x_i)   for all i     # forward pass
#     L  ← (1/N) Σ_i ||y_pred_i − y_i||²  # scalar loss
#     g  ← ∂L/∂phi                        # backward pass (backpropagation)
#     phi ← phi − eta·g                   # update phi in place (Adam variant)
#
# return net                              # phi now (locally) minimizes L
# ──────────────────────────────────────────────────────────────
# ```
#
# We use **Adam**, a gradient-descent variant that adapts the step size per
# parameter — the loop structure is exactly the one above. Alongside the
# training loss we track a **validation loss**: the same MSE on held-out data
# the network never trains on, which measures generalization rather than
# memorization.

# %%
def fit(net, x, y, x_val, y_val, epochs, lr=1e-3):
    # net.parameters() is the collection of all trainable tensors of the
    # network — the W's and b's of the four Linear layers. optim.Adam simply
    # holds pointers to these tensors: "training" means updating them in place.
    print('parameters handed to the optimizer:')
    for name, p in net.named_parameters():
        print(f'  {name:12s} {tuple(p.shape)}')
    print(f'total: {sum(p.numel() for p in net.parameters())} numbers\n')
    # (eps damps Adam's adaptive step once the training gradients get tiny —
    # without it, full-batch Adam on few points goes unstable late in training)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-3)

    hist, best_val, snap_val = [], np.inf, None
    for ep in range(epochs):
        # forward pass: predictions on the training set -> scalar loss L
        loss = ((net(x) - y) ** 2).mean()
        # reset all gradients to zero (PyTorch *accumulates* them by default)
        opt.zero_grad()
        # backward pass: fills p.grad = dL/dp for every parameter p
        loss.backward()
        # gradient step: update every parameter in place using its .grad
        opt.step()
        with torch.no_grad():   # book-keeping only — no gradients needed
            val = ((net(x_val) - y_val) ** 2).mean()
            hist.append((loss.item(), val.item()))
            # keep a copy of the weights whenever the validation loss improves
            if val < best_val:
                best_val, snap_val = val.item(), copy.deepcopy(net.state_dict())
        if (ep + 1) % 1000 == 0:
            print(f'epoch {ep + 1:5d}:  train {loss:.4f}   val {val:.4f}')
    return np.array(hist), snap_val

# %% [markdown]
# ## Example: approximating a trigonometric function
#
# **The generative model.** Our data are $N$ noisy observations of a smooth
# function $f$:
# $$y_i = f(x_i) + \epsilon_i, \qquad f(x) = \sin(2\pi x), \qquad
#   \epsilon_i \sim \mathcal N(0, \sigma^2),$$
# with inputs $x_i \sim U(-1, 1)$ and noise level $\sigma = 0.2$. The network
# only ever sees the pairs $(x_i, y_i)$ — it knows neither $f$ nor which part
# of each $y_i$ is signal and which part is noise $\epsilon_i$.

# %%
def make_data(n, sigma=0.2, seed=0):
    """n samples of the generative model  y = sin(2 pi x) + epsilon."""
    torch.manual_seed(seed)
    x = torch.rand(n, 1) * 2 - 1                         # x_i ~ U(-1, 1)
    y = torch.sin(2 * np.pi * x) + sigma * torch.randn(n, 1)  # y_i = f(x_i) + eps_i
    return x, y

# %% [markdown]
# Draw the data and *look at it first* — always. The dashed curve and the
# shaded band are the generative model above; the network will only ever see
# the blue dots. (The gray dots are a second, held-out draw used for
# validation — the network never trains on them.)

# %%
N_TRAIN, SIGMA = 30, 0.2                        # <-- data knobs
WIDTH, EPOCHS = 256, 3000                       # <-- network knobs

x, y = make_data(N_TRAIN, sigma=SIGMA)                # training set
x_val, y_val = make_data(200, sigma=SIGMA, seed=1)    # held-out set
y_true = torch.sin(2 * np.pi * xg)                    # noise-free truth on the grid

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.fill_between(xg[:, 0], y_true[:, 0] - SIGMA, y_true[:, 0] + SIGMA,
                color='k', alpha=.10, lw=0, label=r'$f(x) \pm \sigma$')
ax.plot(xg, y_true, 'k--', lw=1.2, label=r'$f(x) = \sin(2\pi x)$')
ax.plot(x_val, y_val, '.', color='gray', ms=4, alpha=.5,
        label='validation data (held out)')
ax.plot(x, y, 'C0o', ms=7, label=f'training data ($N = {N_TRAIN}$)')
ax.set(xlabel='x', ylabel='y', title='the data the network gets to see')
ax.legend(fontsize=9)
fig.tight_layout()

# %%
net = MLP(1, 1, WIDTH)                          # try act=torch.tanh
hist, snap_val = fit(net, x, y, x_val, y_val, EPOCHS)

# %%
net_best = MLP(1, 1, WIDTH)                     # the weights at the best epoch
net_best.load_state_dict(snap_val)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax[0].plot(xg, y_true, 'k--', lw=1, label='truth')
ax[0].plot(x, y, 'C0o', ms=5, label='train data')
ax[0].plot(xg, net(xg).detach(), 'C1', lw=1.8, label=f'final fit (epoch {EPOCHS})')
ax[0].plot(xg, net_best(xg).detach(), 'C2', lw=1.4,
           label=f'best-validation fit (epoch {hist[:, 1].argmin() + 1})')
ax[0].set(xlabel='x', ylabel='y'); ax[0].legend(fontsize=8)
ax[1].semilogy(hist[:, 0], label='train loss')
ax[1].semilogy(hist[:, 1], label='validation loss')
ax[1].axhline(SIGMA ** 2, color='gray', ls=':', lw=1, label=r'noise floor $\sigma^2$')
ax[1].set(xlabel='epoch', ylabel='MSE')
ax[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# **Reading the two panels.**
# - *Left:* the final fit passes through every noisy point and invents
#   structure between them that is not in $f$. The early-stopped fit — the same
#   network as it stood at its best validation epoch — is smoother and closer
#   to the truth. That is the network you should have kept.
# - *Right:* the training loss falls indefinitely, because on the training
#   points more optimization is always rewarded. The validation loss instead
#   bottoms out and turns back up; everything after that minimum is the
#   network memorizing noise, which makes predictions on new data worse. This
#   is **overfitting**. Note also that the validation loss can never go below
#   the noise floor $\sigma^2$: no fit of $f$, however perfect, predicts the
#   noise in held-out points.
# - *Order matters:* the noise-chasing wiggles are high-frequency and appear
#   only late — networks fit smooth structure first (**spectral bias**). That
#   is why early stopping works: the signal arrives early and the noise late.
#
# **Exercise 1a — read the plot.**
# 1. Which of the two fits would you trust to predict $y$ at a new $x$ — and
#    how could you make that choice in a *real* experiment, where the truth
#    (black dashed) is not available?
# 2. Where on the loss curves do the two fits live? What is the training loss
#    doing at the epoch where validation is best?
# 3. Why can the validation loss never drop below $\sigma^2 = 0.04$, even for a
#    network that has learned $f$ perfectly?
# 4. *(bonus)* Rebuild with a smooth activation, `MLP(1, 1, WIDTH,
#    act=torch.tanh)`, and retrain. How does the *character* of the
#    overfitting change?

# %%
# @title Answers to 1a { display-mode: "form" }
print("""
1. The early-stopped (green) one. You never need the truth to make that call:
   pick the epoch where the HELD-OUT loss is lowest. That is the whole idea.

2. The final fit sits at the right edge of the plot -- lowest train loss, worst
   validation loss. The early-stopped fit sits at the minimum of the orange
   curve. At that epoch the train loss is still falling, which is the point:
   'training is still improving' is not a reason to keep going.

3. The held-out y values are f(x) + eps. Even a network that knew f exactly
   cannot predict eps, so its MSE on those points is E[eps^2] = sigma^2. That
   floor is a property of the data, not of the network.

4. ReLU builds fits out of straight segments, so overfitting shows up as sharp
   kinks and corners. With tanh/GELU/SELU the network can only bend smoothly,
   so it overfits with gentle waves instead -- the noise still gets absorbed,
   it just looks more respectable while doing it.
""")

# %% [markdown]
# ## Early stopping
#
# Nobody reads a loss curve to pick the best epoch by hand. The standard
# mechanism does it automatically: track the validation loss, remember the
# weights whenever it improves, and stop once it has not improved for
# `patience` epochs — then rewind to the best weights. Three extra lines, and
# it makes `epochs` a knob you no longer have to tune: pass something huge and
# let patience decide.

# %%
def fit_early(net, x, y, x_val, y_val, epochs=100_000, lr=1e-3, patience=300,
              verbose=True):
    """Gradient descent that stops itself once validation stops improving."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-3)
    hist, best_val, best_ep, snap = [], np.inf, 0, None
    for ep in range(epochs):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = ((net(x_val) - y_val) ** 2).mean().item()
        hist.append((loss.item(), val))
        if val < best_val:                      # new best epoch: remember it
            best_val, best_ep = val, ep
            snap = copy.deepcopy(net.state_dict())
        if ep - best_ep > patience:             # no improvement for a while
            break
    net.load_state_dict(snap)                   # rewind to the best epoch
    if verbose:
        print(f'stopped at epoch {ep + 1}; best val loss {best_val:.4f} '
              f'at epoch {best_ep + 1}')
    return np.array(hist)

# %% [markdown]
# **Exercise 1b — your own function.** Fill in `my_function` below. That is the
# only thing you write: the next cell generates noisy data from it, fits an
# early-stopped network, and plots the result. Then push it until it breaks.
#
# 1. **Frequency.** Try $\sin(8\pi x)$, then $\sin(20 \pi x)$. At fixed
#    `N_TRAIN` the network stops resolving the oscillation — is that a lack of
#    capacity or a lack of data? Test your answer by changing each in turn.
# 2. **Sharp features.** Try a narrow bump, `torch.exp(-50 * x**2)`, or a step,
#    `torch.sign(x)`. Where does the fit struggle, and what does that tell you
#    about spectral bias?
# 3. **Capacity and depth.** Scan `WIDTH` over $\{2, 16, 256, 1024\}$, and add
#    or remove a hidden layer in `MLP`. With early stopping on, does the widest
#    network do worst? Compare with what the *final-epoch* network would give.
# 4. **Data.** Scan `N_TRAIN` over $\{10, 30, 100, 500\}$. How quickly does the
#    best validation loss approach the noise floor $\sigma^2$?

# %%
# TODO — your code here: map x (n, 1) -> f(x) (n, 1). No noise; that is added
# for you below.
def my_function(x):
    raise NotImplementedError('write your own target function')


# %%
# @title Reference solution { display-mode: "form" }
def my_function(x):                             # noqa: F811
    """A chirp plus a narrow bump: easy on the left, hard on the right."""
    return torch.sin(2 * np.pi * x * (1 + 2 * (x + 1))) + torch.exp(-60 * (x - 0.1) ** 2)


# %%
# Provided: noisy data from my_function, an early-stopped fit, and the plot.
def study_function(fn, n_train=N_TRAIN, sigma=SIGMA, width=WIDTH, seed=0):
    torch.manual_seed(seed)
    xt = torch.rand(n_train, 1) * 2 - 1
    yt = fn(xt) + sigma * torch.randn(n_train, 1)
    torch.manual_seed(seed + 1)
    xv = torch.rand(400, 1) * 2 - 1
    yv = fn(xv) + sigma * torch.randn(400, 1)
    net = MLP(1, 1, width)
    hist = fit_early(net, xt, yt, xv, yv)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(xg, fn(xg), 'k--', lw=1.2, label='truth')
    ax[0].plot(xt, yt, 'C0o', ms=4, alpha=.6, label=f'train (N = {n_train})')
    ax[0].plot(xg, net(xg).detach(), 'C2', lw=1.6, label='early-stopped fit')
    ax[0].set(xlabel='x', ylabel='y', title=f'WIDTH = {width}'); ax[0].legend(fontsize=8)
    ax[1].semilogy(hist[:, 0], label='train'); ax[1].semilogy(hist[:, 1], label='val')
    ax[1].axhline(sigma ** 2, color='gray', ls=':', lw=1, label=r'$\sigma^2$')
    ax[1].set(xlabel='epoch', ylabel='MSE'); ax[1].legend(fontsize=8)
    fig.tight_layout()
    return hist[:, 1].min()


study_function(my_function)


# %% [markdown]
# ## A harder shape: many numbers in, one parameter out
#
# So far the input was a single number. Nothing in the network required that —
# `MLP(30, 1, ...)` maps a 30-dimensional vector to a scalar just as happily —
# and *that* is the shape of every problem in the rest of this tutorial:
# **a lot of data in, a few parameters out.**
#
# Here is the smallest honest version. A sine of unknown frequency $\nu$ is
# sampled at 30 fixed times and buried in noise:
#
# $$d_j = \sin(2\pi\,\nu\,t_j) + \sigma\,\varepsilon_j, \qquad
#   t_j = \tfrac{j}{29},\quad j = 0 \dots 29, \qquad
#   \nu \sim U(1, 5)\ \text{cycles} .$$
#
# The network sees the 30 noisy samples $d$ and must return $\nu$. (We give it
# the rescaled target $(\nu-1)/4 \in [0,1]$, because networks work best when
# their inputs and outputs are $O(1)$ — this is the same z-scoring we will do
# in every later part.) Note the frequencies stay well below the Nyquist limit
# of 15 cycles for this grid, so nothing here is ambiguous in principle: a
# perfect estimator would do well.
#
# This is a chirp with the chirp switched off — the Part 4 problem, one part
# early.

# %%
N_GRID, NU_LO, NU_HI = 30, 1.0, 5.0
tj = torch.linspace(0, 1, N_GRID)


def sine_data(n, sigma=0.3, seed=0, random_phase=False):
    """n examples: d (n, 30) noisy sine samples, and the rescaled frequency."""
    torch.manual_seed(seed)
    nu = NU_LO + (NU_HI - NU_LO) * torch.rand(n, 1)          # nu ~ U(1, 5)
    phi = torch.rand(n, 1) * 2 * np.pi if random_phase else torch.zeros(n, 1)
    d = torch.sin(2 * np.pi * nu * tj + phi) + sigma * torch.randn(n, N_GRID)
    return d, (nu - NU_LO) / (NU_HI - NU_LO)                 # target in [0, 1]


d_tr, nu_tr = sine_data(300, seed=0)
d_va, nu_va = sine_data(2000, seed=1)

fig, ax = plt.subplots(1, 3, figsize=(13, 2.9), sharey=True)
for a, i in zip(ax, [0, 1, 2]):
    nu_i = NU_LO + (NU_HI - NU_LO) * nu_tr[i, 0]
    a.plot(tj, torch.sin(2 * np.pi * nu_i * tj), 'k--', lw=1, label='hidden signal')
    a.plot(tj, d_tr[i], 'C0o-', ms=4, lw=.8, label='what the network sees')
    a.set(xlabel='t', title=fr'$\nu$ = {nu_i:.2f} cycles')
ax[0].set_ylabel('d'); ax[0].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# Train a 30 → 1 network on 300 such examples. Because the input is no longer
# one-dimensional we cannot draw the fitted function; instead we plot
# **estimate against truth**, the standard way to look at a regression whose
# input is too big to visualize. A perfect estimator would put every point on
# the diagonal.

# %%
freq_net = MLP(N_GRID, 1, 256)                  # 30 inputs, 1 output
hist_f = fit_early(freq_net, d_tr, nu_tr, d_va, nu_va)


def to_nu(z):                                   # rescaled target -> cycles
    return NU_LO + (NU_HI - NU_LO) * z.detach()


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot([NU_LO, NU_HI], [NU_LO, NU_HI], 'k--', lw=1)
ax[0].plot(to_nu(nu_tr), to_nu(freq_net(d_tr)), 'C1.', ms=5, alpha=.6, label='train')
ax[0].plot(to_nu(nu_va), to_nu(freq_net(d_va)), 'C0.', ms=2, alpha=.3,
           label='validation')
rms = (to_nu(freq_net(d_va)) - to_nu(nu_va)).pow(2).mean().sqrt()
ax[0].set(xlabel=r'true $\nu$ [cycles]', ylabel=r'estimated $\nu$ [cycles]',
          title=f'validation RMSE {rms:.3f} cycles', aspect='equal')
ax[0].legend(fontsize=8, markerscale=2)
ax[1].semilogy(hist_f[:, 0], label='train loss')
ax[1].semilogy(hist_f[:, 1], label='validation loss')
ax[1].set(xlabel='epoch', ylabel='MSE (rescaled units)')
ax[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# **Reading the scatter.** Train and validation points lie on the same
# diagonal, so the network learned something about *frequency* rather than
# about these particular 300 examples. The spread grows toward high $\nu$:
# with only 30 samples per example a fast oscillation is measured from fewer
# points per cycle, so it is intrinsically harder — a property of the problem,
# not a defect of the fit.
#
# **A surprise worth pausing on.** Look at the loss curves. The training loss
# collapses by ten orders of magnitude — the network has memorized its 300
# examples right down into the noise — and yet the validation loss simply
# *plateaus*. It never turns back up, and early stopping buys almost nothing
# here. Overfitting is not automatic. This task hides one parameter behind 30
# strongly redundant inputs, so the only way to fit the training set at all is
# to actually extract the frequency; memorizing the leftover noise costs
# nothing on new data. Whether a network overfits is a property of the
# *problem* as much as of the network — which is precisely why you always watch
# a held-out set rather than reasoning about it in advance.
#
# **And what you did not get.** One number per example, with no indication of
# how sure the network is. Near the middle of the range with clean data that
# may be fine; at the edges, or at higher noise, the honest answer is a
# *distribution* over $\nu$. Producing one is what Part 2 is about.
#
# **Exercise 1c.**
# 1. **Noise.** Raise `sigma` to 0.8 and retrain. Does the scatter widen
#    uniformly, or worse at some frequencies than others?
# 2. **Data.** Drop the training set to 50 examples. Does a validation *turn*
#    finally appear, or does the scatter just widen?
# 3. **A nuisance parameter.** Rebuild with `random_phase=True`. Now the same
#    frequency produces completely different-looking data and the network has
#    to become phase-invariant on its own. How much accuracy does that cost?
#    (Part 4 does exactly this with the phase of its chirp, and the real LISA
#    analysis with two gauge angles: parameters you must marginalize over but
#    never infer.)

# %%
# @title Reference solution { display-mode: "form" }
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for a, (n, sg, rp, ttl) in zip(ax, [(300, 0.8, False, r'$\sigma$ = 0.8'),
                                    (50, 0.3, False, 'N = 50'),
                                    (300, 0.3, True, 'random phase')]):
    dt, nt = sine_data(n, sigma=sg, seed=0, random_phase=rp)
    dv, nv = sine_data(2000, sigma=sg, seed=1, random_phase=rp)
    m = MLP(N_GRID, 1, 256)
    h = fit_early(m, dt, nt, dv, nv, verbose=False)
    r = (to_nu(m(dv)) - to_nu(nv)).pow(2).mean().sqrt()
    a.plot([NU_LO, NU_HI], [NU_LO, NU_HI], 'k--', lw=1)
    a.plot(to_nu(nv), to_nu(m(dv)), 'C0.', ms=2, alpha=.3)
    a.set(xlabel=r'true $\nu$ [cycles]', ylabel=r'estimated $\nu$ [cycles]',
          aspect='equal', title=f'{ttl}   (RMSE {r:.3f})')
fig.tight_layout()

print("""
1. The scatter widens everywhere but much more at high frequency: fewer grid
   points per cycle means the noise hurts more where the signal oscillates
   fastest.
2. It mostly just widens. Even at N = 50 the validation loss plateaus rather
   than turning up -- the redundancy of the 30 inputs keeps saving us.
3. Randomising the phase costs real accuracy: the network must now learn a
   quantity that is invariant to a parameter it is never asked about, and it
   has to discover that invariance from examples alone.
""")

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
