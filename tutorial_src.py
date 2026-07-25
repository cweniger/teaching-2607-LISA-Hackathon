# %% [markdown]
# # From MLPs to LISA: simulation-based inference, hands-on
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_lisa_sbi.ipynb)
#
# **LISA tutorial — approximately 105 minutes.**
#
# | part | idea | new ingredient |
# |---|---|---|
# | 1 | fit a *function* with a neural network | MLPs, overfitting, early stopping |
# | 2 | fit a *distribution* | flow matching (FM), conditioning |
# | 3 | fit a *posterior*: feed FM pairs from a simulator | SBI, amortization |
# | 4 | a toy gravitational wave | data compression + **sequential** inference |
# | 5 | the real thing: a massive black-hole binary in LISA data | (pre-simulated) |
#
# Each part fixes the visible failure of the one before. All code is plain
# PyTorch — the same ~10-line loss function you meet in Part 2 analyzes the
# LISA signal in Part 5.
#
# > **Colab setup:** Runtime → Change runtime type → **T4 GPU**, then run all
# > cells top to bottom. Everything also works on CPU, just slower.

# %%
import copy
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
# # Part 1 — Neural networks fit functions (and overfit them)  *(~15 min)*
#
# A multi-layer perceptron (MLP) is just alternating linear maps and simple
# nonlinearities. It can approximate any smooth function — including the noise
# in your training data, which is the failure mode called **overfitting**.
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
# shaded band are the generative model from above; the network will only ever
# see the blue dots. (The gray dots are a second, held-out draw that we will
# use for validation later — the network never trains on them.)

# %%
N_TRAIN, SIGMA = 30, 0.2                        # <-- data knobs (Exercise 1b)

x, y = make_data(N_TRAIN, sigma=SIGMA)                # training set
x_val, y_val = make_data(200, sigma=SIGMA, seed=1)    # noisy held-out set
xg = torch.linspace(-1, 1, 400)[:, None]        # dense grid ...
y_true = torch.sin(2 * np.pi * xg)              # ... with the noise-free truth

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

# %% [markdown]
# ## The network
#
# An MLP is a chain of affine maps, each followed by an elementwise
# nonlinearity $g$. Ours takes a scalar $x$ through three hidden layers of
# width $H$ and reads out a scalar:
#
# $$\begin{aligned}
#   h^{(1)} &= g\big(W^{(1)} x + b^{(1)}\big),
#     & W^{(1)} &\in \mathbb R^{H \times 1}, & b^{(1)} &\in \mathbb R^{H} \\
#   h^{(2)} &= g\big(W^{(2)} h^{(1)} + b^{(2)}\big),
#     & W^{(2)} &\in \mathbb R^{H \times H}, & b^{(2)} &\in \mathbb R^{H} \\
#   h^{(3)} &= g\big(W^{(3)} h^{(2)} + b^{(3)}\big),
#     & W^{(3)} &\in \mathbb R^{H \times H}, & b^{(3)} &\in \mathbb R^{H} \\
#   \hat y  &= W^{(4)} h^{(3)} + b^{(4)},
#     & W^{(4)} &\in \mathbb R^{1 \times H}, & b^{(4)} &\in \mathbb R
# \end{aligned}$$
#
# or, in one line, $\;\hat y = \mathrm{MLP}_\phi(x)$ with the **parameters**
# $\phi = \{W^{(1)}, b^{(1)}, \dots, W^{(4)}, b^{(4)}\}$ — the $2H^2 + 4H + 1$
# numbers that training will adjust. Note the read-out has **no**
# nonlinearity: $\hat y$ must be free to take any real value.
#
# The nonlinearity $g$ is what makes this more than one big affine map (a
# chain of affine maps is just an affine map). By default
# $g(z) = \mathrm{ReLU}(z) = \max(z, 0)$, which builds piecewise-linear fits
# with visible kinks; smooth choices like `torch.tanh`, `nn.functional.gelu`
# or `torch.selu` build smoother fits (try them!). The code below is the
# formula above, line for line.

# %%
class MLP(nn.Module):
    """y = MLP(x): three hidden layers with nonlinearity, one linear read-out."""

    def __init__(self, hidden, act=torch.relu):
        super().__init__()
        self.act = act                        # nonlinearity: torch.relu (default),
                                              # torch.tanh, nn.functional.gelu, ...
        self.fc1 = nn.Linear(1, hidden)       # W1: (hidden, 1),      b1: (hidden,)
        self.fc2 = nn.Linear(hidden, hidden)  # W2: (hidden, hidden), b2: (hidden,)
        self.fc3 = nn.Linear(hidden, hidden)  # W3: (hidden, hidden), b3: (hidden,)
        self.out = nn.Linear(hidden, 1)       # W4: (1, hidden),      b4: (1,)

    def forward(self, x):           # x: (n, 1) — n points, 1 input feature
        h = self.act(self.fc1(x))   # W1 @ x + b1, then act    -> (n, hidden)
        h = self.act(self.fc2(h))   # W2 @ h + b2, then act    -> (n, hidden)
        h = self.act(self.fc3(h))   # W3 @ h + b3, then act    -> (n, hidden)
        y = self.out(h)             # W4 @ h + b4, NO act      -> (n, 1)
        return y                    # the predicted y for each input point

# %% [markdown]
# ## How the fit works
#
# We measure how well the network reproduces the training data with the
# **mean squared error** loss,
# $$\mathcal L(\phi) = \frac{1}{N} \sum_{i=1}^{N}
#   \big(\mathrm{MLP}_\phi(x_i) - y_i\big)^2 ,$$
# and minimize it over *all* weights and biases at once by **gradient
# descent** — an iterative update that repeatedly takes a small step in the
# direction of steepest descent,
# $$\phi_{k+1} = \phi_k - \eta\, \nabla_\phi \mathcal L(\phi_k),
#   \qquad k = 0, 1, 2, \dots$$
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
#     L  ← (1/N) Σ_i (y_pred_i − y_i)²    # scalar loss
#     g  ← ∂L/∂phi                        # backward pass (backpropagation)
#     phi ← phi − eta·g                   # update phi in place (Adam variant)
#
# return net                              # phi now (locally) minimizes L
# ──────────────────────────────────────────────────────────────
# ```
#
# We use **Adam**, a gradient-descent variant that adapts the step size per
# parameter — the loop structure is exactly the one above.
#
# While training runs we monitor two numbers every epoch: the **training
# loss** $\mathcal L$ itself, and the **validation loss** — the same MSE
# evaluated on the held-out data set. The network never trains on those
# points, so the validation loss measures how well it *generalizes* rather
# than memorizes.

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

    hist = []
    best_val, snap_val = np.inf, None
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
            val = ((net(x_val) - y_val) ** 2).mean()   # held-out data
            hist.append((loss.item(), val.item()))
            # keep a copy of the weights whenever the validation loss improves —
            # this snapshot is the network early stopping would return
            if val < best_val:
                best_val, snap_val = val.item(), copy.deepcopy(net.state_dict())
        if (ep + 1) % 1000 == 0:
            print(f'epoch {ep + 1:5d}:  train {loss:.4f}   val {val:.4f}')
    return np.array(hist), snap_val

# %%
WIDTH, EPOCHS = 256, 3000                       # <-- network knobs (Exercise 1b)

net = MLP(WIDTH)                                # try MLP(WIDTH, act=torch.tanh)
hist, snap_val = fit(net, x, y, x_val, y_val, EPOCHS)

# %%
# the network an early-stopper would have kept
net_best = MLP(WIDTH)
net_best.load_state_dict(snap_val)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax[0].plot(xg, y_true, 'k--', lw=1, label='truth')
ax[0].plot(x, y, 'C0o', ms=5, label='train data')
ax[0].plot(xg, net(xg).detach(), 'C1', lw=1.8,
           label=f'final fit (epoch {EPOCHS})')
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
# - *Left:* the final fit threads the noisy points and invents wiggles between
#   them; the early-stopped fit (best validation epoch) is smoother and closer
#   to the truth — that is the network you *should* have kept.
# - *Right:* the training loss falls forever — more epochs always help *on the
#   training points*. The validation loss instead bottoms out and turns up:
#   from that point on the network is memorizing training noise, which makes
#   predictions on *new* data worse. Note it can never beat the noise floor
#   $\sigma^2$ — even a perfect fit of $f$ cannot predict the noise in the
#   held-out points.
# - *Order matters:* the noise-chasing wiggles are high-frequency, and they
#   appear only late — MLPs fit smooth structure first (**spectral bias**).
#   That is why early stopping works: it keeps the signal, drops the noise.
#
# **Exercise 1a — read the plot.**
# 1. Which of the two fits in the left panel would you trust to predict $y$ at
#    a new $x$ — and how could you make that choice in a *real* experiment,
#    where the truth (black dashed) is not available?
# 2. Connect the panels: where on the loss curves do the two fits live? What
#    is the training loss doing at the epoch where validation is best?
# 3. Why can the validation loss never drop below $\sigma^2 = 0.04$, even if
#    the network learned $f$ perfectly?
# 4. *(bonus)* Rebuild the network with a smooth activation,
#    `net = MLP(WIDTH, act=torch.tanh)` (or `nn.functional.gelu`,
#    `torch.selu`), and retrain. How does the *character* of the overfitting
#    wiggles change?

# %% [markdown]
# **Exercise 1b — finish the early stopper.** Nobody stares at loss curves to
# pick the best epoch by hand — training should stop *itself*. The function
# below is `fit` with the standard mechanism built in: it remembers the
# best-validation weights (`snap`) and rewinds to them at the end. One thing
# is missing: the actual **stopping condition**. Add it — one or two lines.

# %%
def fit_early(net, x, y, x_val, y_val, epochs, lr=1e-3, patience=300):
    """Like fit(), but stops itself once validation stops improving."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-3)
    hist, best_val, best_ep, snap = [], np.inf, 0, None
    for ep in range(epochs):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = ((net(x_val) - y_val) ** 2).mean().item()
        hist.append((loss.item(), val))
        if val < best_val:                     # new best epoch: remember it
            best_val, best_ep = val, ep
            snap = copy.deepcopy(net.state_dict())
        # TODO — your code here (1-2 lines), then delete the raise below.
        # Break out of the loop once the last improvement (epoch best_ep) lies
        # more than `patience` epochs in the past.
        raise NotImplementedError('implement the early-stopping condition')
    net.load_state_dict(snap)                  # rewind to the best epoch
    print(f'stopped at epoch {ep + 1}; best val loss {best_val:.4f} '
          f'at epoch {best_ep + 1}')
    return np.array(hist)

# %%
# @title Reference solution { display-mode: "form" }
def fit_early(net, x, y, x_val, y_val, epochs, lr=1e-3, patience=300):  # noqa: F811
    """Like fit(), but stops itself once validation stops improving."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-3)
    hist, best_val, best_ep, snap = [], np.inf, 0, None
    for ep in range(epochs):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = ((net(x_val) - y_val) ** 2).mean().item()
        hist.append((loss.item(), val))
        if val < best_val:                     # new best epoch: remember it
            best_val, best_ep = val, ep
            snap = copy.deepcopy(net.state_dict())
        if ep - best_ep > patience:            # <-- the added condition
            break
    net.load_state_dict(snap)                  # rewind to the best epoch
    print(f'stopped at epoch {ep + 1}; best val loss {best_val:.4f} '
          f'at epoch {best_ep + 1}')
    return np.array(hist)


net_es = MLP(WIDTH)
hist_es = fit_early(net_es, x, y, x_val, y_val, 100_000)  # epochs: huge, on purpose

# %% [markdown]
# **Exercise 1c — now use it.** With early stopping in place `EPOCHS` is no
# longer a knob you have to tune: pass something huge and let `patience`
# decide. So scan the knobs that actually matter, recording the best
# validation loss for each setting.
# 1. Scan the *network*: `WIDTH` $\in \{2, 16, 256, 1024\}$. How does the best
#    validation loss depend on capacity — is the biggest network the worst
#    one? (Compare with what the *final-epoch* network would have given.)
# 2. Scan the *data*: `N_TRAIN` $\in \{10, 30, 100, 500\}$. How quickly does
#    the best validation loss approach the noise floor $\sigma^2$?

# %%
# TODO — your code here.
# Hint: loop over the values, build a fresh MLP each time (seed it first for a
# fair comparison), call fit_early, and collect hist[:, 1].min(). For the
# N_TRAIN scan you also need fresh data: make_data(n, sigma=SIGMA).


# %%
# @title Reference solution { display-mode: "form" }
print('WIDTH scan (N_TRAIN = 30):')
best_w = []
for w in [2, 16, 256, 1024]:
    torch.manual_seed(0)                        # same init draw for fairness
    h = fit_early(MLP(w), x, y, x_val, y_val, 100_000)
    best_w.append(h[:, 1].min())
    print(f'  WIDTH {w:5d}:  best val {best_w[-1]:.4f}   '
          f'(final epoch would give {h[-1, 1]:.4f})')

print('\nN_TRAIN scan (WIDTH = 256):')
best_n = []
for n in [10, 30, 100, 500]:
    xn, yn = make_data(n, sigma=SIGMA)
    torch.manual_seed(0)
    h = fit_early(MLP(256), xn, yn, x_val, y_val, 100_000)
    best_n.append(h[:, 1].min())
    print(f'  N_TRAIN {n:4d}:  best val {best_n[-1]:.4f}')

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].loglog([2, 16, 256, 1024], best_w, 'C0o-')
ax[0].set(xlabel='WIDTH', ylabel='best validation MSE',
          title='capacity (with early stopping)')
ax[1].loglog([10, 30, 100, 500], best_n, 'C1o-')
ax[1].set(xlabel='N_TRAIN', title='training-set size')
for a in ax:
    a.axhline(SIGMA ** 2, color='gray', ls=':', lw=1, label=r'noise floor $\sigma^2$')
    a.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# Two lessons. **Capacity is not the enemy:** with early stopping the wide
# networks are no worse than the medium one (`WIDTH=2` *underfits* — too few
# kinks to make a sine — while 256 and 1024 land in the same place). Without
# early stopping the big ones would keep drifting upward. **Data is what
# buys accuracy:** the best validation loss falls steadily toward the noise
# floor $\sigma^2$ as $N$ grows, and no architecture choice substitutes for
# it. Both lessons carry over verbatim to Part 4, where "more data" means
# "more simulations".

# %% [markdown]
# ---
# # Part 2 — Modeling distributions with flow matching  *(~25 min)*
#
# Part 1 fitted a *function*: one number $y$ for each input $x$. Now we fit a
# **distribution**. We are handed samples from some unknown density $q(w)$ and
# want a machine that produces *more* samples from it — a **generative model**.
#
# The trick that has taken over the field is to build the sampler out of a
# **flow**: start from an easy distribution (a unit Gaussian) and move the
# points continuously until they are distributed like $q$. The motion is
# described by a **velocity field** $v_\phi(w, t)$ — an MLP exactly like
# Part 1's, taking a position $w$ and a time $t \in [0,1]$ — and "training the
# generative model" means fitting that velocity field.
#
# ## The mechanics, in three equations
#
# **1. Training.** Pick a random time $t$, a random noise point
# $w_0 \sim \mathcal N(0, I)$ and a random data sample $w_1 \sim q$. Place
# yourself on the straight line between the two at time $t$, and regress the
# velocity onto the direction that points from $w_0$ to $w_1$:
#
# $$\mathcal L(\phi) = \mathbb E_{t,\, w_0,\, w_1}
#   \Big[\;\big\| \, v_\phi\big(\underbrace{(1-t)\,w_0 + t\,w_1}_{\textstyle w_t},
#   \; t\big) \; - \; (w_1 - w_0) \, \big\|^2 \;\Big],
#   \qquad t \sim U(0,1).$$
#
# Note what is *absent*: no integration, no sampling from the model, no
# density, no Jacobian. It is a plain regression loss — Part 1's `fit` with a
# fancier target.
#
# **2. Sampling.** Draw a noise point and integrate the learned velocity field
# from $t = 0$ to $t = 1$:
#
# $$w(0) = w_0 \sim \mathcal N(0, I), \qquad
#   \frac{\mathrm d w}{\mathrm d t} = v_\phi\big(w(t),\, t\big), \qquad
#   w(1) \sim q_\phi \;\approx\; q .$$
#
# We integrate with plain Euler steps, $w \mathrel{+}= v_\phi(w,t)\,\Delta t$.
#
# **3. Evaluation.** If you also need the *density* of a point (we will, in
# Part 4), integrate the same ODE **backwards** from $w_1$ while accumulating
# the divergence of the velocity field:
#
# $$\log q_\phi(w_1) = \log \mathcal N\big(w(0);\, 0, I\big)
#   \; - \; \int_0^1 \nabla \!\cdot\! v_\phi\big(w(t),\, t\big)\, \mathrm d t .$$
#
# *Why* regressing onto straight lines between unrelated random pairs produces
# a velocity field whose flow transports $\mathcal N(0,I)$ to $q$ is genuinely
# non-obvious, and we will not derive it here — see Lipman et al.
# (arXiv:2210.02747), Liu et al. (arXiv:2209.03003) and Albergo &
# Vanden-Eijnden (arXiv:2209.15571). For our purposes it is a black box with
# three knobs, and the three equations above are all of them.

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
    w1 = 6 * torch.rand(n, device=dev) - 3                  # spread along the arc
    w2 = 0.3 * torch.rand(n, device=dev).mul(2).sub(1) \
        + 0.3 * w1 ** 2 - 1.2                               # bend it
    return torch.stack([w1, w2 + 0.15 * torch.randn(n, device=dev)], 1)


def target_spiral(n):
    """n samples along an Archimedean spiral -> (n, 2)."""
    a = 3 * np.pi * torch.rand(n, 1, device=dev).sqrt()   # angle
    r = 0.45 * a                                          # radius grows with angle
    c = torch.cat([r * a.cos(), r * a.sin()], 1)
    return c + 0.12 * torch.randn(n, 2, device=dev)       # thicken the arm


w_banana = target_banana(20000)
w_spiral = target_spiral(20000)

fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
for a, w, ttl in [(ax[0], w_banana, 'target: banana'),
                  (ax[1], w_spiral, 'target: spiral')]:
    a.plot(w[:4000, 0].cpu(), w[:4000, 1].cpu(), 'k.', ms=1, alpha=.3)
    a.set(title=ttl, xlabel=r'$w_1$', ylabel=r'$w_2$', aspect='equal')
fig.tight_layout()

# %% [markdown]
# ## The implementation
#
# Three short functions, and they are the ones that will still be running in
# Part 5 on real LISA data. `cond` is the conditioning input; leave it `None`
# for now (we use it in the second half of this part).

# %%
# Generic MLP helper for the rest of the notebook: same idea as Part 1's MLP
# class, but with configurable input/output dimensions and depth.
def mlp(d_in, d_out, hidden, layers):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, d_out))


def fm_loss(net, w1, cond=None):
    """Equation 1. This exact function is reused in Parts 3, 4 and 5."""
    w0 = torch.randn_like(w1)                       # noise point
    t = torch.rand(len(w1), 1, device=w1.device)    # random time
    wt = (1 - t) * w0 + t * w1                      # point on the straight line
    v = net(wt, t, cond)                            # predicted velocity there
    return ((v - (w1 - w0)) ** 2).mean()            # regress onto w1 - w0


class VelocityNet(nn.Module):
    """Velocity field v(w, t | cond): an MLP with a Fourier embedding of t."""

    def __init__(self, d_w, d_cond=0, hidden=128, layers=3):
        super().__init__()
        self.freqs = torch.tensor([1., 2., 4., 8.])
        self.net = mlp(d_w + 9 + d_cond, d_w, hidden, layers)   # 9 = 1 + 4 + 4

    def forward(self, w, t, cond=None):
        ft = 2 * np.pi * t * self.freqs.to(t.device)
        temb = torch.cat([t, ft.sin(), ft.cos()], 1)   # t, sin(2 pi f t), cos(...)
        parts = [w, temb] if cond is None else [w, temb, cond]
        return self.net(torch.cat(parts, 1))


@torch.no_grad()
def fm_sample(net, cond, d_w, steps=64, n=None, return_path=False):
    """Equation 2: Euler-integrate dw/dt = v from t=0 (noise) to t=1 (samples)."""
    n = len(cond) if cond is not None else n
    device = cond.device if cond is not None else next(net.parameters()).device
    w = torch.randn(n, d_w, device=device)             # w(0) ~ N(0, I)
    path = [w.clone()]
    for i in range(steps):
        t = torch.full((n, 1), (i + 0.5) / steps, device=device)
        w = w + net(w, t, cond) / steps                # w += v * dt
        path.append(w.clone())
    return (w, torch.stack(path)) if return_path else w

# %%
def train_fm(net, w1, cond=None, steps=3000, batch=512, lr=1e-3, log=True):
    """Minimize fm_loss by Adam -- the same loop as Part 1's fit()."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(w1), (batch,), device=w1.device)
        loss = fm_loss(net, w1[i], None if cond is None else cond[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step + 1) % 1000 == 0:
            print(f'  step {step + 1}/{steps}  loss {loss.item():.3f}  '
                  f'[{time.time() - t0:.0f}s]')
    return net

# %% [markdown]
# ## Train it on the spiral

# %%
snet = VelocityNet(2).to(dev)                       # d_cond = 0: unconditional
train_fm(snet, w_spiral)

samp, path = fm_sample(snet, None, 2, n=4000, return_path=True)
tgrid_fm = np.linspace(0, 1, path.shape[0])

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
# (a) target vs model samples
ax[0].plot(w_spiral[:4000, 0].cpu(), w_spiral[:4000, 1].cpu(), 'k.', ms=1,
           alpha=.15, label='target')
ax[0].plot(samp[:, 0].cpu(), samp[:, 1].cpu(), 'C0.', ms=1, alpha=.3,
           label='flow-matching samples')
ax[0].set(title='the model learned the spiral', xlabel=r'$w_1$', ylabel=r'$w_2$')
ax[0].legend(markerscale=8, fontsize=8); ax[0].set_aspect('equal')
# (b) where each sample came from, and the route it took
p = path[:, :60].cpu()
ax[1].plot(p[:, :, 0], p[:, :, 1], ':', color='gray', lw=.7)
ax[1].plot(p[0, :, 0], p[0, :, 1], 'C2o', ms=4, label=r'base sample $w(0)$')
ax[1].plot(p[-1, :, 0], p[-1, :, 1], 'C0o', ms=4, label=r'final sample $w(1)$')
ax[1].set(title='60 trajectories of the learned flow', xlabel=r'$w_1$')
ax[1].legend(fontsize=8); ax[1].set_aspect('equal')
# (c) one coordinate as a function of t
ax[2].plot(tgrid_fm, path[:, :200, 0].cpu(), lw=.5, alpha=.5)
ax[2].set(title=r'$w_1$ along the flow', xlabel='t', ylabel=r'$w_1$')
fig.tight_layout()

# %% [markdown]
# **Reading the panels.**
# - *Left:* the model reproduces the spiral, arms and gaps included — from a
#   plain regression loss and 3000 Adam steps.
# - *Middle:* every final sample traces back to one Gaussian base point. Note
#   the routes are **curved**, even though training only ever used *straight*
#   lines between random pairs $(w_0, w_1)$ — the network learns the *average*
#   velocity over all pairs passing through a point, and the resulting flow
#   bends. Do not expect a trajectory to connect the pair it was trained on.
# - *Right:* the same thing as a function of $t$: a single Gaussian blob at
#   $t = 0$ progressively separating into the layered structure of the spiral
#   by $t = 1$. Almost all of the shape forms late, near $t = 1$ — remember
#   this, it comes back in Part 4.
#
# **Exercise 2a — your own distribution.** Write a sampler for a target of
# your choice and fit it. All the machinery is above; you only need to supply
# the samples. Ideas: two moons, a checkerboard, a ring, a mixture of a few
# Gaussians, your initials.

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
    w = 1.6 * torch.cat([top, bot]) + 0.09 * torch.randn(2 * (n // 2), 2, device=dev)
    return w[torch.randperm(len(w), device=dev)]    # shuffle: plots take w[:4000]


w_mine = my_target(20000)
mynet = VelocityNet(2).to(dev)
train_fm(mynet, w_mine, log=False)
samp_mine = fm_sample(mynet, None, 2, n=4000)

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.plot(w_mine[:4000, 0].cpu(), w_mine[:4000, 1].cpu(), 'k.', ms=1, alpha=.15,
        label='target')
ax.plot(samp_mine[:, 0].cpu(), samp_mine[:, 1].cpu(), 'C0.', ms=1, alpha=.3,
        label='flow-matching samples')
ax.set(xlabel=r'$w_1$', ylabel=r'$w_2$', title='your own distribution')
ax.legend(markerscale=8, fontsize=8); ax.set_aspect('equal')
fig.tight_layout()

# %% [markdown]
# ## Conditional flow matching
#
# One more ingredient and we are done. Very often we do not want *one*
# distribution but a *family* of them, indexed by some input $c$ — that is a
# **conditional** density $q(w \,|\, c)$. The change is minimal: feed $c$ to
# the velocity field alongside $w$ and $t$,
#
# $$\mathcal L(\phi) = \mathbb E_{t,\, w_0,\, (w_1, c)}
#   \Big[\;\big\| \, v_\phi\big((1-t)\,w_0 + t\,w_1,\; t \,\big|\, c\big)
#   \; - \; (w_1 - w_0) \, \big\|^2 \;\Big],$$
#
# where the pairs $(w_1, c)$ are drawn **jointly**: each training sample comes
# with the $c$ it belongs to. Sampling is unchanged except that you say which
# $c$ you want. In code, that is the `cond` argument we have been passing as
# `None` — nothing else moves.
#
# Demonstration: rings of varying radius. The condition $c$ is the radius,
# the target $q(w\,|\,c)$ is a ring of that radius.

# %%
def target_ring(radius):
    """radius: (n, 1) -> (n, 2) points on a ring of that radius."""
    a = 2 * np.pi * torch.rand_like(radius)
    return (torch.cat([radius * a.cos(), radius * a.sin()], 1)
            + 0.06 * torch.randn(len(radius), 2, device=radius.device))


c_train = 0.5 + 2.0 * torch.rand(40000, 1, device=dev)   # radii in [0.5, 2.5]
w_ring = target_ring(c_train)

rnet = VelocityNet(2, d_cond=1).to(dev)                  # <-- the only change
train_fm(rnet, w_ring, c_train, steps=4000)

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
ax[0].plot(w_ring[:6000, 0].cpu(), w_ring[:6000, 1].cpu(), 'k.', ms=1, alpha=.2)
ax[0].set(title=r'training data: all radii mixed together', xlabel=r'$w_1$',
          ylabel=r'$w_2$')
for r, col in zip([0.7, 1.3, 1.9, 2.4], ['C0', 'C1', 'C2', 'C3']):
    c = torch.full((1500, 1), r, device=dev)
    s = fm_sample(rnet, c, 2).cpu()
    ax[1].plot(s[:, 0], s[:, 1], '.', color=col, ms=1.5, alpha=.5, label=f'c = {r}')
ax[1].set(title='one network, four requested radii', xlabel=r'$w_1$')
ax[1].legend(markerscale=6, fontsize=8)
for a in ax:
    a.set_aspect('equal'); a.set(xlim=(-3, 3), ylim=(-3, 3))
fig.tight_layout()

# %% [markdown]
# The training set (left) is a filled disc — no individual ring is visible in
# it. Yet asking the trained network for $c = 0.7$ or $c = 2.4$ returns a
# clean ring of exactly that radius (right). The network did not memorize four
# rings; it learned the *whole family* $q(w\,|\,c)$ at once, which is why a
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

# %% [markdown]
# ---
# # Part 3 — From generative models to inference: SBI  *(~15 min)*
#
# Here is the whole idea of simulation-based inference, in one sentence:
# **take conditional flow matching and feed it pairs from a simulator.**
#
# In Part 2 the pairs $(w_1, c)$ were points and their radius. Now let
# $w_1 = \theta$ (the parameters we want to infer) and $c = x$ (the data we
# observe), and generate the pairs like this:
#
# $$\theta_i \sim p(\theta) \quad \text{(prior)}, \qquad
#   x_i \sim p(x \,|\, \theta_i) \quad \text{(simulator)} .$$
#
# Those $(\theta_i, x_i)$ are samples from the joint $p(x\,|\,\theta)\,p(\theta)$,
# which we know how to sample *forwards*. Train the conditional model on them
# and it learns the *other* factorization of the same joint — the **posterior**
# $q_\phi(\theta \,|\, x) \approx p(\theta \,|\, x)$. No likelihood evaluation,
# no MCMC, no Bayes' theorem applied by hand: the theorem is enforced simply by
# where the training pairs come from.
#
# And because the model is amortized in $c = x$ (the rings), one training run
# gives you the posterior for *any* observation.
#
# **Toy problem.** A simulator with a curved degeneracy:
# $$\theta \sim U([-2,2]^2), \qquad
#   x = \big(\theta_1 + 1.0\,\varepsilon_1,\;\;
#            \theta_2 + \theta_1^2 + 0.1\,\varepsilon_2\big).$$
# The first data component barely constrains $\theta_1$; the second tightly
# constrains the *combination* $\theta_2+\theta_1^2$ — so the posterior is a
# long thin arc along the parabola $\theta_2 = x_{{\rm obs},2} - \theta_1^2$.
# It is a banana, and this time we did not put it there by hand: it *emerges*
# from the simulator. (Degenerate curved combinations of parameters: the
# everyday reality of GW posteriors.)

# %%
BANANA_NOISE = torch.tensor([1.0, 0.1])             # weak on x1, strong on x2


def banana_sim(theta):
    """The simulator: theta (n,2) -> data x (n,2)."""
    x = torch.stack([theta[:, 0],
                     theta[:, 1] + theta[:, 0] ** 2], 1)
    return x + BANANA_NOISE.to(theta.device) * torch.randn_like(x)


theta_train = torch.rand(20000, 2, device=dev) * 4 - 2    # theta_i ~ prior
x_train = banana_sim(theta_train)                         # x_i ~ p(x | theta_i)
x_obs = torch.tensor([[0.0, 0.7]], device=dev)            # our "observation"


def true_banana_logpost(grid, x_obs):
    """Analytic posterior on a grid (flat prior + Gaussian likelihood)."""
    mu = torch.stack([grid[:, 0], grid[:, 1] + grid[:, 0] ** 2], 1)
    return (-0.5 * ((mu - x_obs) ** 2 / BANANA_NOISE.to(grid.device) ** 2).sum(1))


g = torch.linspace(-2, 2, 300)
GX, GY = torch.meshgrid(g, g, indexing='ij')
grid = torch.stack([GX.ravel(), GY.ravel()], 1).to(dev)
logp_true = true_banana_logpost(grid, x_obs).reshape(300, 300).cpu()


def plot_truth(ax, logp=None):
    p = np.exp((logp_true if logp is None else logp)
               - (logp_true if logp is None else logp).max())
    ax.contour(GX, GY, p, levels=[0.011, 0.14, 0.61], colors='k',
               linewidths=1)                       # 3/2/1 sigma of a Gaussian
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r'$\theta_1$', ylabel=r'$\theta_2$')

# %% [markdown]
# Now train — and notice there is nothing new to write. `VelocityNet`,
# `fm_loss`, `train_fm` and `fm_sample` are the functions from Part 2,
# unchanged; the only difference is that `cond` is now data from a simulator.

# %%
fnet = VelocityNet(2, d_cond=2).to(dev)             # w = theta (2), cond = x (2)
train_fm(fnet, theta_train, x_train)

samp_fm = fm_sample(fnet, x_obs.expand(4000, 2), 2).cpu()

fig, ax = plt.subplots(figsize=(4.8, 4.4))
plot_truth(ax)
ax.plot(samp_fm[:, 0], samp_fm[:, 1], 'C0.', ms=1, alpha=.3)
ax.set_title('posterior from a simulator, no likelihood')
fig.tight_layout()

# %% [markdown]
# Black contours: the exact posterior, which for this toy we can compute
# analytically. Blue: samples from the trained network, given `x_obs`. We just
# did Bayesian inference with a regression loss.
#
# **Exercise 3.**
# 1. **Amortization, again.** The network learned $p(\theta|x)$ for *every*
#    $x$, not just ours. Sample the posterior for `x_obs2 = [[-1.0, 1.5]]` —
#    *without retraining* — and overlay the analytic truth (rebuild the grid
#    posterior with `true_banana_logpost(grid, x_obs2)` and pass it to
#    `plot_truth`). One simulator plus one training run buys you posteriors
#    for all observations.
# 2. **Sanity check.** Simulate a fresh $x$ from a *known* $\theta$, sample the
#    posterior, and check the truth lands inside it. Repeat a few times: how
#    often does the truth fall outside the 1-sigma contour? (This is the seed
#    of coverage testing, the standard way to validate an SBI posterior.)
# 3. **Break it.** Change `BANANA_NOISE` to `[1.0, 0.01]`, making the posterior
#    ten times thinner, and retrain. Does the network still track it? (Keep the
#    answer in mind — Part 4 is about exactly this failure.)

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
AMP = 0.5
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
def fit_pca(theta_bank, K=32):
    clean = chirp_sim(theta_bank, noise=0.0)
    mu = clean.mean(0)
    U, S, Vh = torch.linalg.svd(clean - mu, full_matrices=False)
    eigs = S / np.sqrt(len(clean) - 1)              # per-component SNR
    return mu, Vh[:K], eigs


def draw_prior(n):
    return PRIOR_LO + (PRIOR_HI - PRIOR_LO) * torch.rand(n, 2, device=dev)


mu0, V0, eigs0 = fit_pca(draw_prior(4096))
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.semilogy(eigs0.cpu()[:200])
ax.axhline(1, color='r', ls='--', lw=1, label='noise level')
ax.axvline(32, color='k', ls=':', lw=1, label='K = 32')
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


theta_bank = draw_prior(4096)
x_bank = chirp_sim(theta_bank)

s_bank = summarize(x_bank, mu0, V0)
s_mu, s_sd = s_bank.mean(0), s_bank.std(0) + 1e-6
th_mu, th_sd = theta_bank.mean(0), theta_bank.std(0)

cnet = VelocityNet(2, 32).to(dev)
train_fm(cnet, zscore(theta_bank, th_mu, th_sd),
         zscore(s_bank, s_mu, s_sd), steps=2000)

s_obs = zscore(summarize(x_obs_chirp, mu0, V0), s_mu, s_sd)
post0 = fm_sample(cnet, s_obs.expand(4000, 32), 2) * th_sd + th_mu

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


f0g = torch.linspace(54.6, 56.0, 160, device=dev)
fdg = torch.linspace(16.0, 19.5, 160, device=dev)
lp = chirp_true_logpost(f0g, fdg, x_obs_chirp).cpu()

fig, ax = plt.subplots(figsize=(5.5, 4.4))
ax.plot(post0.cpu()[:, 0], post0.cpu()[:, 1], 'C0.', ms=2, alpha=.3,
        label='amortized posterior')
p = np.exp(lp - lp.max())
ax.contour(f0g.cpu(), fdg.cpu(), p.T, levels=[0.011, 0.14, 0.61], colors='k',
           linewidths=1)
ax.plot(*THETA_TRUE.cpu(), 'r*', ms=14, label='truth')
ax.set(xlabel=r'$f_0$', ylabel=r'$\dot f$', xlim=(48, 64), ylim=(8, 28))
ax.legend(); ax.set_title('amortized: right place, far too blurry')
fig.tight_layout()

# %% [markdown]
# The network found the right region but is **much** wider than the true
# posterior (black contours — note we can compute them exactly here, a luxury
# the real problem doesn't offer). Why? Count the training samples that fall
# inside those contours:

# %%
inside = ((theta_bank[:, 0] > 54.6) & (theta_bank[:, 0] < 56.0)
          & (theta_bank[:, 1] > 16.0) & (theta_bank[:, 1] < 19.5))
print(f'training samples in the posterior neighbourhood: {inside.sum().item()} / {len(theta_bank)}')

# %% [markdown]
# **Sample starvation:** the posterior occupies a tiny fraction of the prior
# volume, so almost no training examples land where the answer lives. More
# capacity cannot fix having no data. The fix is to *move the training
# distribution*: simulate where the current posterior estimate points,
# retrain, repeat — each round ("rung") zooms further in. The training buffer
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
def sequential_chirp(n_rungs=8, gamma=0.5, n_keep=2048, refit_pca=True,
                     loss_fn=fm_loss, verbose=True):
    torch.manual_seed(1)
    buf_theta = draw_prior(4096)
    buf_x = chirp_sim(buf_theta)
    posts, spectra = [], []
    # warm-started nets: keep training the SAME networks across rungs (this is
    # what production codes do; retraining from scratch each rung underfits)
    qc, qm = VelocityNet(2, 32).to(dev), VelocityNet(2, 32).to(dev)
    opt_c = torch.optim.Adam(qc.parameters(), lr=2e-3)
    opt_m = torch.optim.Adam(qm.parameters(), lr=2e-3)
    for rung in range(1, n_rungs + 1):
        # -- gauges: PCA refit on the CURRENT buffer scale + z-scores
        if refit_pca or rung == 1:
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
        wp = torch.cat([fm_sample(qc, so.expand(n_prop // 2, 32), 2),
                        fm_sample(qm, torch.zeros(n_prop // 2, 32, device=dev), 2)])
        lqc = fm_logprob(qc, wp, so.expand(n_prop, 32))
        lqm = fm_logprob(qm, wp, torch.zeros(n_prop, 32, device=dev))
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
        post = fm_sample(qc, so.expand(4000, 32), 2) * tsd + tmu
        posts.append(post.cpu())
        if verbose:
            print(f'rung {rung}: buffer f0 std {buf_theta[:, 0].std():.3f}, '
                  f'posterior f0 std {post[:, 0].std():.3f}')
    return posts, spectra


posts, spectra = sequential_chirp()

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
colors = plt.cm.viridis(np.linspace(0, .9, len(posts)))
for r, (post, c) in enumerate(zip(posts, colors), 1):
    for a in ax[:2]:
        a.plot(post[:, 0], post[:, 1], '.', ms=1.5, alpha=.25, color=c,
               label=f'rung {r}' if a is ax[0] else None)
for a in ax[:2]:
    a.contour(f0g.cpu(), fdg.cpu(), p.T, levels=[0.011, 0.61], colors='k',
              linewidths=1)
    a.plot(*THETA_TRUE.cpu(), 'r*', ms=14)
    a.set(xlabel=r'$f_0$', ylabel=r'$\dot f$')
ax[0].set(xlim=(40, 80), ylim=(0, 40), title='the zoom trajectory')
ax[1].set(xlim=(54.6, 56.0), ylim=(16.0, 19.5),
          title='late rungs vs exact posterior (black)')
ax[0].legend(markerscale=8, fontsize=8)
for r, (e, c) in enumerate(zip(spectra, colors), 1):
    ax[2].semilogy(e[:200], color=c, label=f'rung {r}')
ax[2].axhline(1, color='r', ls='--', lw=1)
ax[2].set(xlabel='PCA component', ylabel='component SNR',
          title='compression gets easier as the prior shrinks')
ax[2].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# Two things happened at once:
# 1. **The posterior tightened** toward the true (black) contours, rung by
#    rung — same network size, same per-rung simulation budget; only the
#    *training distribution* moved.
# 2. **Compression became easy**: at the zoomed prior a handful of PCA
#    components carry all the signal (right panel) — the flat wide-prior
#    spectrum steepened dramatically. This is why adaptive summaries
#    (refitting the basis as you zoom) matter for the real problem.
#
# **Exercise 4.**
# 1. Run with `gamma=0.1` and `gamma=1.0`. Which converges faster? Which is
#    riskier? (Think: what happens if an early, imperfect posterior estimate
#    excludes the truth — can a later rung recover?)
# 2. Run with `refit_pca=False` (freeze the rung-1 basis). How much slower is
#    the zoom? Look at the right panel to see why.
# 3. *(bonus, from Exercise 2b.4)* Define `fm_loss_late` with
#    `t = torch.rand(...) ** (1/8)` for half the batch and pass it as
#    `loss_fn=`. At LISA scale this one-line change moved our posteriors from
#    "16 nats too wide" to "1 nat from mathematically optimal".

# %% [markdown]
# ---
# # Part 5 — The real thing: a massive black-hole binary in LISA data  *(~20 min)*
#
# Same code, real problem: **LDC1-1 (Radler)** — one day of simulated LISA
# data containing a merging massive black-hole binary at SNR ~260. What
# changes compared to the toy:
#
# | | toy chirp | MBHB |
# |---|---|---|
# | data | 1024 samples | 2 TDI channels × 8640 samples, whitened |
# | parameters | 2 (+1 nuisance) | 9 (+2 gauge nuisances) |
# | simulator | 1 line | IMRPhenomD + LISA response (seconds/waveform) |
# | summaries | 32 PCA | 64 PCA |
#
# Because the waveform model needs a compiled stack, the training bank is
# **pre-simulated** (32768 sims from a narrowed prior; the script
# `make_tutorial_simbank.py` in this folder regenerates it). Everything you
# *run* here — compression, `fm_loss`, training, sampling — is identical to
# what you already built.

# %%
# Get mbhb_simbank.npz (~12 MB). Three options, tried in order:
#   1. it is already next to this notebook (cluster / manual upload)
#   2. download from Google Drive: paste the file id you were given
#   3. colab upload widget
import os

BANK = 'mbhb_simbank.npz'
GDRIVE_FILE_ID = ''          # <-- tutor: paste the Drive file id here

if not os.path.exists(BANK):
    for cand in ('mbhb_simbank.npz',):
        if os.path.exists(cand):
            BANK = cand
            break
if not os.path.exists(BANK) and GDRIVE_FILE_ID:
    import gdown                                   # pre-installed on colab
    gdown.download(id=GDRIVE_FILE_ID, output=BANK, quiet=False)
if not os.path.exists(BANK):
    try:
        from google.colab import files             # last resort: manual upload
        print('please upload mbhb_simbank.npz')
        files.upload()
    except ImportError:
        raise FileNotFoundError('mbhb_simbank.npz not found — generate it with '
                                'make_tutorial_simbank.py or set GDRIVE_FILE_ID')

bank = np.load(BANK, allow_pickle=False)
names = [str(n) for n in bank['names']]
print('parameters:', names)
print('training bank:', bank['z_train'].shape, '->', bank['s_train'].shape)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].plot(bank['x_obs'][0], lw=.4, label='observed (A channel, whitened)')
ax[0].plot(bank['x_clean_true'][0], 'C1', lw=.8, label='hidden signal')
ax[0].legend(loc='upper left', fontsize=8)
ax[0].set(xlabel='sample (dt = 10 s)', title='one day of LISA data')
ax[1].semilogy(bank['pca_eigs'])
ax[1].axhline(1, color='r', ls='--', lw=1); ax[1].axvline(64, color='k', ls=':', lw=1)
ax[1].set(xlabel='PCA component', ylabel='component SNR', title='summary spectrum (64 kept)')
fig.tight_layout()

# %%
z = torch.tensor(bank['z_train'], device=dev)
s = torch.tensor(bank['s_train'], device=dev)
zmu, zsd = z.mean(0), z.std(0)
smu, ssd = s.mean(0), s.std(0) + 1e-6

mnet = VelocityNet(9, 64, hidden=256, layers=4).to(dev)
train_fm(mnet, zscore(z, zmu, zsd), zscore(s, smu, ssd), steps=4000, batch=512)

s_obs_t = zscore(torch.tensor(bank['s_obs'], device=dev)[None], smu, ssd)
post = (fm_sample(mnet, s_obs_t.expand(20000, 64), 9) * zsd + zmu).cpu().numpy()

# %%
zt = bank['z_true']
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot(bank['mcmc_dl_cosi'][:, 0], bank['mcmc_dl_cosi'][:, 1], '.', ms=1.5,
        alpha=.15, color='gray', label='reference MCMC (full year of data)')
ax.plot(post[:, 0], post[:, 1], 'C0.', ms=1.5, alpha=.15, label='your posterior (1 day)')
ax.plot(zt[0], zt[1], 'r*', ms=15, label='truth')
ax.set(xlabel=r'$\log_{10} D_L$ [Mpc]', ylabel=r'$\cos\iota$',
       xlim=(4.3, 5.4), ylim=(-1, 1))
ax.legend(loc='lower left', fontsize=8)
ax.set_title('distance–inclination degeneracy of a real MBHB')
fig.tight_layout()

# %% [markdown]
# You just inferred the parameters of a massive black-hole binary — with the
# ten-line loss from Part 2. The X-shaped structure is the famous
# **distance–inclination degeneracy**: a face-on binary far away looks like an
# edge-on binary nearby. Note your posterior is honestly *wider* than the gray
# reference — that analysis used a full year of data, you used one day.
#
# **Exercise 5.**
# 1. Plot other 2-D marginals (e.g. chirp mass vs symmetric mass ratio,
#    columns 2 & 3 — masses to five digits from one day of data!). Add the
#    truth marker. Which parameters are tight, which are degenerate?
# 2. The truth star sits slightly off your posterior median in $D_L$. Is that
#    a bug? (Hint: there is exactly one noise realization in `x_obs`. What
#    would you expect the *distribution* of median-vs-truth offsets to look
#    like over many noise realizations?)
# 3. *(discussion)* This was amortized inference at a narrowed prior. What
#    would you need to add to run it from the *full* prior? (Everything from
#    Part 4: sequential zoom + adaptive summaries. That is what the `falcon`
#    package automates — and with the late-time trick from Exercise 4.3 it
#    reaches posteriors ~1 nat from the information-theoretic optimum.)
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
