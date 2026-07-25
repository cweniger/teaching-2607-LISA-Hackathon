# %% [markdown]
# # From MLPs to LISA: simulation-based inference, hands-on
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_lisa_sbi.ipynb)
#
# **LISA tutorial — approximately 90 minutes.**
#
# | part | idea | new ingredient |
# |---|---|---|
# | 1 | fit a function with a neural network | MLPs, overfitting |
# | 2 | fit a *distribution*: SBI on a banana posterior | flow matching (FM) |
# | 3 | a toy gravitational wave | data compression + **sequential** inference |
# | 4 | the real thing: a massive black-hole binary in LISA data | (pre-simulated) |
#
# Each part fixes the visible failure of the one before. All code is plain
# PyTorch — the same ~10-line loss function you meet in Part 2 analyzes the
# LISA signal in Part 4.
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
# The MLP below maps one input number to one output number through three
# hidden layers. Each layer is an affine map $h \mapsto Wh + b$ (`nn.Linear`)
# followed by an elementwise nonlinearity — by default
# $\mathrm{ReLU}(z) = \max(z, 0)$, which builds piecewise-linear fits with
# visible kinks; smooth activations like `torch.tanh`, `nn.functional.gelu`
# or `torch.selu` build smoother fits (try them!). The entries of the weight
# matrices $W$ and bias vectors $b$ are the trainable parameters. The
# `forward` method spells out, step by step, what happens to a batch of
# inputs.

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
# $$\mathcal L(W, b) = \frac{1}{N} \sum_{i=1}^{N}
#   \big(\mathrm{MLP}_{W,b}(x_i) - y_i\big)^2 ,$$
# and minimize it over *all* weights and biases at once by **gradient
# descent**:
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
# While training runs we monitor, every epoch:
#
# - the **training loss** $\mathcal L$ itself;
# - the **validation loss**: the same MSE evaluated on the held-out data set.
#   The network never trains on those points, so this measures how well it
#   *generalizes* rather than memorizes;
# - a **noise estimate** $\hat\sigma = \sqrt{\mathcal L}$, the RMS residual
#   on the training points. If the network had learned $f$ exactly, the
#   residuals would be pure noise and $\hat\sigma = \sigma = 0.2$. Watch
#   what happens instead: $\hat\sigma$ keeps dropping *below* $\sigma$ —
#   the network increasingly explains the noise as if it were signal.
#   That is overfitting, condensed into a single number.

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
            sigma_hat = loss.sqrt()        # RMS train residual = noise estimate
            hist.append((loss.item(), val.item(), sigma_hat.item()))
            # keep a copy of the weights whenever the validation loss improves —
            # this snapshot is the network early stopping would return
            if val < best_val:
                best_val, snap_val = val.item(), copy.deepcopy(net.state_dict())
        if (ep + 1) % 1000 == 0:
            print(f'epoch {ep + 1:5d}:  train {loss:.4f}   val {val:.4f}   '
                  f'sigma_hat {sigma_hat:.3f}  (true sigma {SIGMA})')
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
ax[1].semilogy(hist[:, 2], 'C4', label=r'$\hat\sigma$ (RMS train residual)')
ax[1].axhline(SIGMA, color='r', ls='--', lw=1, label=fr'true $\sigma = {SIGMA}$')
ax[1].set(xlabel='epoch', ylabel=r'MSE  /  $\hat\sigma$')
ax[1].legend(fontsize=8, ncol=2)
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
#   held-out points. The noise estimate $\hat\sigma$ first drops toward the
#   true $\sigma$ (the network learns $f$; the residuals become pure noise),
#   then keeps sinking below it — the network is absorbing noise into the fit.
#   The moment $\hat\sigma$ crosses $\sigma$ is the moment memorization begins.
# - *Order matters:* the noise-chasing wiggles are high-frequency, and they
#   appear only late — MLPs fit smooth structure first (**spectral bias**).
#   That is why early stopping works: it keeps the signal, drops the noise.
#
# **Exercise 1a — read the plot.**
# 1. Which of the two fits in the left panel would you trust to predict $y$ at
#    a new $x$ — and how could you make that choice in a *real* experiment,
#    where the truth (black dashed) is not available?
# 2. Connect the panels: where on the loss curves do the two fits live? How
#    does the moment $\hat\sigma$ crosses the true $\sigma$ relate to the
#    minimum of the validation curve?
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
#
# Then use it:
# 1. `EPOCHS` is no longer a knob you must tune — pass something huge and let
#    patience decide. Scan the *architecture* instead: `WIDTH`
#    $\in \{2, 16, 256, 1024\}$ (you can also add or remove hidden layers in
#    the `MLP` class). How does the best validation loss depend on capacity —
#    is the biggest network the worst one?
# 2. Scan the *data*: `N_TRAIN` $\in \{10, 30, 100, 500\}$ (re-run the data
#    cell). How quickly does the best validation loss approach the noise
#    floor $\sigma^2$?

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
# ---
# # Part 2 — Fitting distributions: SBI and flow matching  *(~25 min)*
#
# In GW astronomy we don't want one best-fit — we want the **posterior**
# $p(\theta\,|\,x)$. Simulation-based inference (SBI) learns it from pairs
# $(\theta_i, x_i)$: draw $\theta_i$ from the prior, simulate $x_i$, train a
# network that turns $x$ into a distribution over $\theta$.
#
# Toy problem with a *curved* (banana) posterior:
# $$\theta \sim U([-2,2]^2), \qquad
#   x = \big(\theta_1 + 1.0\,\varepsilon_1,\;\;
#            \theta_2 + \theta_1^2 + 0.1\,\varepsilon_2\big).$$
# The first data component barely constrains $\theta_1$; the second tightly
# constrains the *combination* $\theta_2+\theta_1^2$ — so the posterior is a
# long thin arc along the parabola $\theta_2 = x_{{\rm obs},2} - \theta_1^2$.
# (Degenerate curved combinations of parameters: the everyday reality of GW
# posteriors.)
#
# **First attempt:** the simplest "distribution head" — predict a Gaussian
# (mean + covariance) from $x$.

# %%
BANANA_NOISE = torch.tensor([1.0, 0.1])             # weak on x1, strong on x2


def banana_sim(theta):
    x = torch.stack([theta[:, 0],
                     theta[:, 1] + theta[:, 0] ** 2], 1)
    return x + BANANA_NOISE.to(theta.device) * torch.randn_like(x)


theta_train = torch.rand(20000, 2, device=dev) * 4 - 2
x_train = banana_sim(theta_train)
x_obs = torch.tensor([[0.0, 0.7]], device=dev)      # our "observation"


def true_banana_logpost(grid, x_obs):
    """Analytic posterior on a grid (flat prior + Gaussian likelihood)."""
    mu = torch.stack([grid[:, 0], grid[:, 1] + grid[:, 0] ** 2], 1)
    return (-0.5 * ((mu - x_obs) ** 2 / BANANA_NOISE.to(grid.device) ** 2).sum(1))


g = torch.linspace(-2, 2, 300)
GX, GY = torch.meshgrid(g, g, indexing='ij')
grid = torch.stack([GX.ravel(), GY.ravel()], 1).to(dev)
logp_true = true_banana_logpost(grid, x_obs).reshape(300, 300).cpu()


def plot_truth(ax):
    p = np.exp(logp_true - logp_true.max())
    ax.contour(GX, GY, p, levels=[0.011, 0.14, 0.61], colors='k',
               linewidths=1)                       # 3/2/1 sigma of a Gaussian
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r'$\theta_1$', ylabel=r'$\theta_2$')

# %%
# Generic MLP helper for the rest of the notebook: same idea as Part 1's MLP
# class, but with configurable input/output dimensions and depth.
def mlp(d_in, d_out, hidden, layers):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, d_out))


# Gaussian posterior head: x -> (mu, Cholesky of covariance), trained by
# maximizing the Gaussian log-likelihood of the true theta.
gnet = mlp(2, 5, 128, 3).to(dev)                    # 2 mean + 3 Cholesky numbers


def gaussian_nll(out, theta):
    mu, d1, d2, off = out[:, :2], out[:, 2], out[:, 3], out[:, 4]
    L11, L22 = d1.exp(), d2.exp()                   # positive diagonal
    r = theta - mu
    z1 = r[:, 0] / L11
    z2 = (r[:, 1] - off * z1) / L22
    return (0.5 * (z1 ** 2 + z2 ** 2) + d1 + d2).mean()


opt = torch.optim.Adam(gnet.parameters(), lr=1e-3)
for step in range(2000):
    i = torch.randint(0, len(theta_train), (512,), device=dev)
    loss = gaussian_nll(gnet(x_train[i]), theta_train[i])
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    out = gnet(x_obs)[0].cpu()
    mu, L = out[:2], torch.tensor([[out[2].exp(), 0], [out[4], out[3].exp()]])
    samp_g = mu + torch.randn(4000, 2) @ L.T

fig, ax = plt.subplots(figsize=(4.6, 4.2))
plot_truth(ax)
ax.plot(samp_g[:, 0], samp_g[:, 1], 'C3.', ms=1, alpha=.3)
ax.set_title('Gaussian head: cannot bend')
fig.tight_layout()

# %% [markdown]
# The ellipse straddles the banana — a Gaussian **cannot represent curved or
# multi-modal posteriors**, and real GW posteriors are full of both.
#
# ## Flow matching: distributions as flows
#
# Idea: start from noise $w_0\sim\mathcal N(0,1)$ and *transport* it to
# posterior samples $w_1\sim p(\theta|x)$ by integrating a learned velocity
# field $v(w, t\,|\,x)$ from $t=0$ to $1$. Training needs no integration at
# all: draw a training sample $w_1$, a noise point $w_0$, a random time $t$,
# put yourself on the straight line between them, and regress the velocity
# onto the direction $w_1 - w_0$:
#
# $$\mathcal L = \mathbb E\,\big\|\,v\big((1{-}t)w_0 + t\,w_1,\;t\,\big|\,x\big)
#   - (w_1 - w_0)\big\|^2 .$$
#
# That is the whole method — here it is in ten lines.

# %%
def fm_loss(net, w1, cond):
    """The flow-matching objective. This exact function is reused in Parts 3 & 4."""
    w0 = torch.randn_like(w1)
    t = torch.rand(len(w1), 1, device=w1.device)
    wt = (1 - t) * w0 + t * w1
    v = net(wt, t, cond)
    return ((v - (w1 - w0)) ** 2).mean()


class VelocityNet(nn.Module):
    """MLP velocity field v(w, t | cond) with a small Fourier time embedding."""

    def __init__(self, d_w, d_cond, hidden=128, layers=3):
        super().__init__()
        self.freqs = torch.tensor([1., 2., 4., 8.])
        self.net = mlp(d_w + 9 + d_cond, d_w, hidden, layers)

    def forward(self, w, t, cond):
        ft = 2 * np.pi * t * self.freqs.to(t.device)
        temb = torch.cat([t, ft.sin(), ft.cos()], 1)
        return self.net(torch.cat([w, temb, cond], 1))


@torch.no_grad()
def fm_sample(net, cond, d_w, steps=64):
    """Integrate dw/dt = v from t=0 (noise) to t=1 (posterior samples)."""
    w = torch.randn(len(cond), d_w, device=cond.device)
    for i in range(steps):
        t = torch.full((len(cond), 1), (i + 0.5) / steps, device=cond.device)
        w = w + net(w, t, cond) / steps
    return w

# %%
def train_fm(net, w1, cond, steps=3000, batch=512, lr=1e-3, log=True):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(w1), (batch,), device=w1.device)
        loss = fm_loss(net, w1[i], cond[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if log and (step + 1) % 1000 == 0:
            print(f'  step {step + 1}/{steps}  loss {loss.item():.3f}  '
                  f'[{time.time() - t0:.0f}s]')
    return net


fnet = VelocityNet(2, 2).to(dev)
train_fm(fnet, theta_train, x_train)
samp_fm = fm_sample(fnet, x_obs.expand(4000, 2), 2).cpu()

fig, ax = plt.subplots(figsize=(4.6, 4.2))
plot_truth(ax)
ax.plot(samp_fm[:, 0], samp_fm[:, 1], 'C0.', ms=1, alpha=.3)
ax.set_title('flow matching: bends just fine')
fig.tight_layout()

# %% [markdown]
# **Exercise 2.**
# 1. **Amortization.** The network learned $p(\theta|x)$ for *every* $x$, not
#    just ours. Sample the posterior for `x_obs = [[-1.0, 1.5]]` — *without
#    retraining* — and overlay the analytic truth (rebuild `logp_true` for the
#    new observation). One simulator + one training = posteriors for all
#    observations.
# 2. **The ODE knob.** Redo the sampling with `steps=1, 4, 16`. How many steps
#    do you need before the banana stops being distorted?
# 3. *(bonus)* In `fm_loss`, replace `t = torch.rand(...)` by
#    `t = torch.rand(...) ** 0.5` (more training near $t{=}1$, where the flow
#    must form the sharp shape). Does the banana get cleaner at few steps?
#    Remember this trick — it returns at the end of Part 3.

# %% [markdown]
# ---
# # Part 3 — A toy gravitational wave: compression + sequential zoom  *(~30 min)*
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
# **Exercise 3.**
# 1. Run with `gamma=0.1` and `gamma=1.0`. Which converges faster? Which is
#    riskier? (Think: what happens if an early, imperfect posterior estimate
#    excludes the truth — can a later rung recover?)
# 2. Run with `refit_pca=False` (freeze the rung-1 basis). How much slower is
#    the zoom? Look at the right panel to see why.
# 3. *(bonus, from Exercise 2.3)* Define `fm_loss_late` with
#    `t = torch.rand(...) ** (1/8)` for half the batch and pass it as
#    `loss_fn=`. At LISA scale this one-line change moved our posteriors from
#    "16 nats too wide" to "1 nat from mathematically optimal".

# %% [markdown]
# ---
# # Part 4 — The real thing: a massive black-hole binary in LISA data  *(~20 min)*
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
# **Exercise 4.**
# 1. Plot other 2-D marginals (e.g. chirp mass vs symmetric mass ratio,
#    columns 2 & 3 — masses to five digits from one day of data!). Add the
#    truth marker. Which parameters are tight, which are degenerate?
# 2. The truth star sits slightly off your posterior median in $D_L$. Is that
#    a bug? (Hint: there is exactly one noise realization in `x_obs`. What
#    would you expect the *distribution* of median-vs-truth offsets to look
#    like over many noise realizations?)
# 3. *(discussion)* This was amortized inference at a narrowed prior. What
#    would you need to add to run it from the *full* prior? (Everything from
#    Part 3: sequential zoom + adaptive summaries. That is what the `falcon`
#    package automates — and with the late-time trick from Exercise 3.3 it
#    reaches posteriors ~1 nat from the information-theoretic optimum.)
#
# ---
# ## Where to go from here
#
# - **Dynamic (sequential) SBI:** Alvey, Lyu, Weniger et al., arXiv:2510.13997
#   — the tempered-buffer mechanism you used in Part 3, at production scale.
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
