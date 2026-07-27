# From MLPs to LISA — hands-on SBI tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_lisa_sbi.ipynb) &nbsp;**the tutorial**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/lisa_mbhb_first_steps.ipynb) &nbsp;**the sequential zoom on the MBHB**, with waveforms simulated live

**[Intro slides](https://cweniger.github.io/teaching-2607-LISA-Hackathon/)** —
the 30-minute concept lecture given before the tutorial: the science case,
Bayesian inference, neural networks, SBI, flow matching (with a live demo of
the flow), and sequential SBI.

A ~95-minute hands-on tutorial for the LISA Hackathon (July 2026): from
fitting a sine curve with a neural network to running a sequential
simulation-based-inference loop on a gravitational-wave signal — with the
same ten-line flow-matching loss all the way through. The real massive
black-hole binary lives in the companion notebook.

| part | idea | new ingredient |
|---|---|---|
| 1 | fit a *function* with a neural network | MLPs, overfitting, early stopping |
| 2 | fit a *distribution* | flow matching, conditioning |
| 3 | fit a *posterior*: feed the flow pairs from a simulator | SBI, amortization |
| 4 | a toy gravitational wave | data compression + **sequential** inference |

## Run it

Click the Colab badge, select a **T4 GPU runtime** (Runtime → Change runtime
type), and run all cells. Dependencies: torch + numpy + matplotlib only (all
pre-installed on Colab). Full execution takes ~3–5 minutes on a T4; the
exercises are knob-turning experiments on top.

The tutorial needs no external data at all — every simulator in it is a few
lines of torch. For the real LISA problem, continue with
`lisa_mbhb_first_steps.ipynb`, which installs the lisabeta waveform stack from
PyPI (~20 s on Colab) and simulates live.

## Files

| file | role |
|---|---|
| `tutorial_lisa_sbi.ipynb` | the tutorial (open this) |
| `lisa_mbhb_first_steps.ipynb` | **companion:** the sequential zoom on the MBHB, live sims — bare bones (latent-space flows, prior in the proposal mixture, importance-weighted readout, no Procrustes/PSIS/EMA) |
| `*_src.py` | notebook sources (jupytext percent format) |
| `build_notebook.py` | `*_src.py` → `.ipynb` converter |
| `production_posterior.npy` | reference posterior the companion notebook overlays on its own |
| `docs/` | the intro slide deck, served by GitHub Pages (reveal.js) |

The companion notebook needs no pre-simulated data either: it installs the
lisabeta waveform stack from PyPI wheels (~20 s on Colab) and simulates
everything live, running the actual dynamic-SBI loop (4 rounds, ~2000 live
simulations per round) on the 9-parameter MBHB problem.

## Background

The production-scale numbers quoted in Part 4 (sequential SBI reaching
posteriors ~1 nat from the information-theoretic optimum on the LDC1-1 MBHB
problem) come from the dynamic-SBI campaign in
[dsbi-ldc-mbhb](https://github.com/lvhf123/dsbi-ldc-mbhb)
(`standalone_tests/RESULTS_260721_E1_recipe.md`), built on
[falcon](https://github.com/cweniger/falcon) and the Dynamic SBI method
([arXiv:2510.13997](https://arxiv.org/abs/2510.13997)).
