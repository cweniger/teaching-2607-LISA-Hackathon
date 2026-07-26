# From MLPs to LISA — hands-on SBI tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2607-LISA-Hackathon/blob/main/tutorial_lisa_sbi.ipynb)

**[Intro slides](https://cweniger.github.io/teaching-2607-LISA-Hackathon/)** —
the 30-minute concept lecture given before the tutorial: the science case,
Bayesian inference, neural networks, SBI, flow matching (with a live demo of
the flow), and sequential SBI.

A 90-minute hands-on tutorial for the LISA Hackathon (July 2026): from
fitting a sine curve with a neural network to inferring the parameters of a
massive black-hole binary in simulated LISA data — with the same ten-line
flow-matching loss all the way through.

| part | idea | new ingredient |
|---|---|---|
| 1 | fit a function with a neural network | MLPs, overfitting |
| 2 | fit a *distribution*: SBI on a banana posterior | flow matching |
| 3 | a toy gravitational wave | data compression + sequential inference |
| 4 | a massive black-hole binary in LISA data (LDC1 Radler) | (pre-simulated) |

## Run it

Click the Colab badge, select a **T4 GPU runtime** (Runtime → Change runtime
type), and run all cells. Dependencies: torch + numpy + matplotlib only (all
pre-installed on Colab). Full execution takes ~3–5 minutes on a T4; the
exercises are knob-turning experiments on top.

Part 4 needs the pre-simulated training bank `mbhb_simbank.npz` (~9 MB):
the notebook downloads it from Google Drive (file id set in the download
cell) or accepts a manual upload.

## Files

| file | role |
|---|---|
| `tutorial_lisa_sbi.ipynb` | the tutorial (open this) |
| `lisa_sims.ipynb` | companion: pip-install lisabeta, simulate LISA data live |
| `lisa_sequential.ipynb` | companion: the sequential zoom on the MBHB, live sims |
| `*_src.py` | notebook sources (jupytext percent format) |
| `build_notebook.py` | `*_src.py` → `.ipynb` converter |
| `make_tutorial_simbank.py` | regenerates `mbhb_simbank.npz` |
| `DESIGN_NOTES.md` | why the tutorial is built this way |
| `docs/` | the intro slide deck, served by GitHub Pages (reveal.js) |

The two companion notebooks need no pre-simulated data: they install the
lisabeta waveform stack from PyPI wheels (~20 s on Colab) and simulate
everything live — `lisa_sequential.ipynb` runs the actual dynamic-SBI loop
(4 rungs, ~2000 live simulations per rung, ~3 min on CPU) on the
9-parameter MBHB problem.

`make_tutorial_simbank.py` requires the LISA waveform stack (lisabeta +
lisa-data-challenge) and the analysis repository
[dsbi-ldc-mbhb](https://github.com/lvhf123/dsbi-ldc-mbhb) — see
`standalone_tests/INSTALL.md` there. The notebook itself needs none of that.

## Background

The production-scale numbers quoted in Part 4 (sequential SBI reaching
posteriors ~1 nat from the information-theoretic optimum on the LDC1-1 MBHB
problem) come from the dynamic-SBI campaign in
[dsbi-ldc-mbhb](https://github.com/lvhf123/dsbi-ldc-mbhb)
(`standalone_tests/RESULTS_260721_E1_recipe.md`), built on
[falcon](https://github.com/cweniger/falcon) and the Dynamic SBI method
([arXiv:2510.13997](https://arxiv.org/abs/2510.13997)).
