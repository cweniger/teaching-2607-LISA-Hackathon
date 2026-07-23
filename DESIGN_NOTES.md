# Tutorial design notes — why the notebook is built the way it is

Distilled from the design discussion (2026-07-22) that produced
`tutorial_lisa_sbi.ipynb`. Audience: 90-min hands-on for ML newbies (PhD
level) at a LISA tutorial, 30-min concept lecture beforehand.

## Guiding principles

1. **One new idea per section, each fixing the previous section's visible
   failure.** MLPs fit functions but overfit → SBI turns simulation into
   inference but Gaussian heads can't shape → FM shapes anything but starves
   on wide priors → sequential zoom feeds it → same stack analyzes real LISA
   data. This is the spine that keeps "a ridiculous amount of information"
   coherent.
2. **Same visible code end-to-end.** The 10-line `fm_loss` and the small
   `VelocityNet` defined in Part 2 are literally reused in Parts 3 and 4.
   The take-home message: this is not magic infrastructure, it is a loss
   function and a loop you fully understand, and it analyzes a gravitational
   wave.
3. **Zero installs, zero framework.** torch + numpy + matplotlib only (all
   colab-native). We deliberately do NOT use falcon in the notebook: at
   tutorial scale a framework only subtracts — black-box calls, install risk,
   abstraction overhead. falcon gets its billing at the end as "what this
   loop grows into at production scale" (promotion by provenance: the E1
   results shown were made with this method stack). The whole dynamic loop
   at teaching size is ~40 lines; the week's standalone campaign proved the
   ~100-line version beats the production pipeline.
4. **Live where cheap, precomputed where expensive, results-only where
   risky.** Toy simulators run live (microseconds/sim). The MBHB training
   bank is precomputed (`make_tutorial_simbank.py`) because lisabeta/LDC is
   a compiled stack that will not install in colab during a tutorial.
   Sequential inference on the MBHB is *slides only* (E1 results) — staging
   nested sim banks to fake a live zoom would hard-wire the zoom schedule
   and quietly betray the adaptivity that makes sequential SBI interesting.

## Section-specific decisions

### Part 2 (banana)
- **Anisotropic noise (1.0, 0.1) is essential.** With isotropic noise the
  posterior arc is short and blob-like and the Gaussian head does NOT
  visibly fail (tested — figs were nearly identical). Weak constraint on
  θ₁ + tight constraint on the combination θ₂+θ₁² gives a long thin curved
  arc: the Gaussian head produces a hopeless diffuse cloud, FM traces the
  arc. Also the honest GW analogy: degenerate curved combinations of
  parameters.
- Analytic grid posterior as ground truth (a luxury the real problem lacks —
  say so out loud).

### Part 3 (toy chirp)
- The chirp toy exists to motivate the two ingredients the banana cannot:
  high-dim data → **PCA compression** (eigen-spectrum, "how many components
  above noise") and razor likelihood from wide prior → **sequential zoom**.
- **SNR ≈ 11 (A=0.5), not 27.** At SNR 27 the exact posterior is so thin it
  is invisible at plot scale and the zoom can't reach it in 8 rungs. At 11
  everything is visible and the zoom lands inside the exact contours.
- **Phase φ as sampled nuisance** mirrors the real analysis' gauge angles
  (φ₀, ψ); the exact-posterior grid marginalizes it via the
  linear-in-(cosφ,sinφ) trick — the same trick used at production scale.
- **Warm-start the nets across rungs.** Retraining from scratch each rung
  (800 steps) underfits and the final posterior lands ~2σ off the truth,
  reverting toward the buffer mean (tested; the fix moved posterior σ_f0
  from 0.20 to 0.10 ≈ exact width). Also pedagogically truer: production
  codes keep training the same nets as the buffer evolves.
- **PCA refit per rung** doubles as the DynamicSVD argument: the spectrum
  panel shows compression getting easy as the prior shrinks (ties to the
  full-prior census result: frozen 64 PCs capture ~2% of a typical
  full-prior MBHB signal).
- The "sample starvation" cell prints the number of prior-drawn training
  samples inside the posterior neighbourhood (10/4096) BEFORE introducing
  the zoom — the failure is quantified before the fix is shown.
- The `fm_logprob` cell (reverse ODE + divergence for IS weights) is the one
  deliberately "advanced" cell; it is kept because it IS the dynamic-SBI
  mechanism (labelled as falcon's proposal machinery in miniature).

### Part 4 (MBHB)
- **9-dim latent, 2-D plots.** Do NOT reduce to 2 free parameters: the
  production campaign measured that freeing only (D_L, cos ι) washes out the
  caustic (trough 0.55–0.89 vs 0.16 with 9 free). The other 7 parameters are
  invisible to students (a tensor shape) and cost nothing.
- Bank: 32k noisy sims at the MCMC-narrowed prior, 64 PCA coefficients,
  ~9 MB npz. Loading order in the notebook: local file → gdown (tutor pastes
  Drive file id) → colab upload widget.
- BakerMarsat (D_L, cos ι) subsample ships inside the npz for the overlay;
  honest framing in the text: reference used a full year of data, students
  used one day, so wider is correct.
- The φ-marginalization, the caustic, and the "your posterior is honestly
  wider" comparison each echo something from Parts 2–3 — closing the loop.

## Timing budget

Measured 66 s end-to-end on A100 → estimate 3–5 min on colab T4 (small nets;
much of the cost is Python-loop ODE steps, so the GPU gap is modest).
Section targets: P1 ~15 min, P2 ~25, P3 ~30, P4 ~20 + buffer. Newbie
hands-on runs 2–3× solo estimates; every section works with "run all" and
knob-turning exercises rather than blocking fill-in-the-blank cells.

## Known deployment caveats (for the tutor)

- The colab badge pins branch `fix/cw`; update URL if merged/moved.
- Colab-from-GitHub needs the repo public (or per-user GitHub auth in
  colab); fallback = distribute the .ipynb file directly.
- `GDRIVE_FILE_ID` in the Part 4 download cell must be filled with the
  Drive id of `mbhb_simbank.npz` (link-shared).
- Do one real colab dry-run: everything is tested end-to-end on the
  cluster (3 executed passes, all figures inspected), but not in colab's
  own environment.

## Provenance

The quoted production numbers (E1: mean logL 1 nat from `ML − D/2`, the
late-heavy t-sampling lever, the 64-PC full-prior census) come from the
2026-07 standalone campaign — see
`../standalone_tests/RESULTS_260721_E1_recipe.md`.
