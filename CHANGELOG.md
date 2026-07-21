# Changelog

All notable changes to the BAN-501 online course companion are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [1.0.0] — 2026-07-21

First online edition. The BAN-501 course companion moved from a set of Markdown files to a
searchable MkDocs Material site, and every numerical example in it was re-run and checked
against code before publication.

### Added

- **The companion site** — 10 modules, 8 deep-dive appendices, and a Q&A reference (19
  pages, 23 figures) rendered with MkDocs Material: MathJax math, syntax-highlighted code,
  full-text search, light/dark themes, and per-module navigation.
- **Computation-verification suite** (`computations/`) — the 17 example files (98 `demo_*`
  functions) that produce the site's numbers, `generate_figures.py` for the 23 figures, and
  `verify_all.py`, which re-runs every example and compares its output to the numbers shown
  on the site. Run with `pixi run -e compute verify`.
- **Two-environment pixi setup** — a lightweight `docs` environment (build/serve the site,
  all platforms) and a GPU `compute` environment (CUDA 13, linux-64) for running the
  examples and regenerating figures.
- **`CODE_EXAMPLE_MAPPING.md`** — an index mapping all 98 example functions to their site
  page and admonition.
- **Continuous deployment** — a GitHub Actions workflow builds the site with
  `mkdocs build --strict` and publishes it to GitHub Pages on every push to `main`.

### Changed

- **Callouts became admonitions.** Every blockquote callout was upgraded to a MkDocs
  Material admonition: the 98 worked examples are now `!!! example` blocks, deep-dive status
  notes `!!! note`, the universal-approximation theorem `!!! quote`, and so on.
- **Illustrative code modernized.** Residual pandas accessors in the Modules 1/2/9 teaching
  snippets (`.iloc`, `.values`) were converted to polars/NumPy-safe equivalents, keeping the
  course polars-first (the statsmodels formula example stays pandas, as its API requires).
- **Demos rebuilt to teach the concept correctly.** Several examples were redesigned so the
  code actually demonstrates what the surrounding text claims, including: the fairness
  impossibility result (now the Chouldechova/Kleinberg theorem, that equalized odds and
  predictive parity cannot both hold under unequal base rates); permutation importance (now
  uses a genuinely correlated redundant feature); the bias–variance trade-off (now a clean
  U-curve with a true minimum); PCA vs. t-SNE (now lifts the data into 50 dimensions so PCA
  genuinely projects); the ReLU "bump" construction (now a true triangular bump); double
  descent (now min-norm least-squares); and the CLIP loss (now the symmetric image–text
  average).

### Fixed

- **Numerical corrections surfaced by verification.** Discrepancies between the printed
  numbers and the code were reconciled, including a mislabeled vanishing-gradient percentage
  (0.000027% → 0.0027%), a gradient-descent iteration transcription error, and several
  worked-arithmetic slips (for example a $27K time-series swing and a 50-unit age effect).
- **Statement-level corrections in the ported prose.** Claims that were wrong or imprecise
  were fixed, including DBSCAN "handles different densities" and border-point assignment,
  self-attention described as permutation-invariant (it is equivariant), a LayerNorm snippet
  using the sample rather than population standard deviation, an AUC/recall conflation in the
  fairness discussion, and a mislabeled "confidence interval" for a cross-validation mean±std.
- **Numerical stability.** The bias–variance and double-descent demos were made reproducible
  across current library versions (a scaled polynomial pipeline; a well-conditioned min-norm
  setup) so their tables and figures no longer drift with scikit-learn/NumPy updates.

[1.0.0]: https://github.com/nkfreeman-teaching/BAN-501-online-course-companion/releases/tag/v1.0.0
