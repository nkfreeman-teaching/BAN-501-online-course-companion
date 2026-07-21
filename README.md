# BAN 501 Online Course Companion

The student-facing reading companion for **BAN 501: Predictive Modeling** (online, asynchronous offering), built with [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages via GitHub Actions.

**Live site:** https://nkfreeman-teaching.github.io/BAN-501-online-course-companion/

This repo also contains the **computation-verification suite** that reproduces every numerical example in the companion, so the numbers in the book are checkable and reproducible.

> **Project status.** The port/verification/review work is tracked in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (the living plan + status tracker). New instructors updating the companion should read that file and the update workflow below.

## Repository structure

```
BAN-501-online-course-companion/
├── docs/                       # Site content (Markdown)
│   ├── index.md                # Home
│   ├── modules/                # Modules 1-10
│   ├── appendices/             # 8 Deep Dives (supplemental)
│   ├── reference/qa.md         # Q&A reference
│   ├── changelog.md            # Rendered changelog
│   ├── assets/                 # Figures (PNG), by module / deep_dive
│   ├── javascripts/mathjax.js  # MathJax 3 config
│   └── stylesheets/extra.css   # Custom styles
├── computations/               # Computation-verification suite (Phase 3)
├── .github/workflows/deploy.yml
├── mkdocs.yml
├── pixi.toml                   # Two environments: docs (default) + compute
├── CHANGELOG.md                # Canonical changelog (Phase 6)
└── PROJECT_PLAN.md             # Living project plan & status tracker
```

## Environments (pixi)

Two [pixi](https://pixi.sh/) environments:

- **`docs`** (default, all OSes) — builds/serves the site. Lightweight (mkdocs + material).
- **`compute`** (linux-64 + CUDA) — re-runs the 98 computation verifications and regenerates figures. Heavy (pytorch-gpu, scikit-learn, transformers, etc.).

```bash
# Site
pixi run serve            # local dev server at http://127.0.0.1:8000
pixi run build            # build the static site (strict)

# Verification (heavy env; GPU/linux-64)
pixi run -e compute verify    # re-run all computation checks
pixi run -e compute figures   # regenerate figures into docs/assets/
```

> **pixi note (0.72.0):** the `compute` feature declares `cuda` via a `[system-requirements]` table. pixi 0.72 prints a deprecation warning suggesting an inline `platforms = [{ platform = "linux-64", cuda = "13.0" }]` form, but that syntax is not yet accepted by 0.72 (it errors). Keep the current form until a pixi release supports the replacement. See `PROJECT_PLAN.md` Findings Log.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site (`mkdocs build --strict`) and deploys it to GitHub Pages.

**One-time repo setting:** GitHub → Settings → Pages → **Source: GitHub Actions**.

## Updating the companion

1. **Edit content** in `docs/` (modules in `docs/modules/`, deep dives in `docs/appendices/`, images in `docs/assets/`).
2. If you change a numerical example, **re-verify**: `pixi run -e compute verify` (and `pixi run -e compute figures` if a figure changed).
3. **Preview:** `pixi run serve`.
4. **Add a `CHANGELOG.md` entry** describing the change (edit the root `CHANGELOG.md`; the in-site `docs/changelog.md` page renders it automatically via a snippet include, so there is only one file to update).
5. **Commit and push** to `main` — GitHub Actions rebuilds and redeploys.

### Content conventions

- **Math:** inline `$…$`, display `$$…$$` (blank lines before/after display blocks). Rendered by MathJax 3 via `pymdownx.arithmatex`.
- **Figures:** `![Alt](../assets/<subdir>/name.png)`.
- **Callouts:** Material admonitions, e.g. `!!! example "Numerical Example: …"`, `!!! note`, `!!! info`.
- **Code:** language-tagged fenced blocks; polars (not pandas) except where statsmodels requires pandas.
