# BAN-501 Online Course Companion — Project Plan & Status Tracker

This is the **living source of truth** for porting, verifying, reviewing, and publishing the BAN-501 online course companion. Update the status markers, the Findings & Decisions Log, and the Resume Prompt at the end of every phase.

**Status markers:** `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked / needs user

## Status summary

| Phase | Title | Status |
|------:|-------|:------:|
| 1 | Repo scaffold & tooling | `[x]` |
| 2 | Port content + figures | `[x]` |
| 3 | Computation verification (98 numbers) | `[ ]` |
| 4 | Freshness checks | `[ ]` |
| 5 | Codex independent second review | `[ ]` |
| 6 | Changelog | `[ ]` |
| 7 | BAN-501 cleanup + instructor guide | `[ ]` |
| 8 | Finalize & publish | `[ ]` |

---

## Context

BAN-501 ("Predictive Modeling," graduate applied-ML) is being reoffered fully online/asynchronous. The student-facing "course companion" book exists as markdown in `../BAN-501/course-companion/` (10 modules + 8 Deep Dives + a Q&A reference). A *previous* offering's online version was published as an MkDocs site in `../BAN-501-course-companion/` (that site exposed only 4 of the deep dives — it is the tooling template, not the content source). This repo ports the **current** content into a fresh MkDocs site, verifies every computed number with code, runs an independent Codex second review, adds a changelog, and retires the duplicated materials from `../BAN-501/` behind an instructor guide.

**Key source facts** (do not re-derive):
- **Content** `../BAN-501/course-companion/`: 19 md files (spaces in names) = 10 modules + 8 deep dives + `QA.md`; 23 PNG figures in `figures/{module1,module6,module7,module8,deep_dive}/`; math `$…$`/`$$…$$`; language-tagged code fences; pipe tables; **blockquote callouts** (`> **Numerical Example:** … > **Output:** … > **Interpretation:** … > *Source:*`); `%20`-encoded `.md` cross-links; **no YAML front matter**. 98 numeric examples each carry a `> *Source:*` citation.
- **Verification** `../BAN-501/course-companion-computations/`: 17 `*_examples.py` (98 `demo_*` functions) + `generate_figures.py` (marimo, writes the 23 PNGs). Verify = run function → diff stdout vs book `Output:` block. Seeds mostly `42` (one `123`; one `range(50)` loop). Some demos are NOT in `__main__` — call individually.
- **Master index** `../BAN-501/CODE_EXAMPLE_MAPPING.md`: 98 functions → file/section/line (line numbers predate May-2026 edits → drifted; regenerate after port).
- **Outline/quizzes** `../BAN-501/`: companion aligned to the 10-module outline; 12 quizzes (163 items) align 1:1. **Quiz answer letters are the deterministic output of `rebalance_answers.py` — do NOT re-run or hand-edit answers except to fix a genuine error, and note any such fix.** Prior audits report all substantive issues fixed as of ~2026-05-07/08 except Q-05b (Blackboard rendering, deferred) + watch items (deep-dive video coverage; residual pandas idioms in Modules 1/2/9; mapping line drift).
- **Env**: pixi 0.72.0; linux-64; RTX A2000 GPU, CUDA 13.0; BAN-501 heavy env already materialized at `../BAN-501/.pixi/envs/default`. Both existing `pixi.toml`s use the modern `[workspace]` table. The pixi platform nuance = feature-level `platforms` scoping + mismatched lockfile schema (6 vs 7) — validated empirically here (see Findings Log).

## Confirmed decisions (2026-07-21)

1. Site includes **all 8 Deep Dives** as appendices.
2. **Move** the computation suite (`*_examples.py`, `generate_figures.py`, `CODE_EXAMPLE_MAPPING.md`) into this repo; two-environment pixi (`docs` default + `compute` feature, feature-scoped to linux-64).
3. **Upgrade** blockquote callouts to **MkDocs Material admonitions**.
4. After full verification (+ final explicit confirmation), **remove** `course-companion/`, `course-companion-computations/`, and companion-specific audit/mapping docs from `../BAN-501/`; add an **instructor user guide** there.
5. **(2026-07-21, Phase 2)** Callout upgrade beyond the enumerated Numerical-Example/Status mapping = **"convert labeled callouts only"**: `> **Note[…]:**`→`note`, `> **Sidebar: …**`→`info`, the test-set `> **Never let…!**`→`warning`, the `**Theorem** (…)` statement→`quote`. Pure prose analogies/intuitions, the epigraph, the QA cross-ref aside, and the "Three-Component Framework" table (which appears un-boxed elsewhere) stay as blockquotes.

## Working conventions (every phase)

- Update **this file** (status, Findings Log, Resume Prompt) at each phase boundary.
- Preserve pedagogy: *Decision Model / Quality Measure / Update Method* framing; notation (`n` obs, `d` features, `C` classes, `K` clusters, `α` LR, `λ` reg — never `p` for features); `random_state=42`; **polars not pandas** (except statsmodels via `.to_pandas()`). Code readable for Master's students with one semester of Python.
- Accuracy & interpretability first; no change merely for change's sake; every load-bearing number traces to code.
- Sub-agents may fan out mechanical work, but **verify every sub-agent + Codex finding** before acting.
- Commit at phase boundaries **only after asking the user**. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- "Done" requires: `mkdocs build --strict` clean; figures read back as images; computation outputs diffed; concrete numbers quoted.

---

## Phase 1 — Repo scaffold & tooling  `[x]`  (done 2026-07-21)

- [x] `PROJECT_PLAN.md` created (this file)
- [x] `pixi.toml` — `[workspace]` (4 platforms) + `docs` (default) & `compute` (linux-64, CUDA 13) features/environments
- [x] `pixi lock` / install both envs; pixi 0.72 platform findings recorded (Findings Log)
- [x] `mkdocs.yml` — Material, MathJax, extensions; nav = Home / 10 Modules / 8 Deep Dives / Q&A / Changelog
- [x] scaffold: `docs/index.md`, `docs/javascripts/mathjax.js`, `docs/stylesheets/extra.css`
- [x] `.github/workflows/deploy.yml` (Actions/Pages, `mkdocs build --strict`)
- [x] `.gitignore`, `README.md`
- [x] placeholder stubs for all nav pages
- [x] `mkdocs build --strict` passes on skeleton (exit 0; 21 pages; built in 0.20s)

**Verify:** `pixi run build` (strict) exit 0; both envs solve; pixi warnings recorded below.

## Phase 2 — Port content + figures  `[x]`  (done 2026-07-21)

1. Copy 23 figures → `docs/assets/{module1,module6,module7,module8,deep_dive}/`.
2. Port + rename 19 files → `docs/modules/NN-slug.md` (10), `docs/appendices/<slug>.md` (8), `docs/reference/qa.md` (1). Slugs already fixed in nav.
3. Transforms (script mechanical parts; fan out per file, verify each):
   - `figures/<sub>/x.png` → `../assets/<sub>/x.png`.
   - Blank lines around every `$$…$$`.
   - `%20`-encoded `.md` cross-links → correct MkDocs relative paths/slugs.
   - **Callout → admonition:** 5-part numerical example → single `!!! example "Numerical Example: <title>"` (4-space-indented python fence, `**Output:**` + plain fence, `**Interpretation:**`, `*Source: \`computations/<file>.py\` — \`fn()\`*`); deep-dive `> **Status:**` → `!!! note "Supplemental reading"`; `> **Theorem**` → `!!! quote "Theorem"`; "Reading the diagram" → `!!! info "Reading the diagram"`.
   - Wording: "this PDF" → "this companion/site"; keep "hands-on".
4. `mkdocs build --strict` clean.
5. Read rendered pages (module + deep-dive sample, one heavy-math, one heavy-code) — math/code/admonitions/tables/figures correct.

**Verify:** strict build clean; no `%20` links or `figures/` paths remain; visual spot-check.

## Phase 3 — Computation verification (accuracy-critical)  `[ ]`

1. Move `course-companion-computations/*` → `computations/`; `CODE_EXAMPLE_MAPPING.md` → repo root. Update `generate_figures.py` `OUTPUT_DIR` → `docs/assets/`. (Copy, not delete from BAN-501, until Phase 7.)
2. `computations/verify_all.py`: import each `*_examples.py`, call all 98 `demo_*` (incl. those not in `__main__`), capture stdout, compare to book numbers. Run under `pixi run -e compute` (seed 42, GPU parity).
3. Verification report (pass/fail per number). Mismatch → investigate: code right → update book number + **propagate**; code wrong → fix code. Log every change.
4. Regenerate 23 figures; **read each back as image** (figure rubric); confirm parity/improvement.
5. Refresh `CODE_EXAMPLE_MAPPING.md` (docs slugs + current line numbers).

**Verify:** every numeric claim matches code output (quote count + deltas); figures reviewed as images.

## Phase 4 — Freshness checks  `[ ]`

1. Library currency (sklearn ≥1.8, torch ≥2.10/torchvision ≥0.26, transformers ≥5.2, seaborn ≥0.13, polars ≥1.38, numpy ≥2.4, xgboost ≥3.2). Cross-check prior audits — don't reintroduce fixed items (`sparse_output`, `weights=…`, `penalty=None`→`C=np.inf`, XGBoost `early_stopping_rounds`).
2. Outline alignment (section-by-section); confirm all 8 deep-dive cross-refs exist in parent modules.
3. Quizzes: note-only check (do NOT modify; answer letters are `rebalance_answers.py` output). Escalate genuine errors to user.
4. pixi platform concern: document this repo's setup + empirical 0.72 behavior; report (don't silently fix) needed changes to `../BAN-501/{pixi.toml,notebooks/pixi.toml}`.
5. Residual pandas idioms scan (Modules 1/2/9).
6. `mkdocs build --strict` (dead links); "site vs PDF" wording sweep.

**Verify:** re-run touched computations; strict rebuild; no fixed-issue regressions.

## Phase 5 — Codex independent second review  `[ ]`

1. Codex CLI/plugin reviews in batches: (a) content accuracy/clarity, (b) computation correctness, (c) freshness findings, (d) port fidelity.
2. Reconcile Claude vs Codex per disagreement; verify against code/source; iterate to agreement. Log all.
3. Apply agreed fixes (propagate) + re-verify.

**Verify:** no open disagreements; strict build + affected computations clean.

## Phase 6 — Changelog  `[ ]`

1. `CHANGELOG.md` (Keep a Changelog, semver) at root; seed `1.0.0 — Online edition` entry.
2. Surface as `docs/changelog.md` (snippet include or thin copy) + nav.
3. Document update workflow (edit `docs/`, `pixi run -e compute verify`, changelog entry, push → auto-deploy) — feeds Phase 7 guide.

**Verify:** changelog renders in built site.

## Phase 7 — BAN-501 cleanup + instructor user guide  `[ ]`

**Touches a different repo — explicit final confirmation before any deletion.**
1. Pre-flight: confirm this repo fully supersedes source (ported + verified, computations moved + passing, figures regenerated). **Ask user to confirm.**
2. Remove from `../BAN-501/`: `course-companion/`, `course-companion-computations/`, companion-specific docs (`COURSE_COMPANION_AUDIT.md`, `CODE_EXAMPLE_MAPPING.md`, `build-companion` task in `pixi.toml`). **Keep** broader docs (`course-outline-content-audit.md`, `module-quizzes/*`) — trim only companion-specific sections.
3. Add instructor guide `../BAN-501/course-companion-GUIDE.md` (what/where the online companion lives, how to update it, how to rebuild figures). Update `../BAN-501/CLAUDE.md` references to removed paths.

**Verify:** no dangling references in `../BAN-501/` to deleted paths; guide accurate.

## Phase 8 — Finalize & publish  `[ ]`

1. Final `mkdocs build --strict` + full nav read-through; confirm 98 numbers, figures, links.
2. Verify `deploy.yml`; note required Pages "Source: GitHub Actions" setting.
3. Commit + push (ask first); confirm Actions build + Pages deploy.
4. Final status + close-out entry here.

**Verify:** live site loads; spot-checked module renders math/code/figures/admonitions.

---

## Findings & Decisions Log

*(Append dated entries as work proceeds. Record pixi platform findings, computation verification results, freshness findings, Codex reconciliation, and any content changes with rationale.)*

- **2026-07-21 — Phase 1 started.** Scaffold created (pixi two-env, mkdocs.yml full nav, docs skeleton, CI, stubs).

- **2026-07-21 — pixi platform concern (RESOLVED / documented).** This is the "recent pixi change" the user flagged. Under **pixi 0.72.0**:
  - Both existing BAN-501 tomls and this repo already use the modern `[workspace]` table (not the deprecated `[project]`), so no `[project]`→`[workspace]` migration is needed.
  - `pixi lock` emits: *"the `[system-requirements]` table is deprecated in favor of virtual packages on `platforms`"*, suggesting `platforms = [{ platform = "linux-64", cuda = "13.0" }]`.
  - **But that inline table-array syntax is NOT accepted by pixi 0.72.0** — it errors with `expected a string, found table`. So the deprecation is *announced ahead of* the replacement being landable.
  - **Action taken:** kept the working `[feature.compute] platforms = ["linux-64"]` + `[feature.compute.system-requirements] cuda = "13.0"` form (functions correctly; warning only). Documented in `pixi.toml` and `README.md`.
  - **For Phase 4 → user:** the BAN-501 root `pixi.toml` (`[system-requirements] cuda="13.0"`) and `notebooks/pixi.toml` (`[feature.gpu.system-requirements] cuda="13.0"`) emit the same deprecation warning but **should NOT be migrated yet** (the new syntax errors in 0.72.0). Also note `notebooks/pixi.lock` is schema v7 vs root `pixi.lock` v6 — a re-lock would align them; not urgent.
  - Both envs solved cleanly: **compute** → python 3.14.6, pytorch-gpu 2.11.0 (cuda129), torchvision 0.26.0, transformers 5.14.1, scikit-learn 1.9.0, numpy 2.4.6, xgboost 3.3.0, polars 1.42.1, scipy 1.18.0, marimo 0.20.4, shap 0.51.0, statsmodels 0.14.6. **docs** → mkdocs 1.6.1, mkdocs-material 9.7.7 (python 3.14.6), all 4 platforms.
  - **Phase-3 note:** the compute env resolved some libs NEWER than what produced the book numbers (sklearn 1.9 vs ≥1.8; transformers 5.14 vs ≥5.2; pytorch 2.11). Minor low-order-digit drift is possible; treat any drift beyond the book's reported precision as a freshness finding (update + propagate).

- **2026-07-21 — MkDocs 2.0 advisory (FYI, no action).** mkdocs-material 9.7.7 prints a banner warning that a future MkDocs 2.0 will be backward-incompatible (plugins/theming). Informational only; does not affect this build.

- **2026-07-21 — Phase 1 complete.** `pixi run build` (`mkdocs build --strict`) exits 0, generates 21 pages (home + 10 modules + 8 deep dives + qa + changelog) in ~0.2s. Skeleton is buildable with the full intended nav; module/appendix/qa pages are placeholder stubs to be filled in Phase 2, changelog in Phase 6.
  - Committed to `main` as `8cbe679` (initial commit). Not pushed — this local mirror shows `origin/main [gone]`; the GitHub remote may need to be (re)created before the Phase 8 deploy.

- **2026-07-21 — Phase 2 complete (port content + figures).** All 19 source md files ported into `docs/` (10 → `modules/NN-slug.md`, 8 → `appendices/<slug>.md`, QA → `reference/qa.md`) and 23 figures copied → `docs/assets/{module1,module6,module7,module8,deep_dive}/`. Executed via two deterministic, content-preserving scripts (in scratchpad) rather than generative rewriting, to guarantee the 98 numeric outputs survive byte-for-byte.
  - **Transforms applied (counts):** 23 image paths `figures/…`→`../assets/…`; 8 `%20`-encoded `.md` cross-links → correct relative slugs (all module→appendix, e.g. `../appendices/imbalanced-data.md`); 111 single-line `$$…$$` display equations blank-line-normalized — **all display math is single-line (no lone `$$`)**, and consecutive runs (e.g. the 6 LSTM gate equations in Module 8) get exactly one blank inserted between neighbors, not collapsed; source citations `course-companion-computations/`→`computations/` and ` - `→` — ` (em-dash) per the mapping.
  - **Callouts:** 98 five-part Numerical Example blockquotes → `!!! example "Numerical Example: …"` (4-space-indented python fence, blank injected before the `**Output:**` fence, `*Source: \`computations/…\` — \`fn()\`*`). 8 deep-dive `> **Status:**` → `!!! note "Supplemental reading"`. Plan's `> **Theorem**` / "Reading the diagram" mappings were **moot** — neither pattern exists in the corpus. The "this PDF"→site wording transform found **nothing to change** (no PDF/print references exist; QA's generic "This document" is web-appropriate, left as-is).
  - **Labeled-blockquote sub-decision (user, this session):** "convert labeled callouts only" → 7 additional conversions: 4 `Note[…]`→`note`, 1 `Sidebar: Batch Normalization`→`info`, 1 test-set `Warning`, 1 `Theorem (Cybenko…)`→`quote`. **12** blockquotes intentionally left (6 cnn-architecture prose asides, the data-prep epigraph, 3 "Intuition/Analogy" blocks, the "Three-Component Framework: CNNs" table, the QA cross-ref aside). See Confirmed decisions #5.
  - **Fidelity verification (deterministic):** a checker compares every Numerical-Example fence's de-quoted source content vs de-indented dest content → **ZERO byte differences across all 98 examples** (the numbers Phase 3 verifies are intact). Whole-file content-signature diff = exactly the 7 folded bold-label lines (moved into admonition titles), no other content missing/added on either side. All 7 converted bodies confirmed present verbatim.
  - **Build & negative checks:** `mkdocs build --strict` exit 0, 21 pages, ~1.2s, no link/nav warnings. Zero stray `figures/`, `%20`, `course-companion-computations/`, or raw `> **Numerical Example`/`> **Status:` remain in `docs/`. Final rendered-HTML admonition inventory: **example=98, note=12 (8 status + 4 Note), info=1, warning=1, quote=1**.
  - **Rendered spot-checks (headless google-chrome-stable, served build):** Module 3 (math + example admonition + tables), Module 8 (22 display eqs incl. LSTM run rendering as separate blocks; TF-IDF/Word2Vec math; syntax-highlighted code), Module 9 (heavy code), universal-approximators (Supplemental-reading `note`, `quote` Theorem with `ℝⁿ`, multi-panel figures from `../../assets/deep_dive/`), data-preparation (`note` + epigraph-as-blockquote + polars code + table), Module 7 (`info` Sidebar), data-prep (`warning`). All math (MathJax), code (pygments), admonitions, tables, and figures render correctly. (Screenshots in scratchpad; not committed.)
  - **Not committed** (per "commit only after asking"). Working tree now has 19 populated `docs/` pages + 23 assets + the modified `PROJECT_PLAN.md`.

---

## Resume Prompts

Paste the block for the phase you're resuming into a fresh session.

> **Resume — Phase 3 (Computation verification — 98 numbers).** Repo: `/home/nick/Desktop/git-repos/github-backup/nkfreeman-teaching/BAN-501-online-course-companion`. Read `PROJECT_PLAN.md` first (status + Confirmed decisions + Findings Log). Phases 1–2 are complete: the MkDocs site is fully ported (19 files in `docs/`, 23 figures in `docs/assets/`) and builds `--strict` clean; all **98** numerical examples are `!!! example` admonitions whose `*Source:*` lines already cite `computations/<file>.py — demo_*()` (the not-yet-moved location) — verified byte-identical to source. Now execute Phase 3, running Python under `pixi run -e compute` (NOT the docs env): **(1)** COPY `../BAN-501/course-companion-computations/*` → `computations/` and `../BAN-501/CODE_EXAMPLE_MAPPING.md` → repo root (do **not** delete from BAN-501 until Phase 7); set `generate_figures.py` `OUTPUT_DIR` → `docs/assets/`. **(2)** Write `computations/verify_all.py` that imports each of the 17 `*_examples.py`, calls all 98 `demo_*` functions (including those **not** under `__main__` — call individually), captures stdout, and diffs against the `**Output:**` blocks in the ported `docs/` pages (seeds mostly 42; one 123; one `range(50)` loop; RTX A2000 / CUDA 13 GPU parity). **(3)** Produce a pass/fail report per number; on mismatch investigate — code right → update the book number in `docs/` **and propagate** to every dependent number/figure; code wrong → fix code; log every change in the Findings Log. **(4)** Regenerate all 23 figures and **read each back as an image** against the figure rubric. **(5)** Refresh `CODE_EXAMPLE_MAPPING.md` with the new `docs/` slugs + current line numbers. Watch item: the `compute` env resolved some libs **newer** than produced the book numbers (sklearn 1.9, transformers 5.14, pytorch 2.11) — treat drift beyond the book's reported precision as a freshness finding (update + propagate). Honor conventions (`random_state=42`, polars, **quizzes are fixed — do not edit**). Verify: every numeric claim matches code output (quote the count + any deltas); strict rebuild clean; figures reviewed as images. Then update `PROJECT_PLAN.md` status + Findings Log and write the Phase 4 resume prompt.
