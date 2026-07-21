"""
verify_all.py — Phase 3 computation verification for the BAN-501 online companion.

Runs every ``demo_*`` function in the 17 ``*_examples.py`` files, captures each
function's stdout, and checks that the numbers presented in the ported course
book (``docs/**/*.md``) are actually produced by the code.

Design notes
------------
* Each demo is a zero-argument, pure-stdout function. Randomness-using demos
  re-seed themselves (mostly ``np.random.seed(42)``; a couple use 123; the
  torch demos add ``torch.manual_seed(42)``), and every file's ``__main__`` /
  ``main()`` calls the demos in definition order. So running each file's demos
  in definition order inside one process reproduces the exact conditions under
  which the book numbers were generated. We deliberately do NOT reset RNG state
  between calls.
* The book "Output" is a *curated* transcription of stdout (relabeled headers,
  β-subscripts, markdown tables), so a byte diff is meaningless. Instead we do a
  numeric-aware comparison: every number the book shows for a demo must appear
  in that demo's stdout at the book's displayed precision.
* Duplicate function names exist across files, so every check is keyed on the
  (file, function) pair parsed from each admonition's ``*Source:*`` line.

Run with:  pixi run -e compute verify        (i.e. python computations/verify_all.py)
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMP_DIR = REPO / "computations"
DOCS_DIR = REPO / "docs"
OUT_DIR = COMP_DIR / "_verify"
RAW_DIR = OUT_DIR / "raw"

# The 17 computation files, in the order the mapping lists them.
EXAMPLE_FILES = [
    "module2_examples.py",
    "module3_examples.py",
    "module4_examples.py",
    "module5_examples.py",
    "module6_examples.py",
    "module7_examples.py",
    "module8_examples.py",
    "module9_examples.py",
    "module10_examples.py",
    "deep_dive_data_prep_examples.py",
    "deep_dive_evaluation_examples.py",
    "deep_dive_imbalanced_examples.py",
    "deep_dive_timeseries_examples.py",
    "deep_dive_cnn_examples.py",
    "deep_dive_transformer_examples.py",
    "deep_dive_universal_approx_examples.py",
    "deep_dive_surprising_phenomena_examples.py",
]

# ---------------------------------------------------------------------------
# Number extraction / comparison
# ---------------------------------------------------------------------------

# A numeric token not glued to a letter/underscore identifier (so "x1", "beta_0",
# "10x" do not yield spurious digits). Handles sign, decimals, scientific form.
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_])"
)
INF_RE = re.compile(r"(?<![A-Za-z0-9_])(?:inf|-inf|∞|nan)(?![A-Za-z0-9_])", re.I)


def _normalize(text: str) -> str:
    """Fold typographic number forms so tokenization sees plain ASCII numbers.

    * U+2212 minus sign (−) written in markdown tables -> ASCII '-'.
    * Thousands separators (1,024 / 196,672 / 50,176) -> joined digits.
    """
    text = text.replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return text


def _decimals(token: str) -> int:
    """Number of decimal places displayed in a plain-decimal token."""
    if "." in token:
        return len(token.split(".", 1)[1])
    return 0


def extract_numbers(text: str):
    """Return [(raw, value, decimals, is_pct, is_sci)] for every number in text.

    is_pct marks a value immediately followed by '%' (the book often prints a
    fraction the code emits as a percentage, or vice versa).
    is_sci marks scientific notation (compared on value, not decimal places).
    """
    text = _normalize(text)
    out = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        try:
            val = float(tok)
        except ValueError:
            continue
        is_sci = "e" in tok or "E" in tok
        nxt = text[m.end():m.end() + 1]
        is_pct = nxt == "%"
        out.append((tok, val, _decimals(tok), is_pct, is_sci))
    return out


def has_inf(text: str) -> bool:
    return bool(INF_RE.search(_normalize(text)))


def stdout_number_set(stdout: str):
    """All numeric values in stdout."""
    return [v for (_, v, _, _, _) in extract_numbers(stdout)]


def _match_one(value: float, decimals: int, is_sci: bool, stdout_vals) -> bool:
    if is_sci:  # scientific notation: relative tolerance
        tol = max(abs(value) * 1e-3, 1e-12)
        return any(abs(v - value) <= tol for v in stdout_vals)
    target = round(value, decimals)
    step = 10 ** (-decimals)
    # EXACT match at the book's displayed precision: round each stdout value to
    # the book's decimals and require equality. No last-digit slack -- a book
    # number that differs by a full displayed unit is a real mismatch, not a
    # match (that slack could otherwise let a wrong number, e.g. book 71 vs
    # stdout 70, pass silently). Prose-derived numbers that legitimately do not
    # appear in stdout are documented explicitly in ADJUDICATED instead.
    return any(abs(round(v, decimals) - target) < 0.5 * step for v in stdout_vals)


def book_number_found(value, decimals, stdout_vals, is_pct=False, is_sci=False) -> bool:
    """Is a book number reproduced somewhere in stdout, at book precision?

    Tries the value as printed and, for percentages, the fraction form (value/100)
    -- so a book '66%' matches a code-printed 0.661, and '99.75%' matches 0.9975.
    """
    if _match_one(value, decimals, is_sci, stdout_vals):
        return True
    if is_pct:
        frac = value / 100.0
        # a fraction carries two more significant decimals than the percentage
        if _match_one(frac, decimals + 2, is_sci, stdout_vals):
            return True
    return False


# ---------------------------------------------------------------------------
# Docs admonition parsing:  (source_file, func) -> book text for that example
# ---------------------------------------------------------------------------

SOURCE_RE = re.compile(
    r"\*Source:\s*`computations/([A-Za-z0-9_]+\.py)`\s*[–—-]\s*`(demo_[A-Za-z0-9_]+)\(\)`\*"
)
PYFENCE_RE = re.compile(r"```python.*?```", re.S)
ANYFENCE_RE = re.compile(r"```.*?```", re.S)


def parse_docs():
    """Map (file, func) -> dict(full_body, output_block) from every admonition."""
    examples: dict[tuple[str, str], dict] = {}
    for md in sorted(DOCS_DIR.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        lines = text.splitlines()
        i = 0
        n = len(lines)
        while i < n:
            if lines[i].lstrip().startswith("!!! example"):
                # Collect the admonition block: subsequent blank or 4-space lines.
                block = [lines[i]]
                j = i + 1
                while j < n and (lines[j].strip() == "" or lines[j].startswith("    ")):
                    block.append(lines[j])
                    j += 1
                body = "\n".join(block)
                msrc = SOURCE_RE.search(body)
                if msrc:
                    key = (msrc.group(1), msrc.group(2))
                    examples[key] = {
                        "body": body,
                        "page": str(md.relative_to(REPO)),
                    }
                i = j
            else:
                i += 1
    return examples


def book_numbers_for(body: str):
    """Extract book numbers to check: admonition body minus python code + source."""
    # Drop the *Source:* line and python input fences (inputs, not outputs).
    no_src = SOURCE_RE.sub("", body)
    no_py = PYFENCE_RE.sub("", no_src)
    nums = extract_numbers(no_py)
    inf = has_inf(no_py)
    return nums, inf


# ---------------------------------------------------------------------------
# Running the demos
# ---------------------------------------------------------------------------

def import_module_from(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def demo_functions_in_order(mod):
    demos = [
        (name, obj)
        for name, obj in vars(mod).items()
        if name.startswith("demo_") and callable(obj)
        and getattr(obj, "__module__", None) == mod.__name__
    ]
    demos.sort(key=lambda kv: kv[1].__code__.co_firstlineno)
    return demos


def run_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}  # (file, func) -> stdout
    for fname in EXAMPLE_FILES:
        path = COMP_DIR / fname
        mod = import_module_from(path)
        for name, fn in demo_functions_in_order(mod):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    fn()
                out = buf.getvalue()
            except Exception as exc:  # noqa: BLE001
                out = buf.getvalue() + f"\n<<EXCEPTION>> {type(exc).__name__}: {exc}\n"
            results[(fname, name)] = out
            (RAW_DIR / f"{fname[:-3]}__{name}.txt").write_text(out, encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adjudicated residuals: book numbers that legitimately do NOT appear verbatim
# in stdout. Each was traced by hand during Phase 3 (2026-07-21) and confirmed
# correct -- prose-derived arithmetic, "~" approximations, sign conventions,
# rounded presentation tables, or reference constants, not claims the demo
# prints. Keyed by (file, func) -> {tokens, reason}. If a function's actual
# unmatched set grows beyond `tokens`, the extra token is reported as an
# UNEXPLAINED flag, so this allowlist cannot hide a newly-introduced mismatch.
# ---------------------------------------------------------------------------
ADJUDICATED = {
    ("deep_dive_cnn_examples.py", "demo_fc_vs_cnn_parameters"): {
        "tokens": {"150", "84000"},
        "reason": "'150 million' prose rounding of printed 150,529,000; '~84,000x' prose "
                  "rounding of the printed exact ratio 84,001x",
    },
    ("deep_dive_cnn_examples.py", "demo_pooling_dimensions"): {
        "tokens": {"50176", "49", "512"},
        "reason": "prose arithmetic 224^2=50,176 and 7^2=49; 512 is a VGG channel-count reference",
    },
    ("deep_dive_cnn_examples.py", "demo_weight_sharing_savings"): {
        "tokens": {"50000"},
        "reason": "'nearly 50,000x' prose rounding of the printed exact 49,284x",
    },
    ("deep_dive_data_prep_examples.py", "demo_feature_scaling_impact"): {
        "tokens": {"170000", "50"},
        "reason": "nominal design ranges stated in prose: income $170,000 (=200k-30k) and age "
                  "50 years (=70-20); code prints the sampled ranges (167,525 / ~49)",
    },
    ("deep_dive_surprising_phenomena_examples.py", "demo_grokking_simulation"): {
        "tokens": {"97", "1"},
        "reason": "prose random-chance baseline 'test accuracy near random (1/97 ~ 1%)': the "
                  "1/97 fraction and the ~1% both come from p=97, not from the trajectory table",
    },
    ("deep_dive_imbalanced_examples.py", "demo_precision_recall_threshold"): {
        "tokens": {"71", "91"},
        "reason": "prose arithmetic in the interpretation: '71% of positive predictions wrong' "
                  "= 100 - 29 (precision at 0.1); 'miss 91% of actual positives' = 100 - 9 (recall at 0.7)",
    },
    ("deep_dive_transformer_examples.py", "demo_positional_encoding"): {
        "tokens": {"7"},
        "reason": "'dim6-7' prose dimension index (d_model=8); the 8 encoding values per "
                  "position are printed, not the index",
    },
    ("module6_examples.py", "demo_backprop_by_hand"): {
        "tokens": {"-1"},
        "reason": "the '-1' in the derivative formula 'dL/dy_pred = -1/y_pred'; the computed "
                  "result -1.9231 is printed",
    },
    ("deep_dive_timeseries_examples.py", "demo_time_series_decomposition"): {
        "tokens": {"46000", "43000", "64000", "70000"},
        "reason": "book rounds the decomposition table to the nearest $1,000 for presentation "
                  "(code prints exact 45,993 / 42,723 / ...); trend & seasonal structure preserved",
    },
    ("deep_dive_timeseries_examples.py", "demo_walk_forward_vs_kfold"): {
        "tokens": {"0.07"},
        "reason": "book reports |k-fold - walk-forward| = 0.07; code prints the signed -0.07",
    },
    ("module10_examples.py", "demo_ab_test_sample_size"): {
        "tokens": {"62000", "244000"},
        "reason": "'~62,000' / '~244,000' prose approximations of printed 62,470 / 244,250",
    },
    ("module4_examples.py", "demo_oob_vs_cv"): {
        "tokens": {"0.0120", "1.2"},
        "reason": "book reports |OOB-CV| = 0.0120 (code signed -0.0120); '1.2 percentage points' = 0.012",
    },
    ("module4_examples.py", "demo_rf_vs_single_tree"): {
        "tokens": {"22"},
        "reason": "'22% lower standard deviation' derived from printed 0.032 -> 0.025 (not printed directly)",
    },
    ("module6_examples.py", "demo_relu_vs_sigmoid_gradients"): {
        "tokens": {"1.63"},
        "reason": "'1.63x10^-10' written with Unicode superscript in prose; printed as 1.63e-10 (matched)",
    },
    ("module7_examples.py", "demo_cnn_vs_fc_parameters"): {
        "tokens": {"1024"},
        "reason": "prose arithmetic: 32x32 = 1,024 spatial positions",
    },
    ("module7_examples.py", "demo_fc_parameter_explosion"): {
        "tokens": {"150"},
        "reason": "'150 million' prose rounding of the printed 150,529,000",
    },
    ("module8_examples.py", "demo_bert_vs_gpt_scale"): {
        "tokens": {"10", "16000"},
        "reason": "illustrative cost scenario ($10 vs $16,000); not a computed output",
    },
    ("module8_examples.py", "demo_positional_encoding"): {
        "tokens": {"6000"},
        "reason": "'~6,000 positions' prose approximation of printed wavelength 6283.2",
    },
    ("module9_examples.py", "demo_shap_sum_to_prediction"): {
        "tokens": {"0.301", "0.086", "4", "3", "2"},
        "reason": "book prints +/- in a separate column ('feature_2  -  0.301'); values match with sign. "
                  "'features 2 and 3', 'Feature 4' are prose indices",
    },
}


def main():
    print("Running 98 demo functions under", sys.executable)
    results = run_all()
    print(f"Captured stdout for {len(results)} functions.\n")

    examples = parse_docs()
    print(f"Parsed {len(examples)} example admonitions from docs/.\n")

    # Cross-check coverage.
    run_keys = set(results)
    doc_keys = set(examples)
    missing_in_docs = sorted(run_keys - doc_keys)
    missing_in_code = sorted(doc_keys - run_keys)

    report_lines = []
    n_pass = n_flag = n_err = n_adj = n_nonum = 0
    flagged = []

    for key in sorted(run_keys | doc_keys):
        fname, func = key
        stdout = results.get(key)
        ex = examples.get(key)
        if stdout is None:
            report_lines.append(f"[NO-CODE ] {fname} :: {func}  (in docs, no function run)")
            continue
        if "<<EXCEPTION>>" in stdout:
            n_err += 1
            flagged.append(key)
            report_lines.append(f"[ERROR   ] {fname} :: {func}  (raised exception)")
            continue
        if ex is None:
            report_lines.append(f"[NO-DOCS ] {fname} :: {func}  (ran, no admonition found)")
            continue

        nums, book_inf = book_numbers_for(ex["body"])

        # Qualitative examples carry no numbers to check. Report them as
        # NO-NUMS rather than silently awarding an (unverifiable) PASS.
        if not nums and not book_inf:
            n_nonum += 1
            report_lines.append(
                f"[NO-NUMS ] {fname} :: {func}  (qualitative example, no numbers to verify)"
            )
            continue

        stdout_vals = stdout_number_set(stdout)
        not_found = []
        for tok, val, dec, is_pct, is_sci in nums:
            if not book_number_found(val, dec, stdout_vals, is_pct, is_sci):
                not_found.append(tok)
        inf_ok = (not book_inf) or has_inf(stdout)

        adj = ADJUDICATED.get(key)
        residual = not_found
        if adj is not None:
            residual = [t for t in not_found if t not in adj["tokens"]]

        if not not_found and inf_ok:
            n_pass += 1
            report_lines.append(
                f"[PASS    ] {fname} :: {func}  ({len(nums)} numbers matched)"
            )
        elif not residual and inf_ok:
            # Every unmatched token is a documented prose/formatting number.
            n_adj += 1
            report_lines.append(
                f"[ADJUDGD ] {fname} :: {func}  ({len(nums)} numbers; "
                f"{len(not_found)} prose/formatting, verified: {adj['reason']})"
            )
        else:
            n_flag += 1
            flagged.append(key)
            detail = []
            if residual:
                detail.append(f"{len(residual)} UNEXPLAINED not in stdout: {residual[:15]}")
            if not inf_ok:
                detail.append("book shows inf/∞ but stdout does not")
            report_lines.append(
                f"[FLAG    ] {fname} :: {func}  ({len(nums)} numbers; " + "; ".join(detail) + ")"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "# Phase 3 verification report",
        "",
        f"- functions run: {len(results)}",
        f"- admonitions in docs: {len(examples)}",
        f"- PASS: {n_pass}   ADJUDICATED (prose/formatting, verified): {n_adj}"
        f"   NO-NUMS (qualitative): {n_nonum}"
        f"   UNEXPLAINED FLAG: {n_flag}   ERROR: {n_err}",
        f"- verified = PASS + ADJUDICATED = {n_pass + n_adj}; "
        f"qualitative (no numbers) = {n_nonum}; "
        f"clean (no FLAG/ERROR) over {len(results)} = {n_flag == 0 and n_err == 0}",
        f"- keys run but no admonition: {missing_in_docs}",
        f"- admonitions but no code: {missing_in_code}",
        "",
        "## Per-function",
        "",
    ]
    (OUT_DIR / "report.md").write_text(
        "\n".join(header + report_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(report_lines))
    print()
    print("=" * 70)
    print(f"PASS={n_pass}  ADJUDICATED={n_adj}  NO-NUMS={n_nonum}  "
          f"UNEXPLAINED FLAG={n_flag}  ERROR={n_err}  (of {len(results)} functions)")
    print(f"verified (PASS+ADJUDICATED) = {n_pass + n_adj}/{len(results)}  "
          f"| qualitative (no numbers) = {n_nonum}  "
          f"| clean (no FLAG/ERROR) = {n_flag == 0 and n_err == 0}")
    if missing_in_docs:
        print(f"Ran but no admonition matched: {missing_in_docs}")
    if missing_in_code:
        print(f"Admonition but no code matched: {missing_in_code}")
    print(f"Raw outputs: {RAW_DIR}")
    print(f"Report: {OUT_DIR / 'report.md'}")
    print("=" * 70)
    return 0 if (n_flag == 0 and n_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
