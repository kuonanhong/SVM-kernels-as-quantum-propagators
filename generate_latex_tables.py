#!/usr/bin/env python3
"""Generate manuscript-ready LaTeX tables from second-round benchmark CSVs.

Direct execution is supported: when --results-root is omitted, the script searches
recursively beneath the directory containing this Python file.  Full benchmark
results are required for submission; quick/smoke folders remain excluded unless
--allow-quick is explicitly supplied.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "Copper": "copper_round2_cv_summary.csv",
    "Graphene": "graphene_round2_cv_summary.csv",
    "Anharmonic oscillator": "anharmonic_round2_cv_summary.csv",
    "Photonic crystal": "photonic_round2_cv_summary.csv",
    "Fibonacci chain": "quasicrystal_round2_cv_summary.csv",
}

TRADEOFFS = {
    "Photonic crystal": "photonic_round2_nystrom_rank_tradeoff.csv",
    "Fibonacci chain": "quasicrystal_round2_nystrom_rank_tradeoff.csv",
}

SCALABILITY = {
    "Photonic crystal": "photonic_round2_scalability.csv",
    "Fibonacci chain": "quasicrystal_round2_scalability.csv",
}


COLUMN_ALIASES = {
    "Model": ("Model", "model"),
    "CV_MSE_Mean": ("CV_MSE_Mean", "MSE_Mean", "MSEMean"),
    "CV_MSE_CI95": ("CV_MSE_CI95", "MSE_CI95", "MSECI95"),
    "CV_R2_Mean": ("CV_R2_Mean", "R2_Mean", "R2Mean"),
    "CV_R2_CI95": ("CV_R2_CI95", "R2_CI95", "R2CI95"),
    "FitTime_Mean": (
        "FitTime_Mean",
        "FitTime_MeanSec",
        "FitTimeSec_Mean",
    ),
    "FitTime_CI95": (
        "FitTime_CI95",
        "FitTime_CI95Sec",
        "FitTimeSec_CI95",
    ),
}


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(x):
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 10 ** (-(digits + 1)) or ax >= 10 ** (digits + 2)):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def pm(mean: object, ci: object, digits: int = 3) -> str:
    return f"{fmt(mean, digits)} $\\pm$ {fmt(ci, digits)}"


def discover(
    root: Path,
    filename: str,
    allow_quick: bool,
    strict: bool,
) -> Path | None:
    """Find one result file, preferring the most recently modified full result."""
    hits = sorted(root.rglob(filename))
    if not allow_quick:
        hits = [
            path
            for path in hits
            if "quick" not in str(path).lower()
            and "smoke" not in str(path).lower()
        ]

    if not hits:
        return None

    if len(hits) > 1:
        if strict:
            joined = "\n  ".join(str(path) for path in hits)
            raise RuntimeError(
                f"Multiple files named {filename} were found. "
                f"Use a narrower --results-root:\n  {joined}"
            )
        hits.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(
            f"[WARNING] Found {len(hits)} copies of {filename}; "
            f"using the newest: {hits[0]}"
        )

    return hits[0]


def canonicalize_cv_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Accept the small naming differences used by the round-2 scripts."""
    renamed = df.copy()
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        source = next((name for name in aliases if name in renamed.columns), None)
        if source is not None and source != canonical:
            rename_map[source] = canonical

    renamed = renamed.rename(columns=rename_map)
    required = {
        "Model",
        "CV_MSE_Mean",
        "CV_MSE_CI95",
        "CV_R2_Mean",
        "CV_R2_CI95",
    }
    missing = required.difference(renamed.columns)
    if missing:
        raise ValueError(f"{path} lacks required CV columns: {sorted(missing)}")

    if "FitTime_Mean" not in renamed.columns:
        renamed["FitTime_Mean"] = float("nan")
    if "FitTime_CI95" not in renamed.columns:
        renamed["FitTime_CI95"] = float("nan")

    return renamed


def best_rows(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    return df.sort_values("CV_MSE_Mean", ascending=True).head(n).copy()


def performance_table(
    root: Path,
    allow_quick: bool,
    strict: bool,
    missing_files: list[str],
) -> str | None:
    rows: list[str] = []

    for dataset, filename in DATASETS.items():
        path = discover(root, filename, allow_quick, strict)
        if path is None:
            missing_files.append(filename)
            continue

        df = canonicalize_cv_summary(pd.read_csv(path), path)
        for _, rec in best_rows(df, n=4).iterrows():
            rows.append(
                "{} & {} & {} & {} & {} \\\\".format(
                    latex_escape(dataset),
                    latex_escape(rec["Model"]),
                    pm(rec["CV_MSE_Mean"], rec["CV_MSE_CI95"], 3),
                    pm(rec["CV_R2_Mean"], rec["CV_R2_CI95"], 3),
                    pm(rec["FitTime_Mean"], rec["FitTime_CI95"], 3),
                )
            )

    if not rows:
        return None

    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Repeated nested-cross-validation results. Values are means $\pm$ 95\% confidence-interval half-widths. Only the four lowest-MSE methods per task are shown; complete fold-level results are supplied as CSV files.}}
\label{{tab:round2_main_metrics}}
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{llccc}}
\toprule
Task & Model & MSE & $R^2$ & Fit time (s) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip()


def tradeoff_table(
    root: Path,
    allow_quick: bool,
    strict: bool,
    missing_files: list[str],
) -> str | None:
    rows: list[str] = []

    for dataset, filename in TRADEOFFS.items():
        path = discover(root, filename, allow_quick, strict)
        if path is None:
            missing_files.append(filename)
            continue

        df = pd.read_csv(path)
        required = {
            "Approximation",
            "Landmarks",
            "EffectiveRank",
            "RelativeFrobeniusError",
            "MinEigenvalue",
            "MSE",
            "R2",
            "FitTimeSec",
            "KernelMemoryMB",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")

        for _, rec in df.iterrows():
            approximation = str(rec["Approximation"])
            rank_text = (
                "exact"
                if approximation.lower().startswith("exact")
                else str(int(rec["Landmarks"]))
            )
            rows.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    latex_escape(dataset),
                    latex_escape(approximation),
                    rank_text,
                    int(rec["EffectiveRank"]),
                    fmt(rec["RelativeFrobeniusError"], 3),
                    fmt(rec["MinEigenvalue"], 2),
                    fmt(rec["MSE"], 3),
                    fmt(rec["R2"], 3),
                    fmt(rec["FitTimeSec"], 4)
                    + " / "
                    + fmt(rec["KernelMemoryMB"], 3),
                )
            )

    if not rows:
        return None

    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Nystr\"om rank--accuracy--cost analysis. The last column reports fit time in seconds and stored feature/kernel representation in MiB. Small negative minimum eigenvalues at floating-point roundoff are interpreted numerically as zero.}}
\label{{tab:round2_nystrom_tradeoff}}
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{llccccccc}}
\toprule
Task & Approximation & $m$ & Eff. rank & Rel. Gram error & $\lambda_{{\min}}$ & MSE & $R^2$ & Time / memory \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip()


def scalability_table(
    root: Path,
    allow_quick: bool,
    strict: bool,
    missing_files: list[str],
) -> str | None:
    rows: list[str] = []

    for dataset, filename in SCALABILITY.items():
        path = discover(root, filename, allow_quick, strict)
        if path is None:
            missing_files.append(filename)
            continue

        df = pd.read_csv(path)
        required = {
            "N",
            "Method",
            "FeatureTimeSec",
            "KernelTimeSec",
            "MemoryMB",
            "StoredValues",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")

        for _, rec in df.iterrows():
            rows.append(
                "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                    latex_escape(dataset),
                    int(rec["N"]),
                    latex_escape(rec["Method"]),
                    fmt(rec["FeatureTimeSec"], 4),
                    fmt(rec["KernelTimeSec"], 4),
                    fmt(rec["MemoryMB"], 3),
                    int(rec["StoredValues"]),
                )
            )

    if not rows:
        return None

    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Kernel-representation scalability. Exact dense Gram storage grows as $O(n^2)$, whereas an explicit Nystr\"om representation stores $O(nm)$ values for fixed landmark count $m$. Timings are implementation- and hardware-dependent and document the measured tradeoff rather than machine-independent constants.}}
\label{{tab:round2_scalability}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrllrrr}}
\toprule
Task & $n$ & Method & Feature time (s) & Construction time (s) & Memory (MiB) & Stored values \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip()


def missing_results_notice(missing_files: list[str]) -> str:
    unique = list(dict.fromkeys(missing_files))
    if not unique:
        return ""

    escaped = ", ".join(r"\texttt{" + latex_escape(name) + "}" for name in unique)
    return rf"""
% WARNING: one or more final benchmark CSV files were not found.
\begin{{center}}
\fbox{{\parbox{{0.93\linewidth}}{{\textbf{{Incomplete benchmark table set.}}
The following full-run CSV files were not found beneath the selected results root:
{escaped}.
No numerical values have been invented. Run the corresponding full benchmark scripts and regenerate this file before submission.}}}}
\end{{center}}
""".strip()


def quick_results_notice() -> str:
    return (
        "% WARNING: GENERATED WITH --allow-quick. DO NOT SUBMIT THESE NUMBERS.\n"
        "\\begin{center}\\fbox{\\parbox{0.93\\linewidth}{\\textbf{Formatting demonstration only:} "
        "quick/smoke results may be included and must be replaced by full benchmark outputs before submission.}}\\end{center}"
    )


def standalone_document(content: str) -> str:
    return rf"""\documentclass[a4paper,10pt]{{article}}
\usepackage[margin=1.5cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\begin{{document}}
{content}
\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=SCRIPT_DIR,
        help=(
            "Root directory searched recursively for benchmark CSVs "
            "(default: directory containing this script)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manuscript-input .tex file (default: RESULTS_ROOT/round2_generated_tables.tex)",
    )
    parser.add_argument(
        "--standalone-output",
        type=Path,
        default=None,
        help=(
            "Independently compilable .tex file "
            "(default: RESULTS_ROOT/round2_generated_tables_standalone.tex)"
        ),
    )
    parser.add_argument(
        "--allow-quick",
        action="store_true",
        help="Allow quick/smoke folders for formatting tests only",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow incomplete result sets for local formatting tests only",
    )
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Results root is not a directory: {root}")

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "round2_generated_tables.tex"
    )
    standalone_output = (
        args.standalone_output.expanduser().resolve()
        if args.standalone_output is not None
        else root / "round2_generated_tables_standalone.tex"
    )

    missing_files: list[str] = []
    sections = [
        performance_table(root, args.allow_quick, not args.allow_incomplete, missing_files),
        tradeoff_table(root, args.allow_quick, not args.allow_incomplete, missing_files),
        scalability_table(root, args.allow_quick, not args.allow_incomplete, missing_files),
    ]
    sections = [section for section in sections if section]

    if not args.allow_incomplete and missing_files:
        missing_text = "\n  ".join(dict.fromkeys(missing_files))
        raise FileNotFoundError(
            "Required full-run benchmark CSV files are missing:\n  " + missing_text
        )

    notices: list[str] = []
    if args.allow_quick:
        notices.append(quick_results_notice())

    if not sections:
        print(
            "[WARNING] No benchmark CSV files were found. "
            "A LaTeX warning document will be generated without fabricated values."
        )

    content_parts = notices + sections
    content = "\n\n".join(content_parts).strip() + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Results root: {root}")
    print(f"Wrote manuscript input: {output}")

    standalone_output.parent.mkdir(parents=True, exist_ok=True)
    standalone_output.write_text(standalone_document(content), encoding="utf-8")
    print(f"Wrote standalone document: {standalone_output}")

    if missing_files:
        print("Missing final CSV files:")
        for filename in dict.fromkeys(missing_files):
            print(f"  - {filename}")
        print("Incomplete mode was explicitly enabled; do not use this output for submission.")
    else:
        print("All expected full-result CSV files were found.")


if __name__ == "__main__":
    main()
