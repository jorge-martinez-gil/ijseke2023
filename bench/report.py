"""Render a :class:`BenchmarkResult` to Markdown, CSV, and LaTeX.

Every artifact carries a provenance header (timestamp, git commit, seed,
library versions) so any number can be traced back to the exact run that
produced it. Nothing here invents values; it only formats what the runner
measured.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .datasets import DATASETS
from .methods import METHODS
from .runner import BenchmarkResult


def _fmt(value: float, nd: int = 3) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:.{nd}f}"


def _ci(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:
        return ""
    return f"[{lo:.3f}, {hi:.3f}]"


def _sig(p: float) -> str:
    if p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def write_csv(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dataset",
                "method",
                "available",
                "pearson",
                "pearson_ci_low",
                "pearson_ci_high",
                "spearman",
                "spearman_ci_low",
                "spearman_ci_high",
                "delta_pearson_vs_baseline",
                "delta_pearson_pvalue",
                "runtime_s",
                "expression",
            ]
        )
        for o in result.outcomes:
            writer.writerow(
                [
                    o.dataset,
                    o.method,
                    o.available,
                    f"{o.pearson:.6f}" if o.available else "",
                    f"{o.pearson_ci[0]:.6f}" if o.available else "",
                    f"{o.pearson_ci[1]:.6f}" if o.available else "",
                    f"{o.spearman:.6f}" if o.available else "",
                    f"{o.spearman_ci[0]:.6f}" if o.available else "",
                    f"{o.spearman_ci[1]:.6f}" if o.available else "",
                    f"{o.delta_pearson:.6f}" if o.delta_pearson == o.delta_pearson else "",
                    f"{o.delta_pearson_p:.6f}" if o.delta_pearson_p == o.delta_pearson_p else "",
                    f"{o.runtime_s:.3f}" if o.available else "",
                    o.expression,
                ]
            )


def _provenance_lines(result: BenchmarkResult) -> List[str]:
    p = result.provenance
    return [
        f"- Generated: {p['generated_utc']}",
        f"- Git commit: `{p['git_commit']}`",
        f"- Seed: {p['seed']} · Bootstrap resamples: {p['n_boot']} · Metric optimised: {p['metric_optimized']}"
        + (" · **quick mode**" if p.get("quick") == "True" else ""),
        f"- Environment: Python {p['python']}, NumPy {p['numpy']}",
        f"- Baseline for paired tests: {p['baseline'].upper()}",
    ]


def write_markdown(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Benchmark Results (auto-generated)")
    lines.append("")
    lines.append(
        "> **Reproduce:** `python -m bench` regenerates this file from the data "
        "in `datasets/`. Every number below is computed at run time — none are "
        "hard-coded. Confidence intervals are 95% percentile bootstrap intervals."
    )
    lines.append("")
    lines.append(
        "> **Evaluation protocol.** The shipped MC-30 and GeReSiD splits cover the "
        "*same* pairs for fitting and scoring (the validation gold scores match the "
        "training targets), so these correlations measure goodness-of-fit on the "
        "benchmark pairs, consistent with the original study \u2014 not generalisation "
        "to unseen pairs. Use `--cv` for held-out k-fold evaluation on a single split."
    )
    lines.append("")
    lines.extend(_provenance_lines(result))
    lines.append("")

    for ds in result.datasets():
        spec = DATASETS.get(ds)
        title = spec.name if spec else ds
        lines.append(f"## {title}")
        if spec:
            lines.append("")
            lines.append(f"*{spec.domain} — {spec.reference}.* {spec.description}")
        lines.append("")
        lines.append(
            "| Method | Paradigm | Pearson r | 95% CI | Spearman ρ | 95% CI | Δr vs LR | sig. |"
        )
        lines.append("|:---|:---|---:|:---:|---:|:---:|---:|:---:|")
        for o in result.by_dataset(ds):
            adapter = METHODS.get(o.method)
            label = adapter.label if adapter else o.method
            paradigm = adapter.paradigm if adapter else ""
            if not o.available:
                lines.append(
                    f"| {label} | {paradigm} | _skipped_ | | | | | _{o.skipped_reason}_ |"
                )
                continue
            delta = "—" if o.method == result.provenance["baseline"] else _fmt(o.delta_pearson)
            sig = "" if o.method == result.provenance["baseline"] else _sig(o.delta_pearson_p)
            lines.append(
                f"| {label} | {paradigm} | {_fmt(o.pearson)} | {_ci(*o.pearson_ci)} "
                f"| {_fmt(o.spearman)} | {_ci(*o.spearman_ci)} | {delta} | {sig} |"
            )
        lines.append("")
        # Interpretable expressions.
        exprs = [o for o in result.by_dataset(ds) if o.available and o.expression]
        if exprs:
            lines.append("<details><summary>Evolved / fitted aggregation functions</summary>")
            lines.append("")
            for o in exprs:
                adapter = METHODS.get(o.method)
                label = adapter.label if adapter else o.method
                lines.append(f"- **{label}:** `{o.expression}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Significance markers compare each method's Pearson r against the LR "
        "baseline via a paired bootstrap test: `*** p<0.001`, `** p<0.01`, "
        "`* p<0.05`, `ns` = not significant. Small benchmarks (n=30–50) yield "
        "wide intervals — interpret single-dataset differences with care."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("% Auto-generated by `python -m bench`. Do not edit by hand.")
    lines.append(f"% Generated {result.provenance['generated_utc']} "
                 f"(commit {result.provenance['git_commit']}, seed {result.provenance['seed']}).")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Semantic similarity ensemble performance (Pearson $r$ and "
        "Spearman $\\rho$ on the benchmark pairs). Values in brackets are "
        "95\\% bootstrap confidence intervals.}"
    )
    lines.append("\\label{tab:benchmark}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & Pearson $r$ & Spearman $\\rho$ \\\\")
    lines.append("\\midrule")
    for ds in result.datasets():
        spec = DATASETS.get(ds)
        title = spec.name if spec else ds
        rows = [o for o in result.by_dataset(ds) if o.available]
        for i, o in enumerate(rows):
            adapter = METHODS.get(o.method)
            label = adapter.label if adapter else o.method
            cell_ds = title if i == 0 else ""
            r = f"{o.pearson:.3f} [{o.pearson_ci[0]:.3f}, {o.pearson_ci[1]:.3f}]"
            rho = f"{o.spearman:.3f} [{o.spearman_ci[0]:.3f}, {o.spearman_ci[1]:.3f}]"
            lines.append(f"{cell_ds} & {label} & {r} & {rho} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_reports(result: BenchmarkResult, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    paths = [
        out_dir / "RESULTS.md",
        out_dir / "results.csv",
        out_dir / "results.tex",
    ]
    write_markdown(result, paths[0])
    write_csv(result, paths[1])
    write_latex(result, paths[2])
    return paths
