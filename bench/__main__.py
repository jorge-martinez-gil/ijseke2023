"""Command-line entry point: ``python -m bench``.

Examples
--------
    python -m bench                          # everything, default settings
    python -m bench --quick                  # fast, CI-friendly run
    python -m bench --datasets mc            # a single dataset
    python -m bench --methods lr lgp         # a subset of methods
    python -m bench --no-figures             # skip plots (no matplotlib needed)
    python -m bench --output results/        # choose the output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import DATASETS, REPO_ROOT
from .figures import generate_all_figures
from .methods import DEFAULT_METHOD_ORDER, METHODS
from .report import write_all_reports
from .runner import run_benchmark


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Reproducible benchmark for semantic similarity ensembles.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=None,
        help="Datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHOD_ORDER, default=None,
        help="Methods to evaluate (default: all available).",
    )
    parser.add_argument(
        "--metric", choices=["pearson", "spearman"], default="pearson",
        help="Metric optimised by the evolutionary methods (default: pearson).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use small, fast method configurations (for CI / smoke tests).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10000,
        help="Bootstrap resamples for confidence intervals (default: 10000).",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "results",
        help="Directory for tables and figures (default: results/).",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output)
    figures_dir = out_dir / "figures"

    if not args.quiet:
        print("=" * 64)
        print(" Semantic Similarity Ensemble Benchmark")
        print("=" * 64)
        avail = [m for m in (args.methods or DEFAULT_METHOD_ORDER) if METHODS[m].is_available()[0]]
        missing = [m for m in (args.methods or DEFAULT_METHOD_ORDER) if not METHODS[m].is_available()[0]]
        print(f" Datasets : {args.datasets or list(DATASETS)}")
        print(f" Methods  : available={avail}  skipped={missing}")
        print(f" Metric   : {args.metric}   Seed: {args.seed}   Quick: {args.quick}")
        print("-" * 64)

    result = run_benchmark(
        datasets=args.datasets,
        methods=args.methods,
        metric=args.metric,
        seed=args.seed,
        quick=args.quick,
        n_boot=args.n_boot,
        verbose=not args.quiet,
    )

    report_paths = write_all_reports(result, out_dir)
    figure_paths = [] if args.no_figures else generate_all_figures(result, figures_dir)

    if not args.quiet:
        print("-" * 64)
        print(" Wrote:")
        for p in report_paths:
            print(f"   {p}")
        for p in figure_paths:
            print(f"   {p}")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
