"""Benchmark orchestration: fit every method on every dataset and score it.

The runner is deliberately side-effect free — it returns a structured
:class:`BenchmarkResult` that the reporting and figure modules consume. This
separation keeps numbers, tables, and plots in lockstep and makes the harness
easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import platform
import time
from typing import Dict, List, Optional

import numpy as np

from .datasets import DATASETS, load_split
from .methods import METHODS, available_methods
from .metrics import bootstrap_ci, paired_bootstrap_diff, pearson_score, spearman_score


@dataclass
class MethodOutcome:
    """Scores and metadata for one (method, dataset) cell."""

    method: str
    dataset: str
    available: bool
    skipped_reason: str = ""
    pearson: float = float("nan")
    pearson_ci: tuple = (float("nan"), float("nan"))
    spearman: float = float("nan")
    spearman_ci: tuple = (float("nan"), float("nan"))
    # Paired bootstrap comparison against the linear-regression baseline.
    delta_pearson: float = float("nan")
    delta_pearson_ci: tuple = (float("nan"), float("nan"))
    delta_pearson_p: float = float("nan")
    expression: str = ""
    runtime_s: float = float("nan")
    y_pred: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class BenchmarkResult:
    """Everything produced by a benchmark run."""

    outcomes: List[MethodOutcome]
    gold: Dict[str, np.ndarray]
    provenance: Dict[str, str]

    def by_dataset(self, dataset: str) -> List[MethodOutcome]:
        return [o for o in self.outcomes if o.dataset == dataset]

    def datasets(self) -> List[str]:
        seen: List[str] = []
        for o in self.outcomes:
            if o.dataset not in seen:
                seen.append(o.dataset)
        return seen

    def methods(self) -> List[str]:
        seen: List[str] = []
        for o in self.outcomes:
            if o.method not in seen:
                seen.append(o.method)
        return seen


def _git_commit() -> str:
    try:
        import subprocess

        from .datasets import REPO_ROOT

        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_benchmark(
    datasets: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    metric: str = "pearson",
    seed: int = 0,
    quick: bool = False,
    n_boot: int = 10000,
    baseline: str = "lr",
    verbose: bool = True,
) -> BenchmarkResult:
    """Run the full benchmark grid and return a :class:`BenchmarkResult`."""
    datasets = datasets or list(DATASETS.keys())
    methods = methods or [m for m in ["lr", "lgp", "tgp", "cgp"] if m in METHODS]

    outcomes: List[MethodOutcome] = []
    gold: Dict[str, np.ndarray] = {}

    for ds in datasets:
        X_train, y_train = load_split(ds, "training")
        X_test, y_test = load_split(ds, "validation")
        gold[ds] = y_test

        predictions: Dict[str, np.ndarray] = {}

        # First pass: fit and score every method, caching predictions so the
        # paired baseline comparison can run afterwards.
        for m in methods:
            adapter = METHODS[m]
            ok, reason = adapter.is_available()
            if not ok:
                if verbose:
                    print(f"[skip] {m} on {ds}: {reason}")
                outcomes.append(
                    MethodOutcome(method=m, dataset=ds, available=False, skipped_reason=reason)
                )
                continue

            start = time.perf_counter()
            y_pred, info = adapter.fit_predict(X_train, y_train, X_test, metric, seed, quick)
            runtime = time.perf_counter() - start
            predictions[m] = y_pred

            p, p_lo, p_hi = bootstrap_ci(pearson_score, y_test, y_pred, n_boot=n_boot, seed=seed)
            s, s_lo, s_hi = bootstrap_ci(spearman_score, y_test, y_pred, n_boot=n_boot, seed=seed)

            outcomes.append(
                MethodOutcome(
                    method=m,
                    dataset=ds,
                    available=True,
                    pearson=p,
                    pearson_ci=(p_lo, p_hi),
                    spearman=s,
                    spearman_ci=(s_lo, s_hi),
                    expression=info.get("expression", ""),
                    runtime_s=runtime,
                    y_pred=y_pred,
                )
            )
            if verbose:
                print(
                    f"[ok]   {m:4s} on {ds:8s}  Pearson={p:.4f} "
                    f"[{p_lo:.3f}, {p_hi:.3f}]  Spearman={s:.4f}  ({runtime:.2f}s)"
                )

        # Second pass: paired bootstrap comparison vs the baseline.
        if baseline in predictions:
            for o in outcomes:
                if o.dataset != ds or not o.available or o.method == baseline:
                    continue
                if o.method not in predictions:
                    continue
                diff = paired_bootstrap_diff(
                    pearson_score,
                    y_test,
                    predictions[o.method],
                    predictions[baseline],
                    n_boot=n_boot,
                    seed=seed,
                )
                o.delta_pearson = diff["diff"]
                o.delta_pearson_ci = (diff["ci_low"], diff["ci_high"])
                o.delta_pearson_p = diff["p_value"]

    provenance = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "git_commit": _git_commit(),
        "seed": str(seed),
        "metric_optimized": metric,
        "n_boot": str(n_boot),
        "quick": str(quick),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "baseline": baseline,
    }
    return BenchmarkResult(outcomes=outcomes, gold=gold, provenance=provenance)
