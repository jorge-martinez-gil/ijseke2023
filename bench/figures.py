"""Publication-quality figures generated from a :class:`BenchmarkResult`.

Three figures, each designed to communicate a specific scientific point rather
than to decorate:

1. ``fig_correlation_bars`` — performance with bootstrap 95% CIs, so readers see
   the *uncertainty*, not just point estimates (critical for tiny benchmarks).
2. ``fig_pred_vs_gold`` — predicted vs. gold scatter with the identity line,
   exposing calibration and where each method fails.
3. ``fig_improvement_forest`` — a forest plot of each method's gain over the
   linear baseline with CIs straddling zero, the honest way to show whether an
   ensemble actually helps.

Matplotlib is imported lazily; if it is not installed the functions return an
empty list instead of raising, keeping the core harness dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .datasets import DATASETS
from .methods import METHODS
from .runner import BenchmarkResult

# A restrained, colour-blind-friendly palette (Okabe–Ito).
_PALETTE = {
    "lr": "#999999",
    "lgp": "#0072B2",
    "tgp": "#D55E00",
    "cgp": "#009E73",
}
_FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 200,
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "-",
                "axes.axisbelow": True,
                "figure.autolayout": False,
            }
        )
        return plt
    except Exception:
        return None


def _color(method: str, index: int) -> str:
    return _PALETTE.get(method, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _label(method: str) -> str:
    adapter = METHODS.get(method)
    return adapter.label if adapter else method


def _ds_title(ds: str) -> str:
    spec = DATASETS.get(ds)
    return spec.name if spec else ds


def _save(fig, out_dir: Path, name: str, plt) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_correlation_bars(result: BenchmarkResult, out_dir: Path, plt) -> List[Path]:
    datasets = result.datasets()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.4), squeeze=False)
    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        outs = [o for o in result.by_dataset(ds) if o.available]
        methods = [o.method for o in outs]
        x = np.arange(len(methods))
        width = 0.38
        for k, (metric, ci_attr, val_attr, offset) in enumerate(
            [("Pearson r", "pearson_ci", "pearson", -width / 2),
             ("Spearman ρ", "spearman_ci", "spearman", width / 2)]
        ):
            vals = np.array([getattr(o, val_attr) for o in outs])
            los = np.array([getattr(o, ci_attr)[0] for o in outs])
            his = np.array([getattr(o, ci_attr)[1] for o in outs])
            yerr = np.vstack([vals - los, his - vals])
            hatch = "" if k == 0 else "//"
            colors = [_color(m, i) for i, m in enumerate(methods)]
            ax.bar(
                x + offset, vals, width, yerr=yerr, capsize=3,
                color=colors, alpha=0.95 if k == 0 else 0.55,
                edgecolor="black", linewidth=0.6, hatch=hatch, label=metric,
                error_kw={"elinewidth": 1.0},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=0)
        ax.set_ylim(0, 1.0)
        ax.set_title(_ds_title(ds), fontweight="bold")
        if col == 0:
            ax.set_ylabel("Correlation with human judgement")
        # One combined legend (solid = Pearson, hatched = Spearman).
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="#777777", edgecolor="black", label="Pearson r"),
                Patch(facecolor="#777777", edgecolor="black", hatch="//", alpha=0.55, label="Spearman ρ"),
            ],
            loc="lower right", frameon=False, fontsize=9,
        )
    fig.suptitle("Ensemble performance with 95% bootstrap confidence intervals", fontweight="bold")
    return _save(fig, out_dir, "fig_correlation_bars", plt)


def fig_pred_vs_gold(result: BenchmarkResult, out_dir: Path, plt) -> List[Path]:
    datasets = result.datasets()
    methods = [m for m in result.methods()
               if any(o.method == m and o.available for o in result.outcomes)]
    nrows, ncols = len(datasets), max(1, len(methods))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows), squeeze=False)
    for r, ds in enumerate(datasets):
        gold = result.gold[ds]
        for c, m in enumerate(methods):
            ax = axes[r][c]
            outs = [o for o in result.by_dataset(ds) if o.method == m and o.available]
            ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", alpha=0.6)
            if outs and outs[0].y_pred is not None:
                o = outs[0]
                yp = np.asarray(o.y_pred, dtype=float)
                # Min-max scale predictions to [0,1] for comparable axes.
                rng = yp.max() - yp.min()
                yp_s = (yp - yp.min()) / rng if rng > 0 else np.zeros_like(yp)
                ax.scatter(gold, yp_s, s=22, color=_color(m, c), edgecolor="black", linewidth=0.4, alpha=0.85)
                ax.text(0.04, 0.92, f"r={o.pearson:.3f}", transform=ax.transAxes, fontsize=9, va="top")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if r == 0:
                ax.set_title(_label(m), fontweight="bold", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{_ds_title(ds)}\npredicted (scaled)", fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("gold similarity", fontsize=9)
    fig.suptitle("Predicted vs. gold similarity (dashed = perfect calibration)", fontweight="bold")
    return _save(fig, out_dir, "fig_pred_vs_gold", plt)


def fig_improvement_forest(result: BenchmarkResult, out_dir: Path, plt) -> List[Path]:
    baseline = result.provenance.get("baseline", "lr")
    rows = []
    for ds in result.datasets():
        for o in result.by_dataset(ds):
            if not o.available or o.method == baseline:
                continue
            if o.delta_pearson != o.delta_pearson:
                continue
            rows.append((ds, o))
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(6.6, 0.7 * len(rows) + 1.6))
    ypos = np.arange(len(rows))[::-1]
    for y, (ds, o) in zip(ypos, rows):
        lo, hi = o.delta_pearson_ci
        color = _color(o.method, 0)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.scatter([o.delta_pearson], [y], color=color, s=55, zorder=3, edgecolor="black", linewidth=0.6)
        ax.text(hi, y + 0.18, f"  {_sig(o.delta_pearson_p)}", va="center", fontsize=9, color="#333333")
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{_label(o.method)} · {_ds_title(ds)}" for ds, o in rows])
    ax.set_xlabel(f"Δ Pearson r vs. {baseline.upper()} baseline (95% CI)")
    ax.set_title("Does the ensemble beat the linear baseline?", fontweight="bold")
    return _save(fig, out_dir, "fig_improvement_forest", plt)


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


def generate_all_figures(result: BenchmarkResult, out_dir: Path) -> List[Path]:
    """Generate every figure; returns [] (with a notice) if matplotlib is absent."""
    plt = _try_matplotlib()
    out_dir = Path(out_dir)
    if plt is None:
        print("[figures] matplotlib not installed — skipping figure generation.")
        return []
    paths: List[Path] = []
    paths += fig_correlation_bars(result, out_dir, plt)
    paths += fig_pred_vs_gold(result, out_dir, plt)
    paths += fig_improvement_forest(result, out_dir, plt)
    return paths
