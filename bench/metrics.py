"""Correlation metrics with bootstrap confidence intervals and paired tests.

All functions operate on 1-D arrays of equal length and depend only on NumPy
(SciPy is used for Spearman when available, with a NumPy fallback so the
harness never hard-fails on a missing optional dependency).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def pearson_score(y_true, y_pred) -> float:
    """Pearson correlation coefficient r between two 1-D arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rankdata(values) -> np.ndarray:
    """Average ranks for a 1-D array (matches scipy.stats.rankdata ties)."""
    values = np.asarray(values, dtype=float).ravel()
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    unique_starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = unique_starts.cumsum() - 1
    counts = np.bincount(dense)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    average = (starts + cumulative - 1) / 2.0 + 1.0
    ranks = np.empty(values.shape, dtype=float)
    ranks[sorter] = average[dense]
    return ranks


def spearman_score(y_true, y_pred) -> float:
    """Spearman rank correlation rho between two 1-D arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    try:
        import scipy.stats

        rho, _ = scipy.stats.spearmanr(y_true, y_pred)
        return float(rho)
    except Exception:
        return pearson_score(_rankdata(y_true), _rankdata(y_pred))


METRICS = {"pearson": pearson_score, "spearman": spearman_score}


# --------------------------------------------------------------------------- #
# Vectorised kernel used by the bootstrap (B resamples at once). The scalar
# ``pearson_score``/``spearman_score`` above remain the canonical point-estimate
# implementations; Spearman is reduced to Pearson-of-ranks via ``_prepare`` so a
# single kernel covers both metrics consistently.
# --------------------------------------------------------------------------- #
def _pearson_rows(YT: np.ndarray, YP: np.ndarray) -> np.ndarray:
    a = YT - YT.mean(axis=1, keepdims=True)
    b = YP - YP.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _prepare(metric: Callable, y_true, y_pred):
    """Return arrays whose row-wise Pearson reproduces ``metric`` exactly.

    For Spearman this rank-transforms both inputs *once* with average ranks, so
    the bootstrap (Pearson of resampled ranks) uses the same definition as the
    scalar point estimate. For Pearson the inputs are returned unchanged. The
    single shared kernel is therefore always :func:`_pearson_rows`.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if metric is spearman_score:
        return _rankdata(y_true), _rankdata(y_pred)
    return y_true, y_pred


def _resample_matrix(y_true, *preds, n_boot, seed):
    y_true = np.asarray(y_true, dtype=float).ravel()
    preds = [np.asarray(p, dtype=float).ravel() for p in preds]
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    YT = y_true[idx]
    YPs = [p[idx] for p in preds]
    return YT, YPs


def bootstrap_ci(
    metric: Callable,
    y_true,
    y_pred,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a correlation metric.

    Returns ``(point_estimate, ci_low, ci_high)``. Pairs ``(y_true_i,
    y_pred_i)`` are resampled with replacement ``n_boot`` times. Fully
    vectorised over resamples for speed.
    """
    point = metric(y_true, y_pred)
    yt, yp = _prepare(metric, y_true, y_pred)
    YT, (YP,) = _resample_matrix(yt, yp, n_boot=n_boot, seed=seed)
    stats = _pearson_rows(YT, YP)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_diff(
    metric: Callable,
    y_true,
    y_pred_a,
    y_pred_b,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test for the difference metric(a) - metric(b).

    The same resampled indices are applied to both predictors on every
    iteration, which controls for sampling noise shared by the two methods.
    Returns the observed difference, a confidence interval, and a two-sided
    bootstrap p-value obtained by percentile inversion (twice the proportion of
    resamples whose difference falls on the opposite side of zero from the
    observed difference, clipped to 1). This is consistent with the reported
    percentile intervals; it is not a recentred null-hypothesis test.
    """
    observed = metric(y_true, y_pred_a) - metric(y_true, y_pred_b)
    yt, ya = _prepare(metric, y_true, y_pred_a)
    _, yb = _prepare(metric, y_true, y_pred_b)
    YT, (YA, YB) = _resample_matrix(yt, ya, yb, n_boot=n_boot, seed=seed)
    diffs = _pearson_rows(YT, YA) - _pearson_rows(YT, YB)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"diff": observed, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    if observed >= 0:
        p = 2.0 * float(np.mean(diffs <= 0))
    else:
        p = 2.0 * float(np.mean(diffs >= 0))
    p = min(1.0, p)
    return {"diff": float(observed), "ci_low": lo, "ci_high": hi, "p_value": p}
