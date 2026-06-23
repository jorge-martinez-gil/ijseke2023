"""utils.py — Shared utility functions for GP ensemble experiments.

Jorge Martinez-Gil (2023). A Comparative Study of Ensemble Techniques Based
on Genetic Programming. IJSEKE 33(2): 289-312.
"""

from pathlib import Path

import numpy as np


FEATURE_COLS = {
    "training": ["b", "c", "d", "e", "f"],
    "validation": ["bert-cos", "bert-man", "bert-euc", "bert-inn", "bert-ang"],
}

LABEL_COL = {
    "training": "a",
    "validation": "truth",
}


def pearson_score(y_true, y_pred):
    """Return Pearson r between y_true and y_pred (1-D arrays)."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denominator = np.linalg.norm(y_true_centered) * np.linalg.norm(y_pred_centered)
    if denominator == 0:
        return 0.0
    return float(np.dot(y_true_centered, y_pred_centered) / denominator)


def _rankdata(values):
    """Return average ranks for a 1-D array, matching Spearman tie handling."""
    values = np.asarray(values).flatten()
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    unique_starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense_ranks = unique_starts.cumsum() - 1
    counts = np.bincount(dense_ranks)
    cumulative_counts = np.cumsum(counts)
    starts = cumulative_counts - counts
    average_ranks = (starts + cumulative_counts - 1) / 2.0 + 1.0

    ranks = np.empty(values.shape, dtype=float)
    ranks[sorter] = average_ranks[dense_ranks]
    return ranks


def spearman_score(y_true, y_pred):
    """Return Spearman rho between y_true and y_pred (1-D arrays)."""
    try:
        import scipy.stats
    except ImportError:
        return pearson_score(_rankdata(y_true), _rankdata(y_pred))

    rho, _ = scipy.stats.spearmanr(y_true.flatten(), y_pred.flatten())
    return float(rho)


def load_dataset(base_dir, dataset_name, split):
    """Load a training or validation split for the given dataset.

    Parameters
    ----------
    base_dir : str or Path
        Root directory that contains the ``datasets/`` folder.
    dataset_name : str
        One of ``"mc"`` or ``"geresid"``.
    split : str
        One of ``"training"`` or ``"validation"``.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix with shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Label vector with shape ``(n_samples,)``.
    """
    import pandas as pd

    base_path = Path(base_dir)
    path = base_path / "datasets" / f"{dataset_name}-{split}.txt"

    if dataset_name not in {"mc", "geresid"}:
        raise ValueError("dataset_name must be 'mc' or 'geresid'.")
    if split not in {"training", "validation"}:
        raise ValueError("split must be 'training' or 'validation'.")

    raw = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    expected_cols = [LABEL_COL[split], *FEATURE_COLS[split]]
    missing_cols = [column for column in expected_cols if column not in raw.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing expected columns: {', '.join(missing_cols)}")

    X = raw[FEATURE_COLS[split]].to_numpy()
    y = raw[LABEL_COL[split]].to_numpy()
    return X, y
