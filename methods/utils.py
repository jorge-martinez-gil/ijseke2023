"""utils.py — Shared utility functions for GP ensemble experiments.

Jorge Martinez-Gil (2023). A Comparative Study of Ensemble Techniques Based
on Genetic Programming. IJSEKE 33(2): 289-312.
"""

from pathlib import Path

import numpy as np
import scipy.stats


FEATURE_COLS = {
    "training": ["b", "c", "d", "e", "f"],
    "validation": ["bert-cos", "bert-inn", "bert-man", "bert-euc", "bert-ang"],
}

LABEL_COL = {
    "training": "a",
    "validation": "truth",
}


def pearson_score(y_true, y_pred):
    """Return Pearson r between y_true and y_pred (1-D arrays)."""
    return float(np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1])


def spearman_score(y_true, y_pred):
    """Return Spearman rho between y_true and y_pred (1-D arrays)."""
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
    X = raw[FEATURE_COLS[split]].to_numpy()
    y = raw[LABEL_COL[split]].to_numpy()
    return X, y
