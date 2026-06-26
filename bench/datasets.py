"""Dataset registry for the benchmark harness.

A single source of truth for the benchmark collections shipped with the
repository. Each entry records human-readable metadata (used in reports) and
delegates the actual file parsing to :func:`methods.utils.load_dataset`, so the
harness and the standalone method scripts always read the data identically.

Adding a new dataset is intentionally trivial: drop ``<name>-training.txt`` and
``<name>-validation.txt`` into ``datasets/`` and add one :class:`DatasetSpec`
entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_METHODS_DIR = REPO_ROOT / "methods"
if str(_METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(_METHODS_DIR))

from utils import FEATURE_COLS, load_dataset  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a benchmark dataset."""

    key: str
    name: str
    domain: str
    reference: str
    description: str

    @property
    def feature_names(self):
        """Feature column names used on the validation split."""
        return list(FEATURE_COLS["validation"])


DATASETS: Dict[str, DatasetSpec] = {
    "mc": DatasetSpec(
        key="mc",
        name="MC-30",
        domain="General word similarity",
        reference="Miller & Charles (1991)",
        description=(
            "30 noun pairs with human similarity judgements, a classic "
            "general-language word-similarity benchmark."
        ),
    ),
    "geresid": DatasetSpec(
        key="geresid",
        name="GeReSiD",
        domain="Biomedical term similarity",
        reference="Garla & Brandt (2012)",
        description=(
            "Biomedical term pairs with reference similarity scores, used to "
            "test cross-domain robustness of the ensembles."
        ),
    ),
}


def load_split(dataset_key: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a ``training`` or ``validation`` split as ``(X, y)`` arrays."""
    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_key}'. Known: {sorted(DATASETS)}")
    X, y = load_dataset(REPO_ROOT, dataset_key, split)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
