"""Focused regression tests for dataset and objective contracts."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = PROJECT_ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from cgp import objective_pearson, protected_division  # noqa: E402
from utils import load_dataset, pearson_score  # noqa: E402


class DatasetContractTests(unittest.TestCase):
    def test_mc_feature_order_matches_training_and_validation_files(self):
        X_train, y_train = load_dataset(PROJECT_ROOT, "mc", "training")
        X_validation, y_validation = load_dataset(PROJECT_ROOT, "mc", "validation")

        self.assertEqual(X_train.shape, (30, 5))
        self.assertEqual(y_train.shape, (30,))
        self.assertEqual(X_validation.shape, (30, 5))
        self.assertEqual(y_validation.shape, (30,))

        np.testing.assert_allclose(
            X_train[0],
            np.array([0.921, 0.642, 0.642, 0.993, 0.872]),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            X_validation[0],
            np.array([0.9212773, 0.64173526, 0.642264677, 0.992636919, 0.8728529]),
            rtol=0,
            atol=1e-12,
        )

    def test_geresid_split_shapes_are_consistent(self):
        X_train, y_train = load_dataset(PROJECT_ROOT, "geresid", "training")
        X_validation, y_validation = load_dataset(PROJECT_ROOT, "geresid", "validation")

        self.assertEqual(X_train.shape, (50, 5))
        self.assertEqual(y_train.shape, (50,))
        self.assertEqual(X_validation.shape, (50, 5))
        self.assertEqual(y_validation.shape, (50,))

    def test_load_dataset_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            datasets_dir = Path(tmpdir) / "datasets"
            datasets_dir.mkdir()
            (datasets_dir / "mc-training.txt").write_text("a,b,c,d,e\n1,2,3,4,5\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing expected columns: f"):
                load_dataset(tmpdir, "mc", "training")


class CgpContractTests(unittest.TestCase):
    def test_protected_division_keeps_numerator_when_denominator_is_zero(self):
        result = protected_division(np.array([4.0, 6.0]), np.array([2.0, 0.0]))

        np.testing.assert_allclose(result, np.array([2.0, 6.0]))

    def test_pearson_objective_is_lower_for_better_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0])

        perfect = objective_pearson(y_true, y_true)
        inverse = objective_pearson(y_true, y_true[::-1])

        self.assertLess(perfect, inverse)


class MetricContractTests(unittest.TestCase):
    def test_pearson_returns_zero_for_constant_predictions(self):
        score = pearson_score(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
