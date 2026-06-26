"""Tests for the reproducible benchmark harness (bench/)."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.metrics import (  # noqa: E402
    bootstrap_ci,
    paired_bootstrap_diff,
    pearson_score,
    spearman_score,
)
from bench.datasets import DATASETS, load_split  # noqa: E402
from bench.runner import run_benchmark  # noqa: E402
from bench.report import write_all_reports  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_pearson_perfect_and_inverse(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson_score(y, y), 1.0, places=12)
        self.assertAlmostEqual(pearson_score(y, -y), -1.0, places=12)

    def test_pearson_zero_for_constant(self):
        self.assertEqual(pearson_score(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])), 0.0)

    def test_spearman_is_rank_based(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        monotone = np.array([1.0, 4.0, 9.0, 16.0])  # increasing but non-linear
        self.assertAlmostEqual(spearman_score(y, monotone), 1.0, places=12)

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=60)
        x = y + rng.normal(scale=0.5, size=60)
        point, lo, hi = bootstrap_ci(pearson_score, y, x, n_boot=2000, seed=0)
        self.assertTrue(lo <= point <= hi)
        self.assertTrue(0.0 < lo < hi < 1.0)

    def test_bootstrap_ci_is_deterministic(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=40)
        x = y + rng.normal(scale=0.5, size=40)
        a = bootstrap_ci(pearson_score, y, x, n_boot=1500, seed=7)
        b = bootstrap_ci(pearson_score, y, x, n_boot=1500, seed=7)
        self.assertEqual(a, b)

    def test_paired_bootstrap_diff_detects_better_predictor(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=80)
        good = y + rng.normal(scale=0.3, size=80)
        bad = rng.normal(size=80)
        res = paired_bootstrap_diff(pearson_score, y, good, bad, n_boot=2000, seed=0)
        self.assertGreater(res["diff"], 0.0)
        self.assertLess(res["p_value"], 0.05)


class DatasetTests(unittest.TestCase):
    def test_known_shapes(self):
        for ds, n in {"mc": 30, "geresid": 50}.items():
            self.assertIn(ds, DATASETS)
            X, y = load_split(ds, "validation")
            self.assertEqual(X.shape, (n, 5))
            self.assertEqual(y.shape, (n,))


class RunnerTests(unittest.TestCase):
    def test_run_and_report_lr_is_deterministic(self):
        kwargs = dict(datasets=["mc"], methods=["lr"], n_boot=500, seed=0, verbose=False)
        r1 = run_benchmark(**kwargs)
        r2 = run_benchmark(**kwargs)
        o1 = r1.by_dataset("mc")[0]
        o2 = r2.by_dataset("mc")[0]
        self.assertTrue(np.isfinite(o1.pearson))
        self.assertEqual(o1.pearson, o2.pearson)
        self.assertEqual(o1.spearman, o2.spearman)

    def test_reports_are_written_and_have_no_hardcoded_legacy_numbers(self):
        result = run_benchmark(datasets=["mc"], methods=["lr"], n_boot=300, seed=0, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_all_reports(result, Path(tmp))
            for p in paths:
                self.assertTrue(p.exists() and p.stat().st_size > 0)
            md = (Path(tmp) / "RESULTS.md").read_text(encoding="utf-8")
            # The retracted, non-reproducible *headline* figures must never
            # reappear. (Only unambiguous values are checked: 0.901/0.907 were
            # also fabricated but can legitimately occur as bootstrap CI bounds.)
            for fabricated in ("0.874", "0.914"):
                self.assertNotIn(fabricated, md)


if __name__ == "__main__":
    unittest.main()
