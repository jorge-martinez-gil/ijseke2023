"""Tests for the pure Python Linear GP implementation."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = PROJECT_ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from lgp import Instruction, LGPProgram, LinearGPRegressor, protected_division  # noqa: E402


class LgpProgramTests(unittest.TestCase):
    def test_protected_division_preserves_numerator_for_zero_denominator(self):
        result = protected_division(np.array([4.0, 6.0]), np.array([2.0, 0.0]))

        np.testing.assert_allclose(result, np.array([2.0, 6.0]))

    def test_program_executes_register_instructions(self):
        X = np.array([[2.0, 3.0], [4.0, 0.0]])
        program = LGPProgram(
            instructions=(
                Instruction(target=2, operator="add", left=0, right=1),
                Instruction(target=0, operator="mul", left=2, right=1),
            )
        )

        y_pred = program.predict(X, register_count=4)

        np.testing.assert_allclose(y_pred, np.array([15.0, 0.0]))


class LinearGPRegressorTests(unittest.TestCase):
    def test_predict_requires_fit(self):
        model = LinearGPRegressor(population_size=2, generations=1)

        with self.assertRaisesRegex(ValueError, "not fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_fit_returns_finite_predictions(self):
        X = np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ]
        )
        y = np.array([0.0, 1.0, 2.0, 3.0])

        model = LinearGPRegressor(
            population_size=8,
            generations=3,
            min_program_length=2,
            max_program_length=5,
            register_count=6,
            random_state=0,
        ).fit(X, y)

        y_pred = model.predict(X)

        self.assertEqual(y_pred.shape, y.shape)
        self.assertTrue(np.isfinite(y_pred).all())
        self.assertTrue(model.program_lines()[-1].startswith("output: r0 = "))


if __name__ == "__main__":
    unittest.main()
