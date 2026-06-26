"""Method registry / adapters for the benchmark harness.

Every ensemble method exposes the same tiny interface so the runner can treat
them uniformly::

    adapter.is_available() -> (bool, reason)
    adapter.fit_predict(X_train, y_train, X_test, metric, seed, quick)
        -> (y_pred, info)

``info`` is a dict that may contain a human-readable ``expression`` (the
evolved aggregation function) for the interpretability column of the report.

Optional heavyweight dependencies (gplearn, tengp) are imported lazily; if they
are missing the method reports itself as unavailable instead of crashing the
whole run.

Sign convention: some learners (notably gplearn with the ``pearson``/
``spearman`` fitness) optimise the *absolute* correlation, so an evolved feature
can come out anti-correlated with similarity. Predictions are therefore
sign-aligned using the *training* signal only (:func:`_orient`) so that "higher
means more similar" holds for every method, without ever peeking at test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_METHODS_DIR = REPO_ROOT / "methods"
if str(_METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(_METHODS_DIR))


@dataclass(frozen=True)
class MethodAdapter:
    """Uniform wrapper around an ensemble learner."""

    key: str
    label: str
    paradigm: str
    library: str
    is_available: Callable[[], Tuple[bool, str]]
    fit_predict: Callable[..., Tuple[np.ndarray, dict]]


def _module_available(module_name: str) -> Callable[[], Tuple[bool, str]]:
    def check() -> Tuple[bool, str]:
        import importlib.util

        if importlib.util.find_spec(module_name) is None:
            return False, f"requires the optional '{module_name}' package"
        return True, ""

    return check


def _always_available() -> Tuple[bool, str]:
    return True, ""


def _orient(train_pred, y_train, y_pred):
    """Sign-align predictions so higher means more similar (train-calibrated)."""
    from .metrics import pearson_score

    if pearson_score(y_train, train_pred) < 0:
        return -np.asarray(y_pred, dtype=float).ravel()
    return np.asarray(y_pred, dtype=float).ravel()


def _protected_division(x, y):
    return np.divide(x, y, out=np.copy(np.asarray(x, dtype=float)), where=np.asarray(y) != 0)


# --------------------------------------------------------------------------- #
# Linear Regression baseline
# --------------------------------------------------------------------------- #
def _fit_predict_lr(X_train, y_train, X_test, metric, seed, quick):
    try:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        coef = ", ".join(f"{c:+.3f}*x{i}" for i, c in enumerate(model.coef_))
        expr = f"{model.intercept_:+.3f} {coef}"
    except Exception:
        A = np.c_[np.ones(len(X_train)), X_train]
        coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        y_pred = np.c_[np.ones(len(X_test)), X_test] @ coef
        terms = ", ".join(f"{c:+.3f}*x{i}" for i, c in enumerate(coef[1:]))
        expr = f"{coef[0]:+.3f} {terms}"
    return np.asarray(y_pred, dtype=float).ravel(), {"expression": expr}


# --------------------------------------------------------------------------- #
# Linear Genetic Programming (pure-Python implementation shipped in methods/)
# --------------------------------------------------------------------------- #
def _fit_predict_lgp(X_train, y_train, X_test, metric, seed, quick):
    from lgp import LinearGPRegressor

    model = LinearGPRegressor(
        metric=metric,
        population_size=30 if quick else 60,
        generations=15 if quick else 40,
        random_state=seed,
    ).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    expr = model.program_lines()[-1].replace("output: r0 = ", "")
    y_pred = _orient(model.predict(X_train), y_train, y_pred)
    return y_pred, {"expression": expr}


# --------------------------------------------------------------------------- #
# Tree Genetic Programming (gplearn, optional)
# --------------------------------------------------------------------------- #
def _fit_predict_tgp(X_train, y_train, X_test, metric, seed, quick):
    from gplearn.genetic import SymbolicRegressor

    model = SymbolicRegressor(
        metric=metric,
        population_size=200 if quick else 1000,
        generations=10 if quick else 30,
        stopping_criteria=0.001,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        parsimony_coefficient=0.001,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test), dtype=float).ravel()
    y_pred = _orient(model.predict(X_train), y_train, y_pred)
    return y_pred, {"expression": str(model._program)}


# --------------------------------------------------------------------------- #
# Cartesian Genetic Programming (tengp, optional)
# --------------------------------------------------------------------------- #
def _fit_predict_cgp(X_train, y_train, X_test, metric, seed, quick):
    import tengp

    from .metrics import pearson_score, spearman_score

    score = pearson_score if metric == "pearson" else spearman_score
    np.random.seed(seed)

    def objective(y_true, y_pred):
        return -score(y_true, y_pred)

    funset = tengp.FunctionSet()
    funset.add(np.add, 2)
    funset.add(np.subtract, 2)
    funset.add(np.multiply, 2)
    funset.add(_protected_division, 2)
    funset.add(np.sin, 1)
    funset.add(np.cos, 1)

    params = tengp.Parameters(
        n_inputs=X_train.shape[1],
        n_outputs=1,
        function_set=funset,
        n_columns=100 if quick else 200,
        n_rows=2,
    )
    result = tengp.simple_es(X_train, y_train, objective, params, mutation="probabilistic", verbose=0)
    individual = result[0]
    y_pred = np.asarray(individual.transform(X_test), dtype=float).ravel()
    y_pred = _orient(individual.transform(X_train), y_train, y_pred)
    try:
        expr = str(individual.get_expression())
    except Exception:
        expr = "(CGP graph)"
    return y_pred, {"expression": expr}


METHODS: Dict[str, MethodAdapter] = {
    "lr": MethodAdapter(
        key="lr",
        label="LR (baseline)",
        paradigm="Linear regression",
        library="scikit-learn / NumPy",
        is_available=_always_available,
        fit_predict=_fit_predict_lr,
    ),
    "lgp": MethodAdapter(
        key="lgp",
        label="LGP",
        paradigm="Linear Genetic Programming",
        library="pure Python",
        is_available=_always_available,
        fit_predict=_fit_predict_lgp,
    ),
    "tgp": MethodAdapter(
        key="tgp",
        label="TGP",
        paradigm="Tree Genetic Programming",
        library="gplearn",
        is_available=_module_available("gplearn"),
        fit_predict=_fit_predict_tgp,
    ),
    "cgp": MethodAdapter(
        key="cgp",
        label="CGP",
        paradigm="Cartesian Genetic Programming",
        library="tengp",
        is_available=_module_available("tengp"),
        fit_predict=_fit_predict_cgp,
    ),
}

DEFAULT_METHOD_ORDER: List[str] = ["lr", "lgp", "tgp", "cgp"]


def available_methods() -> List[str]:
    """Return the keys of methods whose dependencies are importable."""
    return [k for k in DEFAULT_METHOD_ORDER if METHODS[k].is_available()[0]]
