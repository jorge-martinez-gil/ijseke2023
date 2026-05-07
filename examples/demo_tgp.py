"""Quick-start example for Tree Genetic Programming."""

from pathlib import Path
import sys

from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "methods") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "methods"))

from utils import load_dataset, pearson_score, spearman_score  # noqa: E402


def main():
    """Train a reduced-grid TGP model on MC and print compact metrics."""
    X_train, y_train = load_dataset(PROJECT_ROOT, "mc", "training")
    X_test, y_test = load_dataset(PROJECT_ROOT, "mc", "validation")

    quick_grid = {
        "population_size": [100],
        "generations": [50],
        "stopping_criteria": [0.01],
        "p_crossover": [0.75],
        "p_subtree_mutation": [0.1],
        "p_hoist_mutation": [0.05],
        "p_point_mutation": [0.1],
        "max_samples": [0.9],
        "parsimony_coefficient": [0.01],
    }

    estimator = SymbolicRegressor(metric="pearson", random_state=0)
    search = GridSearchCV(estimator=estimator, param_grid=quick_grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.fit(X_train, y_train).predict(X_test)

    print("\nTree GP Demo (MC-30, quick grid)")
    print("-" * 40)
    print(f"{'Metric':<12} | {'Score':>10}")
    print("-" * 40)
    print(f"{'Pearson':<12} | {pearson_score(y_test, y_pred):>10.6f}")
    print(f"{'Spearman':<12} | {spearman_score(y_test, y_pred):>10.6f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
