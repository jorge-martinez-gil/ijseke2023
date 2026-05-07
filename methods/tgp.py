# -*- coding: utf-8 -*-
"""Tree Genetic Programming for semantic similarity aggregation.

Jorge Martinez-Gil: A Comparative Study of Ensemble Techniques Based on
Genetic Programming: A Case Study in Semantic Similarity Assessment.
Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)
"""

import argparse
from pathlib import Path

import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import GridSearchCV

from utils import load_dataset, pearson_score, spearman_score


def parse_args():
    """Parse command-line options for dataset and optimization metric."""
    parser = argparse.ArgumentParser(description="Run Tree GP with grid search.")
    parser.add_argument(
        "--dataset",
        choices=["mc", "geresid"],
        default="mc",
        help="Dataset to use for training/validation splits.",
    )
    parser.add_argument(
        "--metric",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Fitness metric used by SymbolicRegressor.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for GridSearchCV output.",
    )
    return parser.parse_args()


def main():
    """Train TGP with grid search and report both correlation metrics."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    # Load train/validation splits from project-root datasets.
    X_train, y_train = load_dataset(project_root, args.dataset, "training")
    X_test, y_test = load_dataset(project_root, args.dataset, "validation")

    # Search a broad hyperparameter grid used in the original experiments.
    param_grid = {
        "population_size": [100, 500, 1000],
        "generations": [5000, 9000, 12000],
        "stopping_criteria": [0.01, 0.001],
        "p_crossover": [0.75, 0.9],
        "p_subtree_mutation": [0.1, 0.2],
        "p_hoist_mutation": [0.05, 0.1],
        "p_point_mutation": [0.1, 0.2],
        "max_samples": [0.9, 0.95],
        "parsimony_coefficient": [0.01, 0.001],
    }

    estimator = SymbolicRegressor(metric=args.metric, random_state=0)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        verbose=args.verbose,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    print("Best Hyperparameters:")
    print(grid_search.best_params_)

    # Refit best model on full training split, then evaluate on validation split.
    best_estimator.fit(X_train, y_train)
    y_pred = best_estimator.predict(X_test)

    df_preds = pd.DataFrame({"Actual": y_test.squeeze(), "Predicted": y_pred.squeeze()})
    print(df_preds)

    print(f"Pearson Correlation: {pearson_score(y_test, y_pred):.6f}")
    print(f"Spearman Correlation: {spearman_score(y_test, y_pred):.6f}")


if __name__ == "__main__":
    main()
