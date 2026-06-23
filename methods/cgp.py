# -*- coding: utf-8 -*-
"""Cartesian Genetic Programming for semantic similarity aggregation.

Jorge Martinez-Gil: A Comparative Study of Ensemble Techniques Based on
Genetic Programming: A Case Study in Semantic Similarity Assessment.
Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)
"""

import argparse
from pathlib import Path

import numpy as np

from utils import load_dataset, pearson_score, spearman_score


def parse_args():
    """Parse command-line options for dataset and optimization metric."""
    parser = argparse.ArgumentParser(description="Run Cartesian GP with grid search.")
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
        help="Metric to optimize during evolution.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=100,
        help="Verbosity level passed to tengp.simple_es.",
    )
    return parser.parse_args()


def protected_division(x, y):
    """Divide x by y with safe handling for zero denominators."""
    return np.divide(x, y, out=np.copy(x), where=y != 0)


def objective_pearson(y_true, y_pred):
    """Objective compatible with tengp (negative Pearson for minimization)."""
    return -pearson_score(y_true, y_pred)


def objective_spearman(y_true, y_pred):
    """Objective compatible with tengp (negative Spearman for minimization)."""
    return -spearman_score(y_true, y_pred)


def main():
    """Train CGP over a grid and report Pearson/Spearman correlations."""
    import tengp
    from sklearn.model_selection import ParameterGrid

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    X_train, y_train = load_dataset(project_root, args.dataset, "training")
    X_test, y_test = load_dataset(project_root, args.dataset, "validation")

    # Define function set available to evolved CGP expressions.
    funset = tengp.FunctionSet()
    funset.add(np.add, 2)
    funset.add(np.subtract, 2)
    funset.add(np.multiply, 2)
    funset.add(protected_division, 2)
    funset.add(np.sin, 1)
    funset.add(np.cos, 1)

    # Search over architecture-related hyperparameters.
    param_grid = ParameterGrid({"n_columns": [50, 100, 150, 200, 250, 300], "n_rows": [1, 2, 3, 4]})
    objective = objective_pearson if args.metric == "pearson" else objective_spearman

    best_params = None
    best_fitness = float("inf")

    for params_dict in param_grid:
        params = tengp.Parameters(
            n_inputs=X_train.shape[1],
            n_outputs=1,
            function_set=funset,
            n_columns=params_dict["n_columns"],
            n_rows=params_dict["n_rows"],
        )
        result = tengp.simple_es(
            X_train,
            y_train,
            objective,
            params,
            mutation="probabilistic",
            verbose=args.verbose,
        )
        fitness = result[0].fitness
        if fitness < best_fitness:
            best_fitness = fitness
            best_params = params_dict

    print("Best hyperparameters:", best_params)
    print("Best objective value:", best_fitness)

    params = tengp.Parameters(
        n_inputs=X_train.shape[1],
        n_outputs=1,
        function_set=funset,
        **best_params,
    )
    result = tengp.simple_es(
        X_train,
        y_train,
        objective,
        params,
        mutation="probabilistic",
        verbose=args.verbose,
    )

    print(result[0].fitness)
    print(result[0].get_expression())

    y_pred = result[0].transform(X_test)
    print(f"Pearson Correlation: {pearson_score(y_test, y_pred):.6f}")
    print(f"Spearman Correlation: {spearman_score(y_test, y_pred):.6f}")


if __name__ == "__main__":
    main()
