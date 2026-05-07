# -*- coding: utf-8 -*-
"""Linear Regression baseline for semantic similarity aggregation.

Jorge Martinez-Gil: A Comparative Study of Ensemble Techniques Based on
Genetic Programming: A Case Study in Semantic Similarity Assessment.
Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import load_dataset, pearson_score, spearman_score


def parse_args():
    """Parse command-line options for dataset and target metric."""
    parser = argparse.ArgumentParser(description="Run the Linear Regression baseline.")
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
        help="Primary metric to highlight in output.",
    )
    return parser.parse_args()


def main():
    """Train LR on the selected dataset and report both correlations."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    # Load consistent train/validation splits from the datasets directory.
    X_train, y_train = load_dataset(project_root, args.dataset, "training")
    X_test, y_test = load_dataset(project_root, args.dataset, "validation")

    # Fit baseline linear regression and generate predictions.
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show paired target and prediction values for quick inspection.
    df_preds = pd.DataFrame({"Actual": y_test.squeeze(), "Predicted": y_pred.squeeze()})
    print(df_preds)

    pearson = pearson_score(y_test, y_pred)
    spearman = spearman_score(y_test, y_pred)
    primary = pearson if args.metric == "pearson" else spearman
    secondary_name = "spearman" if args.metric == "pearson" else "pearson"
    secondary = spearman if args.metric == "pearson" else pearson

    print(f"{args.metric.title()} Correlation (primary): {primary:.6f}")
    print(f"{secondary_name.title()} Correlation: {secondary:.6f}")


if __name__ == "__main__":
    main()
