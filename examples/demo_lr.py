"""Quick-start example for the Linear Regression baseline."""

from pathlib import Path
import sys

from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "methods") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "methods"))

from utils import load_dataset, pearson_score, spearman_score  # noqa: E402


def main():
    """Train LR on MC and print a compact result table."""
    X_train, y_train = load_dataset(PROJECT_ROOT, "mc", "training")
    X_test, y_test = load_dataset(PROJECT_ROOT, "mc", "validation")

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nLinear Regression Demo (MC-30)")
    print("-" * 36)
    print(f"{'Metric':<12} | {'Score':>10}")
    print("-" * 36)
    print(f"{'Pearson':<12} | {pearson_score(y_test, y_pred):>10.6f}")
    print(f"{'Spearman':<12} | {spearman_score(y_test, y_pred):>10.6f}")
    print("-" * 36)


if __name__ == "__main__":
    main()
