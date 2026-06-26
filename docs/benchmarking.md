# Benchmarking guide

The `bench/` package turns this repository into a small, reproducible
benchmarking platform for semantic similarity ensembles. This guide explains how
to run it and how to extend it.

## Running

```bash
python -m bench                 # all available methods, all datasets
python -m bench --quick         # fast configuration (good for CI / smoke tests)
python -m bench --help          # all options
```

Outputs are written to `results/`:

| File | Purpose |
|:---|:---|
| `RESULTS.md` | Human-readable tables with 95% bootstrap CIs, paired significance tests, and evolved expressions. |
| `results.csv` | Tidy machine-readable results (one row per method × dataset). |
| `results.tex` | `booktabs` LaTeX table for papers. |
| `figures/*.png`, `figures/*.pdf` | Publication-quality figures. |

Every artifact carries a provenance header (timestamp, git commit, seed, library
versions). Runs are deterministic for a fixed `--seed`.

## Design

```text
datasets.py  ->  registry of benchmark datasets (key, metadata, loader)
methods.py   ->  registry of method adapters (uniform fit_predict interface)
metrics.py   ->  Pearson/Spearman + vectorised bootstrap CIs + paired tests
runner.py    ->  fits every method on every dataset -> BenchmarkResult
report.py    ->  BenchmarkResult -> Markdown / CSV / LaTeX
figures.py   ->  BenchmarkResult -> matplotlib figures
__main__.py  ->  CLI glue
```

The runner is side-effect free: it returns a structured `BenchmarkResult` that
reporting and plotting consume, so tables, numbers, and figures stay in lockstep.

## Statistics

- **Confidence intervals** are 95% percentile bootstrap intervals over resampled
  pairs (`--n-boot`, default 10000), computed in a fully vectorised way.
- **Significance** vs. the baseline uses a *paired* bootstrap test: the same
  resampled indices are applied to both predictors on each iteration, and the
  two-sided p-value is the proportion of resamples whose difference flips sign.
- On benchmarks with 30–50 pairs the intervals are wide by necessity; do not
  over-read small single-dataset gaps.

## Adding a dataset

1. Add `datasets/<name>-training.txt` and `datasets/<name>-validation.txt` with
   the same column layout as the existing files (`a,b,c,d,e,f` for training;
   `truth,bert-cos,…` for validation).
2. Register a `DatasetSpec` in `bench/datasets.py`.

That is all — the new dataset is picked up by `python -m bench`.

## Adding a method

Implement a `fit_predict(X_train, y_train, X_test, metric, seed, quick)` function
that returns `(y_pred, info)` and register a `MethodAdapter` in `bench/methods.py`
(see the LR/LGP examples). Optional dependencies should be detected with
`_module_available("pkg")` so the method is skipped gracefully when absent.

## Sign convention

Some learners optimise the *absolute* correlation, so an evolved feature can come
out anti-correlated with similarity. The harness sign-aligns predictions using
the **training** signal only (`_orient`), so "higher = more similar" holds for
every method without ever looking at test labels.
