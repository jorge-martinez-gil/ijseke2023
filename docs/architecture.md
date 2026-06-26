# Architecture

## Two layers

1. **Standalone method scripts** in `methods/` (`lr.py`, `lgp.py`, `tgp.py`,
   `cgp.py`, sharing `utils.py`). Each has a small CLI and reproduces a single
   method on a single dataset — convenient for teaching and quick experiments.
2. **The benchmark harness** in `bench/`, which evaluates *all* methods on *all*
   datasets and produces reports and figures with a single command
   (`python -m bench`). See [`benchmarking.md`](benchmarking.md).

Both layers read the datasets through the same loader, so results are consistent.

## Experimental pipeline

```text
        +------------------------+
        |   datasets/ (MC, …)    |
        +-----------+------------+
                    | load_split
          +---------v----------+
          |  feature matrix X  |   (each column = one similarity measure)
          |  gold vector y     |
          +---------+----------+
                    |
   +----------------+-----------------+
   |        method adapters           |   LR · LGP · TGP · CGP
   |  fit_predict(X_train,y,X_test)   |
   +----------------+-----------------+
                    | predictions
          +---------v----------+
          |   metrics + CIs    |   Pearson / Spearman, bootstrap, paired tests
          +---------+----------+
                    |
        +-----------v------------+
        | reports + figures      |   Markdown / CSV / LaTeX / PNG / PDF
        +------------------------+
```

## Evaluation protocol

The shipped MC-30 and GeReSiD splits cover the **same pairs** for fitting and
scoring (validation gold scores match the training targets; only the feature
representation differs). Reported correlations therefore measure goodness-of-fit
on the benchmark pairs, consistent with the original study — not generalisation
to unseen pairs. Held-out k-fold evaluation is planned (see the README roadmap).
