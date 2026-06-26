# Benchmark Results (auto-generated)

> **Reproduce:** `python -m bench` regenerates this file from the data in `datasets/`. Every number below is computed at run time — none are hard-coded. Confidence intervals are 95% percentile bootstrap intervals.

> **Evaluation protocol.** The shipped MC-30 and GeReSiD splits cover the *same* pairs for fitting and scoring (the validation gold scores match the training targets), so these correlations measure goodness-of-fit on the benchmark pairs, consistent with the original study — not generalisation to unseen pairs. Use `--cv` for held-out k-fold evaluation on a single split.

- Generated: 2026-06-26 04:05:04Z
- Git commit: `59230ca`
- Seed: 0 · Bootstrap resamples: 10000 · Metric optimised: pearson
- Environment: Python 3.10.12, NumPy 2.2.6
- Baseline for paired tests: LR

## MC-30

*General word similarity — Miller & Charles (1991).* 30 noun pairs with human similarity judgements, a classic general-language word-similarity benchmark.

| Method | Paradigm | Pearson r | 95% CI | Spearman ρ | 95% CI | Δr vs LR | sig. |
|:---|:---|---:|:---:|---:|:---:|---:|:---:|
| LR (baseline) | Linear regression | 0.757 | [0.580, 0.892] | 0.757 | [0.542, 0.898] | — |  |
| LGP | Linear Genetic Programming | 0.751 | [0.548, 0.895] | 0.718 | [0.498, 0.865] | -0.006 | ns |
| TGP | Tree Genetic Programming | 0.750 | [0.593, 0.867] | 0.763 | [0.547, 0.906] | -0.008 | ns |
| CGP | Cartesian Genetic Programming | _skipped_ | | | | | _requires the optional 'tengp' package_ |

<details><summary>Evolved / fitted aggregation functions</summary>

- **LR (baseline):** `-0.256 +2.352*x0, -0.355*x1, +2.736*x2, -3.897*x3, +1.694*x4`
- **LGP:** `(sin(x2) + (x2 / 3))`
- **TGP:** `sub(add(div(sub(sub(X1, add(X1, X2)), div(add(X2, X0), X3)), X4), X2), add(0.683, add(sub(mul(X1, X1), X2), mul(div(X1, div(X4, X1)), X3))))`

</details>

## GeReSiD

*Biomedical term similarity — Garla & Brandt (2012).* Biomedical term pairs with reference similarity scores, used to test cross-domain robustness of the ensembles.

| Method | Paradigm | Pearson r | 95% CI | Spearman ρ | 95% CI | Δr vs LR | sig. |
|:---|:---|---:|:---:|---:|:---:|---:|:---:|
| LR (baseline) | Linear regression | 0.736 | [0.624, 0.825] | 0.744 | [0.630, 0.835] | — |  |
| LGP | Linear Genetic Programming | 0.735 | [0.617, 0.826] | 0.742 | [0.625, 0.835] | -0.002 | ns |
| TGP | Tree Genetic Programming | 0.739 | [0.617, 0.834] | 0.743 | [0.629, 0.835] | 0.003 | ns |
| CGP | Cartesian Genetic Programming | _skipped_ | | | | | _requires the optional 'tengp' package_ |

<details><summary>Evolved / fitted aggregation functions</summary>

- **LR (baseline):** `-0.714 -1.362*x0, -0.436*x1, +1.035*x2, +1.652*x3, +0.864*x4`
- **LGP:** `(((((-1 + ((x3 + 6) * (x3 + 6))) + cos(cos((((x3 + 6) * (x3 + 6)) + (x2 / (-2 + sin(x2))))))) * 7) * (((-1 + ((x3 + 6) * (x3 + 6))) + cos(cos((((x3 + 6) * (x3 + 6)) + (x2 / (-2 + sin(x2))))))) * 7)) + (x2 / ((1 + 3) / sin(-1))))`
- **TGP:** `div(sub(mul(X4, X2), add(X2, X0)), div(add(X4, X1), mul(X3, X3)))`

</details>

---

Significance markers compare each method's Pearson r against the LR baseline via a paired bootstrap test: `*** p<0.001`, `** p<0.01`, `* p<0.05`, `ns` = not significant. Small benchmarks (n=30–50) yield wide intervals — interpret single-dataset differences with care.
