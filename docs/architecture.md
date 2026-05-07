# Architecture

## Experimental Pipeline

The project follows a common pipeline for all methods:

1. Load benchmark pairs from `datasets/`.
2. Use precomputed semantic features as model inputs.
3. Train an ensemble model (LR, LGP, TGP, or CGP).
4. Evaluate predictions against gold similarity scores.

## Method Relationship Diagram

```text
                +------------------+
                |  Shared Datasets |
                |  (MC-30, GERESID)|
                +---------+--------+
                          |
          +---------------+---------------+
          |                               |
  +-------v-------+               +-------v-------+
  |  LR Baseline  |               | GP Ensembles  |
  |  (Python)     |               | (LGP/TGP/CGP) |
  +-------+-------+               +-------+-------+
          |                               |
          +---------------+---------------+
                          |
                  +-------v-------+
                  |  Evaluation    |
                  | Pearson/Rho    |
                  +---------------+
```

## Evaluation Protocol

- Each method uses the same training/validation split per dataset.
- Training files provide input similarity features and a target score.
- Validation files provide BERT-based features with reference truth values.
- Final quality is reported with Pearson and Spearman correlations.
