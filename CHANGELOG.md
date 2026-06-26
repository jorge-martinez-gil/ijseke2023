## [1.2.0] — 2026-06-26
### Added
- **Reproducible benchmark harness** (`bench/`): one command, `python -m bench`,
  evaluates every method on every dataset and writes Markdown, CSV, and LaTeX
  tables plus publication-quality figures to `results/`.
- Bootstrap 95% confidence intervals and a paired bootstrap significance test
  against the linear baseline.
- `Dockerfile` for a pinned, reproducible environment.
- `docs/benchmarking.md` extension guide and unit tests for the harness.

### Changed
- **README** rewritten around reproducibility and discoverability. The previous
  headline results table was **removed**: those figures could not be reproduced
  from the data in `datasets/` and have been replaced by auto-generated results
  (with confidence intervals) that anyone can regenerate with `python -m bench`.
- Documented the evaluation protocol explicitly: the shipped splits measure
  goodness-of-fit on the same pairs, not generalisation to unseen pairs.
- Tree GP predictions are now sign-aligned on the training signal (gplearn's
  `pearson`/`spearman` fitness optimises absolute correlation).

## [1.1.0] — 2026-05-07
### Changed
- Improved documentation and README.
- Refactored Python sources: added CLI, removed deprecated pandas arguments.
- Added shared `utils.py` module.
- Added CI via GitHub Actions.
- Added `examples/` directory with demo scripts.
- Added CONTRIBUTING.md and CHANGELOG.md.

## [1.0.0] — 2023-02-01
### Added
- Initial public release accompanying IJSEKE 33(2) paper.
- LR, LGP, TGP, CGP implementations.
- MC-30 and GERESID benchmark datasets.
