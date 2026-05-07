# Contributing

Thanks for your interest in improving this repository.

## Report Bugs or Request Features

Please open a GitHub Issue with:
- a clear title,
- steps to reproduce (for bugs), and
- expected vs. observed behavior.

## Submit a Pull Request

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make focused commits with clear messages.
4. Open a Pull Request describing motivation, changes, and validation.

## Code Style

- Follow PEP 8 for Python code.
- Add clear module/function docstrings.
- Type hints are encouraged for new code.

## Adding a New GP Method

- Place new implementations in `methods/`.
- Reuse utilities from `methods/utils.py` for dataset loading and metrics.
- Expose a CLI with `--dataset` and `--metric` for reproducibility.
- Add a small runnable demo under `examples/` when possible.
