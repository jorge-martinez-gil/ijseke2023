<p align="center">
  <h1 align="center">A Comparative Study of Ensemble Techniques Based on Genetic Programming</h1>
  <p align="center"><em>A Case Study in Semantic Similarity Assessment (IJSEKE 2023)</em></p>
</p>

<p align="center">
  <a href="https://doi.org/10.1142/S0218194022500772"><img src="https://img.shields.io/badge/DOI-10.1142%2FS0218194022500772-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+">
  <a href="https://doi.org/10.1142/S0218194022500772"><img src="https://img.shields.io/badge/Paper-IJSEKE%202023-orange" alt="Paper"></a>
</p>

Code repository for the paper:

> J. Martinez-Gil: *A Comparative Study of Ensemble Techniques Based on Genetic Programming: A Case Study in Semantic Similarity Assessment.* Int. J. Softw. Eng. Knowl. Eng. 33(2): 289–312 (2023). DOI: [10.1142/S0218194022500772](https://doi.org/10.1142/S0218194022500772)

## Abstract

Semantic similarity assessment is a core Natural Language Processing (NLP) task aimed at quantifying how close two text units are in meaning. This repository accompanies a comparative analysis of ensemble methods for that task, focusing on approaches grounded in genetic programming (GP). GP-based ensembles are appealing because they can evolve non-linear combinations of heterogeneous similarity signals while preserving interpretability of the learned aggregation expressions. The experiments compare a linear regression baseline against linear, tree-based, and cartesian GP strategies under a consistent evaluation protocol.

## Repository Structure

```text
.
├── datasets/                    # Benchmark splits for MC-30 and GERESID
├── methods/
│   ├── lr.py                    # Linear Regression baseline
│   ├── tgp.py                   # Tree Genetic Programming (gplearn)
│   ├── cgp.py                   # Cartesian Genetic Programming (tengp)
│   ├── utils.py                 # Shared dataset/metric utilities
│   └── lgp/                     # Java Eclipse project for Linear GP
├── examples/
│   ├── demo_lr.py               # Quick LR example
│   └── demo_tgp.py              # Quick TGP example (reduced grid)
├── docs/
│   └── architecture.md          # Experimental pipeline overview
├── .github/workflows/ci.yml     # CI checks
├── CONTRIBUTING.md              # Contribution guide
├── CHANGELOG.md                 # Release history
├── requirements.txt             # Python dependencies
└── README.md
```

## Methods

| Method | Language | Paradigm | Library |
|---|---|---|---|
| LR (baseline) | Python | Linear regression ensemble | scikit-learn |
| LGP | Java | Linear Genetic Programming | chen0040 GP toolkit |
| TGP | Python | Tree Genetic Programming | gplearn |
| CGP | Python | Cartesian Genetic Programming | tengp |

## Datasets

Two benchmark datasets are included and evaluated with fixed training/validation splits.

| Dataset | Domain | Train pairs | Test pairs |
|---|---:|---:|---:|
| MC-30 | General word similarity | 30 | 30 |
| GERESID | Biomedical term similarity | 600 | 600 |

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/jorge-martinez-gil/ijseke2023.git
   cd ijseke2023
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run methods from the `methods/` folder:
   ```bash
   cd methods
   python lr.py --dataset mc --metric pearson
   python tgp.py --dataset mc --metric pearson
   python cgp.py --dataset mc --metric pearson
   ```

## Results

Representative IJSEKE 2023 results are summarized below (higher is better).

| Method | Pearson (r) | Spearman (ρ) |
|---|---:|---:|
| LR | 0.874 | 0.862 |
| LGP | 0.901 | 0.889 |
| TGP | 0.914 | 0.903 |
| CGP | 0.907 | 0.895 |

## Reproducibility

- Python methods target Python **3.8+**.
- Random seed is fixed where supported (e.g., `random_state=0` in TGP).
- Use `--metric pearson` or `--metric spearman` in each Python method to select the target optimization metric.
- All methods use the same dataset splits in `datasets/` to ensure fair comparisons.

## Citation

If you use this work, please cite:

```bibtex
@article{martinezgil2023c,
  author       = {Jorge Martinez-Gil},
  title        = {A Comparative Study of Ensemble Techniques Based on Genetic Programming: {A} Case Study in Semantic Similarity Assessment},
  journal      = {Int. J. Softw. Eng. Knowl. Eng.},
  volume       = {33},
  number       = {2},
  pages        = {289--312},
  year         = {2023},
  url          = {https://doi.org/10.1142/S0218194022500772},
  doi          = {10.1142/S0218194022500772}
}
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
