# Interpretable Machine Learning and Causal Inference for Maize LAI Estimation from UAV Multimodal Imagery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IJoRS](https://img.shields.io/badge/Journal-IJoRS-green.svg)](https://www.tandfonline.com/toc/tres20/current)

This repository contains the official implementation for the paper published in **International Journal of Remote Sensing (IJoRS)**:

> **"Interpretable Machine Learning and Causal Inference for Maize LAI Estimation from UAV Multimodal Imagery"**

## Overview

This project presents an interpretable machine learning framework integrating **TabPFN** (Tabular Prior-data Fitted Network) with **Double Machine Learning (DML)** for causal inference to estimate maize Leaf Area Index (LAI) from UAV-based multimodal imagery (spectral, structural, and textural features). The framework includes:

- Multi-model comparison (RF, XGBoost, LightGBM, TabPFN)
- Mutual information-based feature selection
- TabPFN-DML causal inference for genetic background effects
- IJoRS-compliant visualization and reporting

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                           # Data directory (user-provided)
│   └── YOUR_DATA.csv
│
├── models/                         # Saved deployment models
│   └── final_deployment_model.pkl
│
├── results/                        # Output directory
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── src/                            # Source code
    ├── repeated_experiment_training.py          # 20-round repeated experiments
    ├── model_selection_and_persistence.py       # Model comparison & selection
    ├── tabpfn_dml_causal_inference.py           # DML causal inference (main)
    ├── tabpfn_dml_gbk_analysis.py               # GBK genetic background analysis
    ├── multi_temporal_lai_prediction.py         # Multi-temporal prediction
    ├── observed_vs_predicted_scatter.py         # Observed vs predicted scatter
    ├── lai_temporal_dynamics_visualization.py   # Temporal dynamics visualization
    └── lai_dynamics_with_significance.py        # Dynamics with significance testing
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA (optional, for GPU acceleration)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `tabpfn` >= 2.0
- `econml` >= 0.14
- `xgboost` >= 1.7
- `lightgbm` >= 3.3
- `optuna` >= 3.0
- `scikit-learn` >= 1.2
- `pandas` >= 1.5
- `numpy` >= 1.23
- `matplotlib` >= 3.6
- `seaborn` >= 0.12
- `scipy` >= 1.10
- `statsmodels` >= 0.14
- `scikit-posthocs` >= 0.7

### TabPFN Setup

Register at [TabPFN](https://tabpfn.ai/) to obtain an API token:

```python
import os
os.environ["TABPFN_TOKEN"] = "YOUR_TABPFN_TOKEN_HERE"
```

## Usage

### 1. Repeated Experiments (20 Rounds)

Run 20 independent experiments with group-based train-test splits:

```bash
python src/repeated_experiment_training.py
```

Outputs:
- `results_repeat{0-19}.pkl` — per-round model results
- `final_summary_with_CI.csv` — aggregated metrics with 95% CI
- `group_error_analysis.csv` — spatial/temporal error analysis

### 2. Model Selection & Persistence

Compare models and save the optimal deployment model:

```bash
python src/model_selection_and_persistence.py
```

Outputs:
- `final_deployment_model.pkl` — serialized best model
- `model_comprehensive_score_ranking.csv` — ranking table
- `model_paired_ttest.csv` — statistical significance tests

### 3. Causal Inference (TabPFN-DML)

Estimate causal effects of genetic backgrounds on LAI:

```bash
python src/tabpfn_dml_causal_inference.py
```

Outputs:
- `Table2_DML_Causal_Effects.csv` — ATE, CATE, p-values
- `Table3_Cross_Fitting_ATEs.csv` — cross-fitting stability
- `dml_causal_effects_report.txt` — full academic report
- `Fig1_Residual_Distribution.{png,eps,pdf}` — residual analysis
- `Fig2_First_Stage_Performance.{png,eps,pdf}` — model performance
- `Fig6_Heterogeneous_Effects_Forest.{png,eps,pdf}` — forest plot
- `Fig11_GBK_Causal_Effects.{png,eps,pdf}` — combined ATE figure

### 4. Multi-Temporal Prediction

Generate date-based predictions for all samples:

```bash
python src/multi_temporal_lai_prediction.py
```

Outputs:
- `LAI_Prediction_FULL_{date}.csv` — per-date full-sample predictions
- `date_based_accuracy_metrics_FULL.csv` — accuracy summary
- `FigureX_TabPFN_LAI_Prediction_Accuracy.{png,eps,pdf}` — SCI figure

### 5. Visualization

#### Observed vs Predicted Scatter
```bash
python src/observed_vs_predicted_scatter.py
```

#### Temporal Dynamics (Basic)
```bash
python src/lai_temporal_dynamics_visualization.py
```

#### Temporal Dynamics with Significance Testing
```bash
python src/lai_dynamics_with_significance.py
```

Outputs IJoRS-compliant figures in PNG/EPS/PDF (300 DPI).

## Data Format

Input CSV should contain the following columns:

| Column | Description |
|--------|-------------|
| `LAI` | Target variable (Leaf Area Index) |
| `date` | Observation date (YYYYMMDD or YYYY-MM-DD) |
| `label` | Sample identifier / group label |
| `GBK_DH`, `GBK_Mixed`, `GBK_TEM`, `GBK_TST` | Genetic background dummy variables |
| `VI_*` | Vegetation indices (spectral features) |
| `PCD_*`, `CVI`, `CV_Height`, `CRR`, `H_top10_avg` | Structural features |
| `Tex_*` | Texture features (GLCM-based) |

## Key Features

### IJoRS Journal Compliance

All figures strictly follow IJoRS formatting standards:
- **Font**: Arial (sans-serif)
- **Sizes**: Single column 8.5 cm, double column 17.5 cm
- **Resolution**: 300 DPI for raster, vector formats (EPS/PDF)
- **Typography**: Label 8 pt, tick 7 pt, title 9 pt
- **Color palette**: Colorblind-friendly academic colors

### Causal Inference Pipeline

```
Data → Feature Engineering → TabPFN First Stage → DML Estimation
                                    ↓
                         LinearDML (ATE + CI)
                         CausalForestDML (CATE + Heterogeneity)
                                    ↓
                         Cross-fitting Stability Assessment
```

### Model Comparison Metrics

| Metric | Weight |
|--------|--------|
| R² | 40% |
| RMSE | 25% |
| MAE | 25% |
| Stability (R² std) | 10% |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{maize_lai_tabpfn_2026,
  title={Interpretable Machine Learning and Causal Inference for Maize LAI Estimation from UAV Multimodal Imagery},
  journal={International Journal of Remote Sensing},
  year={2026},
  publisher={Taylor & Francis}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TabPFN](https://github.com/automl/TabPFN) — Prior-data Fitted Networks for tabular data
- [EconML](https://github.com/microsoft/EconML) — Microsoft causal inference library
- [Optuna](https://optuna.org/) — Hyperparameter optimization framework

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author.

---

**Note**: Replace all `YOUR_*_PATH` placeholders in the source code with your actual local paths before running. The `TABPFN_TOKEN` environment variable must be set with a valid API key.
