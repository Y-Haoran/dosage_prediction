# Same-Episode `S. aureus` Model Comparison

This comparison keeps the cohort, split, and feature set fixed.

- cohort: primary urgent/emergency same-episode first Gram-positive alerts
- rows: `3877`
- unique patients: `3588`
- positives: `1021` (`26.3%`)
- features: `19` pruned enriched features

## Held-Out Test Results

| Model | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.820 | 0.707 | 0.604 | 0.502 | 0.757 | 0.739 | 0.155 |
| random_forest | 0.814 | 0.713 | 0.619 | 0.569 | 0.678 | 0.780 | 0.136 |
| elastic_net_logistic | 0.812 | 0.672 | 0.617 | 0.604 | 0.632 | 0.794 | 0.171 |
| logistic_regression | 0.812 | 0.672 | 0.615 | 0.600 | 0.632 | 0.792 | 0.171 |

## Interpretation

- Logistic Regression remains a strong transparent baseline.
- Elastic Net shows whether a sparser linear model can keep the signal.
- Random Forest checks whether a simpler bagged tree model is enough.
- XGBoost remains the strongest nonlinear benchmark in this repo.
