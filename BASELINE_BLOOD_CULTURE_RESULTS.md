# Baseline Results For Blood-Culture Alert Classification

## What Was Trained

Two baseline models were trained on the current clinically meaningful label set:

- Logistic Regression
- XGBoost

Target:

- `0` = `probable_contaminant_or_low_significance_alert`
- `1` = `probable_clinically_significant_bsi_alert`

The `indeterminate` group was excluded from training and evaluation.

## Cohort Used For Baseline Training

- total high-confidence binary rows: `2,506`
- train rows: `1,755`
- validation rows: `379`
- test rows: `372`
- train positives: `859`
- validation positives: `196`
- test positives: `191`

Subject-level split:

- train subjects: `1,658`
- validation subjects: `355`
- test subjects: `356`

## Two Feature Settings Were Evaluated

### 1. Full features

This includes:

- demographics and alert context
- prior culture history
- pre-alert labs
- pre-alert ICU vital summaries
- pre-alert antibiotics
- pre-alert ICU support proxies
- organism-family indicators

### 2. No-organism features

This is the stricter experiment.

It removes organism-family indicators and keeps only the pre-alert clinical context and physiology.

This is the main result if we want the model to work before formal organism identification is available.

## Test-Set Results

### Full features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.968` | `0.979` | `0.958` | `0.968` | `0.990` | `0.991` | `0.029` |
| XGBoost | `0.982` | `0.984` | `0.979` | `0.981` | `0.998` | `0.998` | `0.021` |

Confusion counts:

- Logistic Regression: `TP=183`, `TN=177`, `FP=4`, `FN=8`
- XGBoost: `TP=187`, `TN=178`, `FP=3`, `FN=4`

Interpretation:

These are upper-bound results. They are useful, but not the cleanest scientific result because organism-family indicators are powerful and may not be available at the earliest alert time.

### No-organism features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.868` | `0.873` | `0.864` | `0.866` | `0.940` | `0.943` | `0.101` |
| XGBoost | `0.864` | `0.880` | `0.848` | `0.863` | `0.944` | `0.949` | `0.093` |

Confusion counts:

- Logistic Regression: `TP=165`, `TN=157`, `FP=24`, `FN=26`
- XGBoost: `TP=162`, `TN=159`, `FP=22`, `FN=29`

Interpretation:

This is the main result.

Even without organism-family identity, the model still performs strongly on the held-out test set. That supports the idea that the pre-alert EHR contains meaningful clinical signal about whether the alert is likely clinically significant.

## Validation Thresholds

The probability threshold was chosen on the validation split by maximizing validation F1, then applied unchanged to the test split.

Chosen thresholds:

- full-feature logistic regression: `0.26`
- full-feature XGBoost: `0.31`
- no-organism logistic regression: `0.41`
- no-organism XGBoost: `0.47`

## What These Results Mean

The safest summary is:

- the new clinical-significance label is learnable
- organism identity helps, as expected
- but strong performance remains even after removing organism-family features
- simple baselines already perform well on this task

One interesting detail:

- XGBoost has slightly better AUROC, AUPRC, and Brier score
- Logistic Regression has slightly better test F1 in the no-organism setting

So there is not a single winner for every metric.

## Important Caution

These are still baseline results against a research label, not a gold-standard manual review label.

So they should be read as:

- a strong feasibility result
- an early benchmark
- not a final clinical deployment claim

## Files

- Metrics JSON: [reports/blood_culture_baseline_metrics.json](reports/blood_culture_baseline_metrics.json)
- Split counts: [reports/blood_culture_baseline_split_counts.json](reports/blood_culture_baseline_split_counts.json)
- Feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- Trainer: [scripts/train_blood_culture_baselines.py](scripts/train_blood_culture_baselines.py)
