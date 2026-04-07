# Baseline Results For Gram-Positive Blood-Culture Labels

## What was trained

Two baseline models were trained on the high-confidence binary subset:

- Logistic Regression
- XGBoost

Target:

- `0` = `likely_contaminant`
- `1` = `likely_true_bsi`

The `indeterminate` group was excluded from training and evaluation.

## Cohort used for baseline training

- Total high-confidence binary rows: `3,251`
- Train rows: `2,269`
- Validation rows: `490`
- Test rows: `492`
- Train positives: `759`
- Validation positives: `147`
- Test positives: `160`

Subject-level split:

- Train subjects: `2,152`
- Validation subjects: `461`
- Test subjects: `462`

## Two feature settings were evaluated

### 1. Full features

This includes:

- alert context
- demographics
- prior culture history
- pre-alert labs
- pre-alert antibiotics
- organism family features

### 2. No-organism features

This excludes organism-family indicators and is a more conservative test.

That matters because the current provisional label is partly defined using organism type. So the no-organism setting is a better estimate of how much signal comes from the rest of the EHR rather than from the label heuristic itself.

## Test-set results

### Full features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.991` | `0.982` | `1.000` | `0.994` | `0.9999` | `0.9999` | `0.0030` |
| XGBoost | `1.000` | `1.000` | `1.000` | `1.000` | `1.0000` | `1.0000` | `0.0005` |

Confusion counts:

- Logistic Regression: `TP=160`, `TN=329`, `FP=3`, `FN=0`
- XGBoost: `TP=160`, `TN=332`, `FP=0`, `FN=0`

Interpretation:

These scores are almost certainly too optimistic for scientific interpretation, because organism-family features overlap strongly with how the current provisional labels were defined.

### No-organism features

| Model | F1 | Precision | Recall | Accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.545` | `0.514` | `0.581` | `0.685` | `0.723` | `0.570` | `0.210` |
| XGBoost | `0.648` | `0.658` | `0.638` | `0.774` | `0.837` | `0.737` | `0.160` |

Confusion counts:

- Logistic Regression: `TP=93`, `TN=244`, `FP=88`, `FN=67`
- XGBoost: `TP=102`, `TN=279`, `FP=53`, `FN=58`

Interpretation:

This is the more useful first result.

Even without organism-family features, there is still real predictive signal in the pre-alert EHR, and XGBoost clearly outperforms logistic regression on this first baseline.

## Validation-threshold selection

The probability threshold was chosen on the validation split by maximizing validation F1, then applied unchanged to the test split.

Chosen thresholds:

- Full-feature logistic regression: `0.22`
- Full-feature XGBoost: `0.68`
- No-organism logistic regression: `0.52`
- No-organism XGBoost: `0.59`

## What these results mean

The main conclusion is not that the task is solved.

The safer conclusion is:

- the provisional label set is learnable
- some of the strongest signal comes from organism identity
- but even after removing organism-family features, the task still has meaningful signal
- XGBoost is the stronger first baseline

## Important caution

These are baseline results against a provisional label, not a clinician-validated gold standard.

So these numbers should be read as:

- a pipeline feasibility result
- an early benchmark
- not a final clinical-performance claim

## Files

- Metrics JSON: [reports/blood_culture_baseline_metrics.json](reports/blood_culture_baseline_metrics.json)
- Split counts: [reports/blood_culture_baseline_split_counts.json](reports/blood_culture_baseline_split_counts.json)
- Feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- Trainer: [scripts/train_blood_culture_baselines.py](scripts/train_blood_culture_baselines.py)
