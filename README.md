# MIMIC-IV ICU Modeling Project

This subproject turns the local MIMIC-IV copy in `../hosp` and `../icu` into:

- a leakage-safe adult ICU cohort
- hourly binned sequence tensors for transformer models
- tabular summary features for baseline models
- a novelty transformer with patient-specific decay

The design is intentionally conservative:

- Unit of prediction: one `stay_id`
- History window: first `24` ICU hours
- Future label window: next `6` hours
- Primary tasks:
  - `vasopressor_next_6h`
  - `in_hospital_mortality`
  - `long_icu_los`

The novelty model is not "just a transformer on MIMIC-IV". It uses a patient-specific decay mechanism:

- the model learns how quickly old measurements should fade
- decay rates are conditioned on static patient context
- missingness, observation counts, and time-since-last-observation are explicit inputs

This is closer to a physiology-informed sequence model than a generic event encoder.

## Project Layout

- `mimic_iv_project/config.py`
  - paths, feature specs, task setup
- `mimic_iv_project/feature_catalog.py`
  - resolves regex-based feature definitions to local MIMIC item IDs
- `mimic_iv_project/data_pipeline.py`
  - builds cohort, sequence tensors, and tabular features
- `mimic_iv_project/train_baselines.py`
  - logistic regression, random forest, and optional XGBoost
- `mimic_iv_project/models.py`
  - patient-specific decay transformer
- `mimic_iv_project/train_transformer.py`
  - multitask sequence training

## Features

Dynamic signals extracted into hourly bins:

- Vitals from `chartevents`
  - heart rate
  - systolic / diastolic / mean BP
  - respiratory rate
  - temperature
  - SpO2
  - weight
- Labs from `labevents`
  - creatinine
  - BUN
  - sodium
  - potassium
  - chloride
  - bicarbonate
  - glucose
  - lactate
  - WBC
  - hemoglobin
  - platelets
  - bilirubin
- Actions / outputs
  - vasopressor indicator from `inputevents`
  - urine output from `outputevents`

Each hourly bin stores:

- aggregated value
- observation mask
- observation count
- time since last observation

Static features per stay:

- age
- sex
- admission type
- insurance
- race
- first ICU careunit
- diagnosis count

## Outputs

Artifacts are written to `artifacts/`:

- `resolved_catalog.json`
- `cohort.csv`
- `sequence_dataset.npz`
- `sequence_metadata.json`
- `tabular_features.csv`
- `tabular_metadata.json`

## Setup

The local machine already has `numpy`, `pandas`, and `torch`. Baseline tree models need extra packages.

```bash
cd /lustre/scratch127/mave/sge_analysis/team229/ds39/RD/LLaMA-Mesh/patient_AI/mimic_iv_project
python3 -m pip install --user -r requirements.txt
```

## Build The Dataset

Build everything:

```bash
cd /lustre/scratch127/mave/sge_analysis/team229/ds39/RD/LLaMA-Mesh/patient_AI/mimic_iv_project
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all
```

Build a smaller smoke-test sample first:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all --max-stays 256 --max-chunks 4 --project-root ./_smoke
```

## Train Baselines

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_baselines
```

This trains one model per task for:

- logistic regression
- random forest
- XGBoost if installed

## Train The Novelty Transformer

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_transformer --epochs 20 --batch-size 64
```

To train from an alternate artifact directory, for example a smoke build:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_transformer --project-root ./_smoke --epochs 5 --batch-size 32
```

Default setup:

- multitask binary prediction
- static-context-conditioned decay
- transformer encoder over hourly bins
- observation-aware input representation

## Research Framing

If you want to pitch this as a paper, the clean story is:

`Patient-Specific Decay Transformer for ICU Risk and Intervention Forecasting in Irregular MIMIC-IV Time Series`

Baseline claim:

- strong tabular baselines on the same 24h history window

Novelty claim:

- decay is learned per patient from static context rather than fixed globally
- the model uses explicit missingness and time-gap structure instead of pretending the sequence is regular
- evaluation includes both outcome prediction and near-term intervention forecasting

## Practical Notes

- `chartevents` and `labevents` are large, so the pipeline reads them in chunks
- the current code uses the first `24h` and the next `6h` by default to keep the project tractable
- you can expand feature specs in `config.py` once the first run works
