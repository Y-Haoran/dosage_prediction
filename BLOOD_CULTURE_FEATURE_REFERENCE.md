# Blood-Culture Feature Reference

## Main Point

The main baseline in this repo is the **clean 41-feature first-alert model**.

These features are **defined by us** from raw MIMIC-IV tables. They are not a built-in MIMIC-IV feature table.

They are extracted from the **24 hours before the first Gram-positive alert time**, plus a few alert-level context fields.

## Main Files

- feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- feature CSV: `artifacts/blood_culture/first_gp_alert_features.csv`
- clean baseline metrics: [reports/blood_culture_primary_feature_metrics.json](reports/blood_culture_primary_feature_metrics.json)
- synthetic schema example: [examples/blood_culture_feature_example_synthetic.csv](examples/blood_culture_feature_example_synthetic.csv)

## Training Target

For the main binary baseline:

- `target_true_bsi = 0` means `probable_contaminant_or_low_significance_alert`
- `target_true_bsi = 1` means `probable_clinically_significant_bsi_alert`
- `target_true_bsi = NaN` means `indeterminate`

Only the high-confidence binary subset is used for training.

## Main 41 Features

### 1. Basic

- `anchor_age`

### 2. Care setting and acuity

- `in_icu_at_alert`
- `vasopressor_event_count_24h`
- `vasopressor_active_24h`
- `vasopressor_on_at_alert`
- `mechanical_ventilation_chart_events_24h`
- `mechanical_ventilation_24h`

### 3. Prior microbiology

- `prior_positive_specimens_24h`
- `prior_positive_specimens_7d`
- `prior_gp_positive_specimens_24h`
- `prior_gp_positive_specimens_7d`

### 4. WBC

- `lab_wbc_last_24h`
- `lab_wbc_min_24h`
- `lab_wbc_max_24h`
- `lab_wbc_count_24h`

### 5. Platelets

- `lab_platelets_last_24h`
- `lab_platelets_min_24h`
- `lab_platelets_count_24h`

### 6. Creatinine

- `lab_creatinine_last_24h`
- `lab_creatinine_max_24h`
- `lab_creatinine_count_24h`

### 7. Lactate

- `lab_lactate_last_24h`
- `lab_lactate_max_24h`
- `lab_lactate_count_24h`

### 8. Heart rate

- `vital_heart_rate_last_24h`
- `vital_heart_rate_min_24h`
- `vital_heart_rate_max_24h`
- `vital_heart_rate_count_24h`

### 9. Respiratory rate

- `vital_resp_rate_last_24h`
- `vital_resp_rate_max_24h`
- `vital_resp_rate_count_24h`

### 10. Temperature

- `vital_temperature_c_last_24h`
- `vital_temperature_c_min_24h`
- `vital_temperature_c_max_24h`
- `vital_temperature_c_count_24h`

### 11. MAP

- `vital_map_last_24h`
- `vital_map_min_24h`
- `vital_map_count_24h`

### 12. SpO2

- `vital_spo2_last_24h`
- `vital_spo2_min_24h`
- `vital_spo2_count_24h`

## Where These Come From

Typical source tables:

- age and demographics: `hosp/patients.csv`, `hosp/admissions.csv`
- prior blood-culture history: `hosp/microbiologyevents.csv`
- labs: `hosp/labevents.csv`
- ICU stay flags: `icu/icustays.csv.gz`
- vitals and support proxies: `icu/chartevents.csv.gz`

## What Is Not Used

The main baseline does **not** use:

- organism-family indicator columns
- repeat-culture columns
- post-alert antibiotic columns
- identifiers
- label columns

That is why this 41-feature set is the clean baseline for the repo.

## Other Feature Sets

There is also a larger exploratory feature space in the project, but it should be treated as secondary. The main repo story is the 41-feature first-alert baseline.
