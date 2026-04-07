# Blood-Culture Feature Reference

## Why this file exists

People looking at the GitHub repo need to see:

- what the model features actually are
- how the final tabular dataset is structured
- an example table without exposing restricted MIMIC-derived patient rows

So this repo includes:

- this feature reference
- a synthetic example dataset with a few hundred rows

The synthetic file is only for schema illustration. It is **not** real patient data and it should not be used for scientific analysis.

## Files

- Feature builder: [scripts/build_blood_culture_features.py](scripts/build_blood_culture_features.py)
- Synthetic example dataset: [examples/blood_culture_feature_example_synthetic.csv](examples/blood_culture_feature_example_synthetic.csv)
- Real feature metadata: [artifacts/blood_culture/blood_culture_feature_metadata.json](artifacts/blood_culture/blood_culture_feature_metadata.json)

## Training target

For the first binary baseline:

- `target_true_bsi = 0` means `likely_contaminant`
- `target_true_bsi = 1` means `likely_true_bsi`
- `target_true_bsi = NaN` means `indeterminate`

Only the high-confidence binary subset is used for the first baseline model.

## Feature groups

The current tabular baseline uses `91` training features.

### 1. Alert context: 9 features

- `has_storetime`
- `has_charttime`
- `row_count`
- `unique_org_count`
- `in_icu_at_alert`
- `anchor_age`
- `alert_hours_from_admit`
- `alert_weekend`
- `alert_night`

### 2. Organism-family indicators: 11 features

- `org_cons`
- `org_s_aureus`
- `org_enterococcus`
- `org_viridans_strep`
- `org_beta_or_anginosus_strep`
- `org_corynebacterium`
- `org_bacillus`
- `org_cutibacterium_propionibacterium`
- `org_lactobacillus`
- `org_polymicrobial_gp`
- `org_other_gp`

### 3. Prior microbiology history: 5 features

- `prior_positive_specimens_24h`
- `prior_positive_specimens_7d`
- `prior_gp_positive_specimens_24h`
- `prior_gp_positive_specimens_7d`
- `prior_same_organism_positive_7d`

### 4. Lab summaries from the 24h lookback: 35 features

For each lab, the feature set includes:

- `last`
- `min`
- `max`
- `mean`
- `count`

Labs used:

- `wbc`
- `hemoglobin`
- `platelets`
- `creatinine`
- `lactate`
- `sodium`
- `potassium`

### 5. Pre-alert antibiotic exposure in the 24h lookback: 11 features

- `abx_total_admin_24h`
- `abx_total_admin_24h_flag`
- `abx_vancomycin_iv_like_24h`
- `abx_vancomycin_iv_like_24h_flag`
- `abx_linezolid_24h`
- `abx_linezolid_24h_flag`
- `abx_daptomycin_24h`
- `abx_daptomycin_24h_flag`
- `abx_broad_gram_negative_24h`
- `abx_broad_gram_negative_24h_flag`
- `abx_anti_mrsa_24h_flag`

### 6. One-hot encoded demographics and admission context: 20 features

- sex indicators
- admission type indicators
- insurance group indicators
- race group indicators

## Important distinction

The full feature table also contains non-training columns such as:

- `hadm_id`
- `subject_id`
- `micro_specimen_id`
- `alert_time`
- `provisional_label`
- `label_source`

Those columns are useful for analysis and tracking, but they are not part of the training feature matrix.

## What is not used as a predictor

To avoid leakage, the current training feature list excludes:

- `repeat_any_positive_48h`
- `repeat_gp_positive_48h`
- `repeat_same_organism_48h`
- `repeat_positive_specimen_count_48h`
- `provisional_label`
- `label_source`

Those are used to create the provisional labels, so they cannot be fed back into the model.

## How to regenerate the real feature table

```bash
PYTHONPATH=. python3 scripts/build_blood_culture_features.py --project-root . --raw-root /path/to/mimic_root
```

## Why the synthetic example is useful

The synthetic CSV lets other people:

- inspect column names
- understand data types and rough ranges
- test parsing / loading code
- review the schema in GitHub

without publishing restricted MIMIC-derived patient rows.
