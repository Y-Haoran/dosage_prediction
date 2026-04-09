# Early Prioritization of Same-Episode `Staphylococcus aureus` at the First Gram-Positive Blood-Culture Alert

In `3,877` urgent/emergency single-organism first-alert admissions, a pruned `19`-feature model using pre-alert EHR plus microbiology-history features achieved AUROC `0.820` and AUPRC `0.707` with `XGBoost`. The main finding is that early `S. aureus` prioritization is driven more by microbiology-process context and prior organism history than by physiology alone.

## Quick Read

Question  
At the first Gram-positive blood-culture alert, can pre-alert EHR and microbiology-history features identify which same-episode alerts will later finalize as `Staphylococcus aureus`?

Best current cohort  
Urgent/emergency, single-organism, same-episode first Gram-positive alerts; `3,877` admissions, `1,021` `S. aureus` positives (`26.3%`).

Recommended current model  
Pruned `19`-feature `XGBoost`: AUROC `0.820`, AUPRC `0.707`, F1 `0.604`.

Main scientific message  
Performance improved sharply only after adding microbiology-aware process/history features, which shows this task is not mainly a generic physiology problem.

## Clinical Vignette

One patient arrives unwell. A blood culture is drawn and later flags Gram-positive. At that moment the species is not yet finalized. This model estimates how likely that same blood-culture episode is to later finalize as `S. aureus`, so the team can prioritize urgent review, repeat cultures, source evaluation, and line or device assessment.

## Cohort Flow

```text
All blood cultures
-> first Gram-positive blood-culture alert per admission
-> exclude polymicrobial first-alert episodes
-> keep urgent/emergency admissions
-> tie the label to the same microbiology episode
-> final analytic cohort: 3,877 rows, 1,021 S. aureus positives
```

## Official Current Result

This is the result readers should treat as the current best version of the secondary project.

- cohort: urgent/emergency, single-organism, same-episode first Gram-positive alerts
- features: pruned `19`-feature set
- official current model: `XGBoost`
- best ranking result: AUROC `0.820`, AUPRC `0.707`, F1 `0.604`

Useful companion baselines on the same cohort and same `19` features:

- `Random Forest`: AUROC `0.814`, AUPRC `0.713`, F1 `0.619`
- `Elastic Net Logistic`: AUROC `0.812`, AUPRC `0.672`, F1 `0.617`
- `Logistic Regression`: AUROC `0.812`, AUPRC `0.672`, F1 `0.615`

Interpretation:

- `XGBoost` is the best ranking model
- `Random Forest` gives the strongest AUPRC and F1
- `Elastic Net` is the cleanest compact linear benchmark

## What One Row Means

One row means:

- one hospital admission
- one first Gram-positive blood-culture alert
- one same-episode final species label

This is not:

- one row per bottle
- one row per specimen forever
- one row per patient forever

## Final `19` Features

### Microbiology History

- `prior_subject_s_aureus_positive_90d`
- `prior_subject_s_aureus_positive_365d`
- `prior_subject_cons_positive_90d`
- `prior_subject_cons_positive_365d`
- `prior_subject_any_staph_positive_365d`
- `prior_positive_specimens_7d`

### Blood-Culture Process

- `index_hours_draw_to_alert`
- `prealert_blood_culture_draws_7d`
- `prealert_blood_culture_draws_24h`
- `prealert_blood_culture_draws_6h`

### Host Context And Acuity

- `anchor_age`
- `in_icu_at_alert`
- `vital_map_count_24h`
- `vital_temperature_c_max_24h`

### Pre-Alert Labs

- `lab_platelets_last_24h`
- `lab_creatinine_last_24h`
- `lab_creatinine_count_24h`
- `lab_lactate_max_24h`
- `lab_wbc_last_24h`

Important note:

- no antibiotic feature survived into the final pruned `19`-feature set
- the retained feature groups are dominated by microbiology history, microbiology process, and host acuity

## Key Findings

- the same-episode cohort is scientifically cleaner than the older broad admission-level `S. aureus` task because the alert and the final species label belong to the same microbiology pathway
- generic physiology alone was not enough; the big performance jump came only after adding draw-to-alert time, blood-culture process counts, and prior staphylococcal history
- the signal stayed strong after reducing from `54` to `19` features, which makes the model easier to explain clinically
- the strongest features are clinically coherent rather than arbitrary: prior `S. aureus` history, faster positivity, repeated pre-alert blood-culture drawing, platelets, creatinine, and acuity
- the result is robust across several model families, not only one algorithm

## What This Is

- a same-episode early prioritization model for `S. aureus` risk after the first Gram-positive blood-culture alert
- a pre-alert risk score built from routine EHR data plus microbiology-process and microbiology-history features
- a tool to support earlier review, repeat cultures, source search, and device assessment

## What This Is Not

- not a general sepsis predictor
- not a contaminant classifier
- not a replacement for final speciation
- not an automatic treatment recommendation engine

## Development History

The final result came from three clear stages:

1. Physiology-only same-episode baseline  
   Logistic Regression AUROC `0.666`, XGBoost AUROC `0.640`

2. Enriched microbiology-aware model  
   Logistic Regression AUROC `0.807`, XGBoost AUROC `0.817`

3. Pruned `19`-feature model  
   Logistic Regression AUROC `0.812`, XGBoost AUROC `0.820`

This development path is useful scientifically because it shows where the signal actually comes from:

- not mainly from generic vitals and routine labs
- mainly from microbiology-process context and prior organism history, with supportive host-state information

## Current Limitations

- this remains a secondary analysis, not the main project in the repo
- device and source-aware features are still missing
- prior MRSA-specific history is not yet separated cleanly
- the model supports prioritization, not definitive action

## Next Feature Priorities

- central-line and device information
- prior MRSA history where recoverable safely
- infection-source clues
- richer microbiology context
- note-derived features only if timing can be made clearly leakage-safe

## Recommended Files

- current findings: [reports/s_aureus_same_episode_key_findings.md](../../reports/s_aureus_same_episode_key_findings.md)
- model comparison: [reports/s_aureus_same_episode_model_comparison.md](../../reports/s_aureus_same_episode_model_comparison.md)
- feature reduction: [reports/s_aureus_same_episode_feature_reduction_report.md](../../reports/s_aureus_same_episode_feature_reduction_report.md)
- enriched metrics: [reports/s_aureus_same_episode_enriched_metrics.json](../../reports/s_aureus_same_episode_enriched_metrics.json)
- pruned metrics: [reports/s_aureus_same_episode_pruned_metrics.json](../../reports/s_aureus_same_episode_pruned_metrics.json)
- comparison metrics: [reports/s_aureus_same_episode_model_comparison.json](../../reports/s_aureus_same_episode_model_comparison.json)
- enriched trainer: [scripts/train_s_aureus_same_episode_enriched.py](../../scripts/train_s_aureus_same_episode_enriched.py)
- pruned trainer: [scripts/train_s_aureus_same_episode_pruned.py](../../scripts/train_s_aureus_same_episode_pruned.py)
- comparison trainer: [scripts/train_s_aureus_same_episode_model_comparison.py](../../scripts/train_s_aureus_same_episode_model_comparison.py)
