# 48-Hour Broad-Spectrum Antibiotic Review In MIMIC-IV

## What Question Are We Tackling?

This project asks one clear clinical question:

At 48 hours after starting empiric broad-spectrum antibiotics in ICU patients, is continued broad-spectrum therapy still justified?

This is meant to support antibiotic stewardship review, not to replace a doctor.

The intended model output is:

- `continue broad-spectrum likely justified`
- `high-priority candidate for de-escalation review`

## Related EDA Reports

This repo now also includes exploratory analyses for adjacent MIMIC-IV clinical prediction questions:

- [EDA_BLOOD_CULTURE_LABEL_VALIDITY.md](EDA_BLOOD_CULTURE_LABEL_VALIDITY.md)
  - asks whether a first Gram-positive positive blood-culture alert is more likely true bloodstream infection or contamination
- [BASELINE_BLOOD_CULTURE_RESULTS.md](BASELINE_BLOOD_CULTURE_RESULTS.md)
  - reports Logistic Regression and XGBoost baseline results for the high-confidence blood-culture subset
- [BLOOD_CULTURE_FEATURE_REFERENCE.md](BLOOD_CULTURE_FEATURE_REFERENCE.md)
  - lists the training features and links to a synthetic example dataset with a few hundred rows
- [EDA_ANTIBIOTIC_PROJECT.md](EDA_ANTIBIOTIC_PROJECT.md)
  - sizes the original 48-hour antibiotic-review cohort

## Why This Question?

This is a cleaner stewardship question than asking whether the very first empiric choice was correct.

Why 48 hours:

- it matches real antibiotic review practice better than 0 hours
- it is different from the older "early empiric appropriateness" framing
- by 48 hours, cultures, vitals, labs, and response to treatment are more informative

## What Data Are Available In MIMIC-IV?

MIMIC-IV has the main pieces needed for a first version of this project:

- antibiotic administration
  - `hosp/emar.csv`
  - `hosp/emar_detail.csv`
- medication orders and scheduling
  - `hosp/pharmacy.csv`
  - `hosp/prescriptions.csv`
  - `hosp/poe.csv`
- microbiology and susceptibility information
  - `hosp/microbiologyevents.csv`
- ICU physiology and organ support
  - `icu/chartevents.csv.gz`
  - `icu/inputevents.csv.gz`
  - `icu/outputevents.csv.gz`
- hospital labs and admission context
  - `hosp/labevents.csv`
  - `hosp/admissions.csv`
  - `hosp/patients.csv`

So the data are available for:

- what antibiotic was given
- when it was given
- whether broad-spectrum therapy was continued
- what cultures showed
- how the patient was responding by 48 hours

## What This Repo Contains Right Now

This repo already has the core modeling scaffold:

- MIMIC-IV cohort building
- time-series feature extraction
- tabular baseline training
- transformer training
- a patient-specific decay transformer for irregular ICU data

This part is already coded and smoke-tested.

## What Has Already Been Achieved

The current codebase already does these things end to end:

- builds a leakage-safe ICU cohort
- creates hourly binned sequence features
- creates tabular summary features
- trains logistic regression, random forest, and XGBoost baselines
- trains a transformer model

So the repo is not empty or only an idea. The pipeline exists.

## What Still Needs To Be Added For The Final Antibiotic Project

The antibiotic-specific part is the next step.

Still to implement:

- define the broad-spectrum antibiotic list
- identify empiric antibiotic start time
- build the 48-hour review snapshot
- define the 72-hour outcome label
- classify `continued` versus `narrowed / stopped / switched off broad-spectrum`
- review edge cases with a doctor or antimicrobial pharmacist

That means:

- the modeling scaffold is built
- the antibiotic stewardship label logic is the main remaining project-specific task

## First Study Design

The clean first version is:

- cohort: ICU patients started on empiric broad-spectrum antibiotics
- input window: data available up to 48 hours after antibiotic start
- output window: what happens by 72 hours

First label:

- positive class: broad-spectrum therapy is continued past the 48-hour review point
- negative class: therapy is narrowed, stopped, or switched off broad-spectrum by 72 hours

## What We Want The Model To Be

This should be a clinician-facing review tool, not an automatic stop order.

The most realistic use is:

- flag patients for de-escalation review
- help stewardship teams prioritize review
- provide decision support alongside clinical judgement

## Why The Modeling Approach Could Be Effective

The repo compares two levels of model:

- simple baselines
- a transformer that handles irregular ICU time series better

The transformer idea is still useful here because antibiotic review depends on time-varying evidence:

- fever trend
- blood pressure trend
- lactate trend
- white cell count trend
- culture results appearing over time
- changing organ support needs

The patient-specific decay module is meant to help because older observations should not have equal importance for every patient.

## Important Limitations

This repo does not yet claim a final clinical model.

Current limitations:

- antibiotic-specific labels are not fully implemented yet
- smoke-test results are only pipeline checks
- there is no clinician-reviewed gold-standard label set yet
- no external validation has been done
- this is a research prototype, not a prescribing tool

## Best Way To Read The Repo

If you are new here, think of the repo in two layers:

1. already built:
   the MIMIC-IV modeling engine
2. next focused step:
   the 48-hour antibiotic stewardship cohort and labels

## Files

- `mimic_iv_project/config.py`
  - paths and base feature setup
- `mimic_iv_project/data_pipeline.py`
  - cohort and dataset construction
- `mimic_iv_project/train_baselines.py`
  - tabular baselines
- `mimic_iv_project/models.py`
  - patient-specific decay transformer
- `mimic_iv_project/train_transformer.py`
  - transformer training
- `PROJECT_BRIEF.md`
  - one-page description of the antibiotic project

## Quick Start

Install dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

Point the code to a MIMIC-IV root directory containing `hosp/` and `icu/`:

```bash
export MIMIC_IV_ROOT=/path/to/mimic_root
```

Run a small smoke build:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.data_pipeline --build-all --max-stays 256 --max-chunks 4 --project-root ./_smoke
```

Train the current baselines:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_baselines --project-root ./_smoke
```

Train the current transformer:

```bash
PYTHONPATH=. python3 -m mimic_iv_project.train_transformer --project-root ./_smoke --epochs 5 --batch-size 32
```

## More Detail

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the project aim, label definition, available MIMIC-IV tables, and next implementation steps.
