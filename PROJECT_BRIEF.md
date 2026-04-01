# Project Brief

## Working Title

Machine learning for 48-hour review of empiric broad-spectrum antibiotics in ICU patients using MIMIC-IV

## Main Aim

Develop a machine-learning model in MIMIC-IV that identifies ICU patients whose empiric broad-spectrum antibiotic therapy is likely no longer justified at 48 hours.

## One Objective

Using only information available up to 48 hours after empiric broad-spectrum antibiotic initiation, predict whether the patient is likely to require continued broad-spectrum coverage or is a high-priority candidate for de-escalation review by 72 hours.

## Why This Is A Good Project

This is a clean stewardship decision point.

It is not:

- "what antibiotic should I start right now?"
- "was the empiric choice correct at time zero?"

It is:

- "at 48 hours, should we still be continuing broad-spectrum therapy?"

That is more clinically actionable and closer to stewardship workflow.

## First Label Definition

Use a simple first label.

Review point:

- 48 hours after empiric broad-spectrum antibiotic start

Outcome window:

- up to 72 hours after start

Class definition:

- positive: broad-spectrum therapy continues past the 48-hour review point
- negative: therapy is narrowed, stopped, or switched off broad-spectrum by 72 hours

## Core Data Available In MIMIC-IV

Antibiotic administration:

- `hosp/emar.csv`
- `hosp/emar_detail.csv`

Medication orders:

- `hosp/pharmacy.csv`
- `hosp/prescriptions.csv`
- `hosp/poe.csv`

Microbiology:

- `hosp/microbiologyevents.csv`

Clinical response:

- `icu/chartevents.csv.gz`
- `icu/inputevents.csv.gz`
- `icu/outputevents.csv.gz`
- `hosp/labevents.csv`

Patient context:

- `hosp/admissions.csv`
- `hosp/patients.csv`
- `icu/icustays.csv.gz`

## Candidate Inputs By 48 Hours

- current antibiotic regimen
- antibiotic administration history
- dose / route / frequency where available
- culture timing and any available microbiology results
- susceptibility information where available
- vitals trends
- lab trends
- organ support markers
- ICU unit and admission context

## Model Output

The safest initial output is a review recommendation:

- broad-spectrum likely still justified
- high-priority candidate for de-escalation review

This is better than telling clinicians to stop a drug automatically.

## Recommended Modeling Plan

Baselines:

- logistic regression
- random forest
- XGBoost

Main model:

- patient-specific decay transformer

Why the transformer may help:

- ICU data are irregular
- evidence arrives over time
- missingness matters
- old observations should decay in importance

## Clinician Input Needed

Doctor or antimicrobial pharmacist input is important for:

- defining the broad-spectrum drug list
- defining what counts as de-escalation
- checking edge cases
- making sure the output is clinically usable

## What Is Already Implemented In This Repo

- generic ICU cohort pipeline
- time-series feature extraction scaffold
- baseline training code
- transformer training code

## What Is Not Yet Implemented

- antibiotic-specific cohort extraction
- 48-hour stewardship label logic
- clinician-reviewed case definitions
- final stewardship evaluation

## Honest Status

This repo is best described as:

- a working MIMIC-IV modeling scaffold
- now being refocused into a 48-hour antibiotic stewardship project
