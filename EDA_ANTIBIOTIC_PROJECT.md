# MIMIC-IV EDA For The 48h Antibiotic Review Project

## Project question

At 48 hours after starting empiric broad-spectrum antibiotics in ICU patients, is continued broad-spectrum therapy still justified?

## Tables used in this first pass

- `icu/icustays.csv.gz`
- `hosp/emar.csv`
- `hosp/microbiologyevents.csv`

## Cohort definitions used here

This is only a first-pass EDA. The antibiotic lists are still provisional.

### 1. Provisional broad-spectrum set

This pass matched medication names containing:

- `piperacillin-tazobactam`
- `meropenem`
- `cefepime`
- `ceftazidime`
- `imipenem`
- `aztreonam`
- `ertapenem`
- `vancomycin`
- `linezolid`

### 2. Strict antipseudomonal set

This narrower pass matched:

- `piperacillin-tazobactam`
- `meropenem`
- `cefepime`
- `ceftazidime`
- `imipenem`
- `aztreonam`

## Overall ICU size

- Total ICU stays: `94,458`
- ICU stays with length of stay `>= 48h`: `46,337`
- ICU stays with length of stay `>= 72h`: `31,178`

## Exposure counts

### Provisional broad-spectrum set

- ICU stays with at least one matched administration during ICU: `20,610`
- ICU stays with first matched administration in the first `24h` of ICU: `17,212`
- ICU stays with `48h` ICU follow-up from first administration: `11,730`
- ICU stays with `72h` ICU follow-up from first administration: `8,933`
- Distinct hospital admissions with any matched administration: `64,455`
- Distinct subjects with any matched administration: `40,406`

Derived percentages:

- `21.82%` of all ICU stays had a matched administration during ICU
- `18.22%` of all ICU stays had first matched administration within the first `24h`
- `68.15%` of first-24h ICU starts still had at least `48h` ICU follow-up
- `51.90%` of first-24h ICU starts still had at least `72h` ICU follow-up

Most frequent matched medication names:

- `Vancomycin`
- `CefePIME`
- `Piperacillin-Tazobactam`
- `Vancomycin Oral Liquid`
- `Meropenem`
- `CefTAZidime`
- `Linezolid`
- `Aztreonam`

### Strict antipseudomonal set

- ICU stays with at least one matched administration during ICU: `15,305`
- ICU stays with first matched administration in the first `24h` of ICU: `12,176`
- ICU stays with `48h` ICU follow-up from first administration: `9,585`
- ICU stays with `72h` ICU follow-up from first administration: `7,520`
- Distinct hospital admissions with any matched administration: `42,504`
- Distinct subjects with any matched administration: `27,866`

Derived percentages:

- `16.20%` of all ICU stays had a matched administration during ICU
- `12.89%` of all ICU stays had first matched administration within the first `24h`
- `78.72%` of first-24h ICU starts still had at least `48h` ICU follow-up
- `61.76%` of first-24h ICU starts still had at least `72h` ICU follow-up

Most frequent matched medication names:

- `CefePIME`
- `Piperacillin-Tazobactam`
- `Meropenem`
- `CefTAZidime`
- `Aztreonam`

## Microbiology availability

### Provisional broad-spectrum set

- Admissions with any microbiology event by `48h`: `45,684`
- Admissions with any microbiology event by `72h`: `47,055`
- Admissions with organism named by `72h`: `19,227`
- Admissions with susceptibility interpretation by `72h`: `11,419`

Derived percentages:

- `70.88%` of matched admissions had some microbiology event by `48h`
- `17.72%` of matched admissions had susceptibility interpretation by `72h`

### Strict antipseudomonal set

- Admissions with any microbiology event by `48h`: `32,930`
- Admissions with any microbiology event by `72h`: `33,806`
- Admissions with organism named by `72h`: `14,274`
- Admissions with susceptibility interpretation by `72h`: `8,608`

Derived percentages:

- `77.48%` of matched admissions had some microbiology event by `48h`
- `20.25%` of matched admissions had susceptibility interpretation by `72h`

## What these numbers mean

The project is feasible. Even with a fairly strict ICU-centered definition, there are about `7,520` ICU stays that look eligible for a `48h review -> 72h outcome` design.

That is enough for:

- a first classical baseline model
- a transformer-style sequence model
- subgroup analysis by major antibiotic family or ICU type

## Important caveats from this first pass

The medication matching is not clean enough yet to be the final cohort definition.

Examples:

- `Vancomycin Oral Liquid`
- `Vancomycin Enema`
- `Vancomycin-Heparin Lock`
- `Cefepime Desensitization`

These should probably be excluded from the final stewardship cohort.

So the realistic next step is to replace regex-only matching with:

- a clinician-reviewed antibiotic list
- route filtering
- administration-status filtering
- episode-level logic for empiric starts and de-escalation

## Practical recommendation

For the first real paper version, the cleanest starting cohort is:

- ICU stays only
- first broad-spectrum or antipseudomonal administration within first `24h` of ICU
- at least `72h` ICU follow-up from first administration
- final target built from medication changes between `48h` and `72h`

That should give a usable starting cohort of roughly:

- `7.5k` ICU stays with the strict antipseudomonal definition
- up to about `8.9k` ICU stays with the broader provisional definition

## Output files from this pass

- `/nfs/users/nfs_d/ds39/mimic_antibiotic_eda_summary.json`
