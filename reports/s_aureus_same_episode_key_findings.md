# Key Findings: Same-Episode `S. aureus` Prediction

## Why This Result Matters

This secondary task asks a clinically practical question:

> At the first Gram-positive blood-culture alert, which patients are more likely to have that same blood-culture episode later finalize as `Staphylococcus aureus`?

The value of the task is not automatic treatment escalation. The value is earlier prioritization:

- faster senior review
- faster repeat blood-culture planning
- faster source search
- faster line or device assessment
- faster recognition that the patient may need a more serious `S. aureus` workup

## Main Positive Finding

The strongest result is that the model improved substantially once we added clinically targeted microbiology-process and prior-staphylococcal-history features.

Primary urgent / emergency same-episode cohort:

- `41`-feature baseline
  - Logistic Regression: AUROC `0.666`, F1 `0.459`
  - XGBoost: AUROC `0.640`, F1 `0.465`

- `54`-feature enriched model
  - Logistic Regression: AUROC `0.807`, F1 `0.589`
  - XGBoost: AUROC `0.817`, F1 `0.606`

This is the main scientific message of the secondary project:

- generic physiology alone was not enough
- once the feature set became more microbiology-aware, the task became much more learnable

That is a positive finding because it tells us where the real signal lives.

## The Signal Is Concentrated In A Small Number Of Features

We then reduced the enriched model from `54` features to `19` features using SHAP-style importance and correlation pruning.

Pruned `19`-feature model:

- Logistic Regression: AUROC `0.812`, AUPRC `0.672`, F1 `0.615`
- XGBoost: AUROC `0.820`, AUPRC `0.707`, F1 `0.604`

This is one of the strongest findings in the repo.

It means:

- we do not need a very large feature table to keep the signal
- the clinically useful information is concentrated in a compact set of interpretable variables
- the current task is easier to explain to clinicians than a large opaque model

## What The Most Important Features Say

Top SHAP-style features in the enriched model were:

1. `prior_subject_s_aureus_positive_90d`
2. `index_hours_draw_to_alert`
3. `prealert_blood_culture_draws_7d`
4. `lab_platelets_last_24h`
5. `lab_platelets_min_24h`
6. `anchor_age`
7. `prior_subject_s_aureus_positive_365d`
8. `lab_creatinine_last_24h`
9. `prealert_blood_culture_draws_24h`
10. `lab_creatinine_max_24h`

These features point to a clinically coherent story.

### 1. Prior `S. aureus` history matters

Recent prior `S. aureus` positivity was the strongest feature.

That is a meaningful finding because it suggests the model is detecting:

- recurrent infection risk
- persistent unresolved source risk
- host or device situations where `S. aureus` keeps reappearing

This is not random statistical noise. It is a clinically plausible high-yield signal.

### 2. Faster draw-to-alert time matters

`index_hours_draw_to_alert` was one of the strongest features.

This suggests that culture process dynamics contain important information. In practice, faster positivity can reflect a more substantial bloodstream burden and a more convincing true pathogen signal.

That is encouraging because it means the model is not relying only on generic illness severity. It is using information from the microbiology pathway itself.

### 3. Repeated pre-alert blood-culture drawing matters

The number of blood-culture draws before the alert was also highly important.

This suggests the model is picking up a real bedside pattern:

- clinicians were already concerned enough to keep drawing cultures
- that concern often precedes later-confirmed `S. aureus`

That is useful because it connects the model directly to clinical workflow and clinical suspicion.

### 4. Platelets and creatinine still matter

Platelet and creatinine features remained important even after adding microbiology-process variables.

This means the model is not purely a microbiology-history model. It still benefits from host state and acuity:

- thrombocytopenia may reflect systemic inflammation or sepsis severity
- renal dysfunction may reflect a sicker host phenotype or more complicated infection context

### 5. The task is not just a physiology problem

The biggest improvement came from adding:

- prior `S. aureus` history
- prior CoNS and staphylococcal history
- draw-to-alert time
- pre-alert blood-culture draw counts

This is one of the clearest scientific insights from the whole analysis:

- if we want to predict later-confirmed `S. aureus`, microbiology-process context and prior organism history matter more than routine vitals alone

## The Signal Is Robust Across Model Families

Using the same pruned `19` features and the same primary cohort:

- Logistic Regression: AUROC `0.812`, AUPRC `0.672`, F1 `0.615`
- Elastic Net Logistic: AUROC `0.812`, AUPRC `0.672`, F1 `0.617`
- Random Forest: AUROC `0.814`, AUPRC `0.713`, F1 `0.619`
- XGBoost: AUROC `0.820`, AUPRC `0.707`, F1 `0.604`

This is another positive result.

It means the dataset is carrying real signal rather than depending on one special algorithm. Different model families recover similar performance:

- linear models
- penalized linear models
- bagged trees
- boosted trees

That makes the result more credible.

## Practical Clinical Interpretation

The current project does not support an automatic treatment rule.

What it does support is an early-prioritization idea:

- some first Gram-positive alerts already look more `S. aureus`-like than others
- those patients can be reviewed faster
- those alerts can be escalated faster for source evaluation and repeat-culture planning

That is a strong and realistic use case for a secondary model.

## Bottom Line

The best positive message from this analysis is:

- the refined same-episode dataset is scientifically meaningful
- the signal improved sharply once we used microbiology-aware features
- the signal remains strong after pruning to a compact `19`-feature model
- the main drivers are clinically interpretable and biologically plausible
- the result is stable across multiple model families

So the current `S. aureus` project is no longer just an exploratory side task. It has become a credible, clinically interpretable secondary analysis that complements the main contaminant-vs-significant first-alert project.
