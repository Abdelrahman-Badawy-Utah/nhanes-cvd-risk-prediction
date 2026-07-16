# Model Card: Cardiovascular Disease Risk Prediction Model

This model card summarizes what the model does, what it was trained on,
how well it performs, and its known limitations, following the general
practice of model documentation for machine learning systems.

## Model Overview

The model estimates the probability that an individual has
cardiovascular disease (CVD) -- including angina, coronary heart
disease, congestive heart failure, heart attack, or stroke -- based on
age, blood pressure, cholesterol, body measurements, smoking status, and
dietary intake.

## Training Data

NHANES (National Health and Nutrition Examination Survey), 2015-2016,
2017-2020 (pre-pandemic), and 2021-2023 cycles, National Center for
Health Statistics. Approximately 17,500 U.S. adults after data cleaning.

## Performance

Five classifiers were evaluated (Logistic Regression, Random Forest,
SVM, XGBoost, LightGBM). All achieved comparable discrimination --
approximately 81-82% AUROC, meaning the model correctly ranks a
randomly selected CVD-positive individual above a randomly selected
CVD-negative individual about 81-82% of the time. No model showed a
statistically meaningful advantage over the others.

The most consistent predictors across models were **age**, **total
cholesterol**, and **smoking status**, each positively associated with
predicted risk. Smoking status ranked third in overall importance,
ahead of waist circumference and blood pressure -- consistent with its
well-established role as a cardiovascular risk factor.

## Calibration

Initial predicted probabilities were poorly calibrated: among
individuals assigned a given predicted risk (e.g., 40%), the observed
rate of CVD was consistently lower than the predicted value. This is an
expected consequence of training on class-balanced (oversampled) data.
Post-hoc calibration (Platt scaling) was applied and verified to correct
this -- predicted probabilities now closely match observed outcome rates
without materially affecting AUROC.

## Fairness / Subgroup Performance Audit

**Objective:** determine whether the model's ability to discriminate
between CVD-positive and CVD-negative individuals is consistent across
demographic subgroups not used as model inputs (sex, race/ethnicity).

**Rationale:** a model can exhibit strong aggregate performance while
performing unevenly across subgroups. This audit evaluates that
directly rather than assuming uniformity from an overall metric.

**Findings:**

- **By sex:** AUROC was approximately 0.84 for male participants and
  0.79 for female participants -- a modest but measurable difference.
- **By race/ethnicity:** AUROC ranged from approximately 0.78
  (Other/Multiracial and Non-Hispanic Black participants) to 0.85
  (Mexican American participants). In practical terms, this indicates
  the model's risk estimates are currently less reliable for
  Other/Multiracial and Non-Hispanic Black patients than for several
  other groups in this dataset.

**Caveats:**

- Several subgroups have limited sample sizes in the test set (as few
  as ~170-410 individuals for some race/ethnicity categories, versus
  over 1,400 for others), so subgroup-level AUROC estimates carry
  greater uncertainty for smaller groups.
- The cause of this disparity is not established by this analysis. It
  may reflect unequal sample sizes, differential diagnosis or
  self-report accuracy across groups, unmeasured confounders, or a
  combination of factors. This model card reports the disparity; it
  does not attribute a mechanism.
- This finding should be treated as a documented basis for further
  investigation prior to any use of this model across these subgroups,
  not as a resolved or acceptable status quo.

## Intended Use and Out-of-Scope Uses

This model is intended for educational and portfolio demonstration
purposes only. It is **not validated for and must not be used for**
clinical decision-making, diagnosis, insurance underwriting, or any
application affecting an individual's healthcare or opportunities.

## Additional Limitations

- **Associational, not causal.** Reported relationships (including
  dietary associations) are observational. No claim is made that
  modifying a given predictor would change an individual's actual risk.
- **Self-reported outcome and exposure.** CVD status is based on
  participant recall of a prior physician diagnosis, and smoking status
  is based on self-report -- both subject to recall error and
  under-reporting.
- **Cross-sectional design.** The data represent a single point in time
  per participant; the analysis cannot establish temporal or causal
  ordering between predictors and outcome.
- **Missing medication data.** Medication use (e.g., lipid-lowering
  therapy) was not available. This plausibly explains an observed
  pattern in which very low cholesterol among older adults was
  associated with higher predicted risk -- likely reflecting that
  individuals already under treatment for cardiovascular conditions tend
  to have pharmacologically lowered cholesterol, rather than low
  cholesterol itself being a risk factor.
- **Smoking status is coarse.** The current/former/never categorization
  does not capture intensity (cigarettes per day) or duration (years
  smoked, time since quitting for former smokers), both of which are
  clinically relevant to cardiovascular risk.

## Requirements for Real-World Use

Before any application beyond portfolio/educational use, this model
would require: external validation on an independent cohort, clinical
review, a defined remediation plan for the subgroup performance gaps
identified above, and regulatory review appropriate to the intended
application.
