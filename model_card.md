# Model Card: Cardiovascular Disease Risk Prediction Model

This model card summarizes what the model does, what it was trained on,
how well it performs, and its known limitations, following the general
practice of model documentation for machine learning systems.

## Model Overview

The model estimates the probability that an individual has
cardiovascular disease (CVD) -- including angina, coronary heart
disease, congestive heart failure, heart attack, or stroke -- based on
age, blood pressure, total and HDL cholesterol, kidney function
(eGFR), diabetes status, smoking status, body measurements, and dietary
intake.

## Training Data

NHANES (National Health and Nutrition Examination Survey), 2015-2016,
2017-2020 (pre-pandemic), and 2021-2023 cycles, National Center for
Health Statistics. Approximately 17,000 U.S. adults after data cleaning.

## Performance

Five classifiers were evaluated (Logistic Regression, Random Forest,
SVM, XGBoost, LightGBM). All achieved comparable discrimination --
approximately 81-82% AUROC, meaning the model correctly ranks a
randomly selected CVD-positive individual above a randomly selected
CVD-negative individual about 81-82% of the time. No model showed a
statistically meaningful advantage over the others.

The strongest and most consistent predictors were **age**, **total
cholesterol**, and **waist circumference**. Notably, all three newly
added clinical predictors -- **estimated kidney function (eGFR)**,
**diabetes status**, and **HDL cholesterol** -- ranked among the top 6
predictors by SHAP importance, each in the clinically expected
direction: lower eGFR (worse kidney function), having diabetes, and
lower HDL cholesterol were each associated with higher predicted risk.

## A Note on Smoking Status

Smoking status -- also newly added to this project, alongside HDL,
diabetes status, and eGFR -- was **not** selected by the automatic
feature selection step (Recursive Feature Elimination) in the current
run. This is a genuine, reported result, not an error.

This does not mean smoking is unimportant for cardiovascular risk in
general; it is one of the most well-established cardiovascular risk
factors in the clinical literature. A more likely explanation is that
feature selection methods like RFE tend to keep one representative from
a cluster of correlated risk factors rather than all of them, and
smoking's signal in this dataset likely overlaps substantially with the
other newly added predictors (diabetes status, kidney function, and
HDL cholesterol are all known to correlate with smoking history at the
population level). This is a useful illustration of a general
limitation of automatic feature selection: a variable being dropped
reflects redundancy with other selected variables, not necessarily a
lack of real-world importance.

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

- **By sex:** AUROC was approximately 0.83 for male participants and
  0.80 for female participants -- a modest but measurable difference.
- **By race/ethnicity:** AUROC ranged from approximately 0.79
  (Non-Hispanic Black participants) to 0.84 (Other Hispanic
  participants). In practical terms, this indicates the model's risk
  estimates are currently less reliable for Non-Hispanic Black patients
  than for several other groups in this dataset.

**Caveats:**

- Several subgroups have limited sample sizes in the test set (as few
  as ~175-405 individuals for some race/ethnicity categories, versus
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
- Notably, the newer clinical PREVENT risk calculator that inspired
  several of this project's added predictors similarly removed race as
  a direct model input in 2023, for related fairness reasons -- this
  project's approach (excluding race/sex from prediction, auditing for
  it afterward) reflects the same underlying principle.

## Intended Use and Out-of-Scope Uses

This model is intended for educational and portfolio demonstration
purposes only. It is **not validated for and must not be used for**
clinical decision-making, diagnosis, insurance underwriting, or any
application affecting an individual's healthcare or opportunities.

## Additional Limitations

- **Associational, not causal.** Reported relationships (including
  dietary associations) are observational. No claim is made that
  modifying a given predictor would change an individual's actual risk.
- **Self-reported outcome and exposures.** CVD status, smoking status,
  and diabetes status are all based on participant self-report /
  recall of a prior physician diagnosis -- subject to recall error and
  under-reporting.
- **Cross-sectional design.** The data represent a single point in time
  per participant; the analysis cannot establish temporal or causal
  ordering between predictors and outcome.
- **Missing medication data.** Medication use (e.g., lipid-lowering or
  antihypertensive therapy) was not available. This plausibly explains
  an observed pattern in which very low cholesterol among older adults
  was associated with higher predicted risk -- likely reflecting that
  individuals already under treatment for cardiovascular conditions tend
  to have pharmacologically lowered cholesterol, rather than low
  cholesterol itself being a risk factor. It may also affect the
  interpretation of blood pressure and eGFR values for patients on
  relevant medications.
- **Diabetes status is coarse.** A binary yes/no does not capture
  diabetes duration, type (1 vs. 2), or glycemic control (e.g., HbA1c
  level), all of which are clinically relevant to cardiovascular risk.
- **Smoking status is coarse.** The current/former/never categorization
  does not capture intensity (cigarettes per day) or duration, both
  clinically relevant -- and, as noted above, was not selected as a
  model predictor in the current run despite being a candidate.

## Requirements for Real-World Use

Before any application beyond portfolio/educational use, this model
would require: external validation on an independent cohort, clinical
review, a defined remediation plan for the subgroup performance gaps
identified above, and regulatory review appropriate to the intended
application.
