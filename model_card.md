# Model Card: NHANES Cardiovascular Disease Risk Model

This model card follows the general format popularized by Mitchell et al.
(2019), "Model Cards for Model Reporting," adapted for a research/
portfolio project.

## Model Details

- **Developed by:** [Your name here] as a portfolio project.
- **Model type:** Gradient-boosted decision tree classifier (LightGBM in
  the reference run; the pipeline in this repo automatically selects
  whichever of 5 candidate model types -- Logistic Regression, Random
  Forest, SVM, XGBoost, LightGBM -- achieves the best cross-validated
  AUROC, so the specific winning model can vary run to run).
- **Task:** Binary classification -- predicting the probability that an
  adult has ever been diagnosed with cardiovascular disease (CVD).
- **Inputs:** Up to 20 variables spanning dietary nutrient intake (from a
  single 24-hour recall), anthropometric measurements, blood pressure,
  blood biomarkers, and age. The exact variable list is chosen
  automatically by Recursive Feature Elimination and is saved alongside
  the model artifact.
- **Output:** A calibrated probability (0-1) of having CVD.
- **Inspired by:** Ahiduzzaman & Hasan (2025), "Interpretable machine
  learning for cardiovascular risk prediction," PLoS One. This project
  independently re-implements and extends the general approach; it is
  not an official replication and makes several deliberate methodological
  changes described in the README.

## Intended Use

- **Primary intended use:** Educational and portfolio demonstration of an
  end-to-end, methodologically careful ML pipeline on public health
  survey data.
- **Intended users:** People evaluating the author's data science skills
  (e.g. hiring managers, technical interviewers); learners studying
  applied ML on health data.
- **Out-of-scope uses:** This model is **not validated for, and must
  never be used for, real clinical decision-making, individual diagnosis,
  insurance underwriting, or any application affecting a real person's
  healthcare, coverage, or opportunities.** It was trained on a research
  dataset with substantial limitations (see below) and has not undergone
  the validation, regulatory review, or prospective testing required for
  any real-world health application.

## Training Data

- **Source:** National Health and Nutrition Examination Survey (NHANES),
  2017-2023 cycles, National Center for Health Statistics (public,
  de-identified data).
- **Population:** U.S. adults aged 18+, after excluding anyone missing
  dietary intake data or CVD status (~14,500-15,000 people, varying
  slightly by exact run).
- **Outcome definition:** Self-reported, physician-diagnosed history of
  angina, coronary heart disease, congestive heart failure, heart attack,
  or stroke (combined into one "has CVD" indicator).
- **Class balance:** Approximately 88% no-CVD / 12% has-CVD. The training
  set is rebalanced via random oversampling of the minority class
  (applied fresh inside every cross-validation fold -- see README for why
  this matters); the test set is left at its natural, imbalanced ratio.

## Evaluation Data

- A held-out 20% test split, stratified by outcome, never used in
  feature selection or hyperparameter tuning.

## Metrics

Reported in `results/performance_table.csv` and
`results/auroc_confidence_intervals.csv`: Accuracy, Precision, Recall
(Sensitivity), Specificity, F1-score, and AUROC with bootstrapped 95%
confidence intervals. In a reference run, all 5 models landed within a
tight AUROC band (~0.82-0.83) with heavily overlapping confidence
intervals -- meaning no single algorithm is statistically distinguishable
from the others on this task and dataset.

Calibration (`results/calibration_curves_before.png`) showed that raw
model probabilities substantially **over-predicted risk** across the
board -- a direct side effect of training on oversampled data. Post-hoc
Platt scaling calibration corrected this (Brier score roughly halved)
while leaving AUROC essentially unchanged, and is applied by default in
the Streamlit app and saved model artifact.

## Fairness / Subgroup Analysis

The model's predictors deliberately exclude sex and race/ethnicity, but
`src/fairness.py` audits test-set performance broken down by these
variables (never fed to the model) plus age band (which IS a predictor).

**In a reference run**, AUROC varied meaningfully across subgroups:

- **By sex:** ~0.84 (male) vs. ~0.81 (female).
- **By race/ethnicity:** ranged from ~0.75 (Non-Hispanic Black
  participants) to ~0.92 (Non-Hispanic Asian participants), with other
  groups in between.
- **By age band:** AUROC within individual age bands (~0.72-0.77) is
  noticeably lower than the model's overall AUROC (~0.83) -- expected,
  since much of the model's overall discrimination comes from age itself
  varying between people; once you already condition on a narrow age
  range, less signal remains.

**How to interpret this responsibly:** these gaps are real and worth
disclosing, but several of the subgroup samples are small (e.g. n=125-338
for some race/ethnicity categories in the test set), so individual
subgroup AUROC estimates carry substantial uncertainty themselves and
shouldn't be treated as precise. This audit identifies a **disparity to
investigate further** (e.g. with a larger sample, subgroup-specific
calibration, or additional features), not a proven causal mechanism. No
claim is made here about *why* performance differs across groups --
plausible contributors include unequal sample sizes, differences in
healthcare access affecting diagnosis/self-report accuracy, or
unmeasured confounders, none of which this project attempts to
disentangle.

## Ethical Considerations

- **Self-reported outcome:** CVD status relies on participants recalling
  a prior doctor's diagnosis, introducing potential recall error or
  under-diagnosis, which may not be uniform across subgroups (e.g.,
  healthcare access affects diagnosis rates).
- **Correlation, not causation:** All relationships identified (including
  dietary associations) are observational. Nothing here implies that
  changing a given variable would change a person's actual risk.
- **Sensitive demographic analysis:** The fairness audit reports
  real, measured differences in model performance across race/ethnicity
  and sex. These figures are reported transparently and should not be
  used to draw conclusions about biological differences between groups;
  the most defensible reading is that the model (and/or the underlying
  data collection) performs unevenly and warrants further investigation
  before any real-world use.

## Caveats and Recommendations

- Not clinically validated; for portfolio/educational use only.
- Cross-sectional design; no temporal or external validation performed.
- Missing medication-use data plausibly confounds the cholesterol-age
  relationship identified via SHAP (see README) -- a limitation also
  flagged in the source paper that inspired this project.
- Hyperparameter search spaces and cross-validation fold counts were kept
  modest to run in a reasonable time on limited hardware; a production
  system would benefit from a wider search and more folds.
- Before any real-world use, this model would need: external validation
  on an independent cohort, prospective testing, clinical oversight,
  a fairness remediation plan for the subgroup gaps identified above, and
  regulatory review appropriate to its intended use.
