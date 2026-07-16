# Data

This folder contains the NHANES data extract used by this project.

## Source

**National Health and Nutrition Examination Survey (NHANES)**, 2015-2016,
2017-2020 (pre-pandemic), and 2021-2023 cycles, published by the CDC's
National Center for Health Statistics. Public, de-identified data.

- Official source: https://www.cdc.gov/nchs/nhanes

## File

`nhanes_cvd_extract.xlsx` -- one row per person, one sheet named `Sheet1`,
including:

- `RIDAGEYR` (age), `RIAGENDR` (sex), `RIDRETH3` (race/ethnicity)
- Dietary variables from the NHANES DR1T* series (protein, carbohydrates,
  vitamins, minerals, etc.)
- `BMXBMI`, `BMXWAIST` (body measurements)
- `SBP1`, `DBP1` (blood pressure)
- `LBXTC` (total cholesterol), `HDL` (HDL cholesterol), `LBXHSCRP` (CRP)
- `SMQ020`, `SMQ040` (raw smoking questionnaire responses) and a derived
  `smoking_status` column (0 = never smoked, 1 = former smoker,
  2 = current smoker)
- `DIQ010` (raw diabetes questionnaire response) and a derived
  `diabetes_status` column (0 = no, 1 = yes)
- `LBXSCR` (serum creatinine) and a derived `eGFR` column (estimated
  glomerular filtration rate, a measure of kidney function)
- `cvd` (a 0/1 column indicating cardiovascular disease status, built
  from the NHANES questions on angina, coronary heart disease,
  congestive heart failure, heart attack, and stroke)

## How this extract was built

This extract was built using R and the `nhanesA` package, pulling and
merging the relevant components (demographics, dietary interview, body
measurements, blood pressure, total and HDL cholesterol, CRP, smoking
and diabetes questionnaires, serum creatinine, and the medical
conditions questionnaire) across three survey cycles, then combining
them into the single flat table the notebook expects.

**Derived variables:**

- **Smoking status** is derived from `SMQ020` ("smoked at least 100
  cigarettes in life") and `SMQ040` ("do you now smoke cigarettes").
  Never smokers answered no to `SMQ020`; former smokers answered yes to
  `SMQ020` but "not at all" to `SMQ040`; current smokers answered yes to
  `SMQ020` and "every day" or "some days" to `SMQ040`.
- **Diabetes status** is derived directly from `DIQ010` ("has a doctor
  ever told you that you have diabetes").
- **eGFR** is calculated from serum creatinine (`LBXSCR`), age, and sex
  using the 2021 CKD-EPI creatinine equation -- the current race-free
  standard recommended by the National Kidney Foundation and American
  Society of Nephrology, replacing an earlier version that included a
  race coefficient.

**A note on cross-cycle variable naming:** two of the added variables
changed names partway through the 2015-2023 span covered here -- HDL
cholesterol (`LBDHDD` in earlier cycles, `LBXHDD` in 2021-2023) and, if
added in the future, blood pressure medication use (`BPQ050A` in
earlier cycles, `BPQ150` in 2021-2023, with a changed skip pattern).
The extraction script resolves these automatically per cycle, matching
the same pattern already used for blood pressure measurement method
(oscillometric vs. auscultatory), which also changed across cycles.

**A note on survey weights:** NHANES is a complex survey design with its
own sampling weights, intended to make estimates representative of the
full U.S. population. This project, like the paper that inspired it,
does **not** apply survey weighting -- results describe patterns within
this specific sample of survey respondents, not necessarily
weighted/generalizable estimates for the full U.S. population. This is a
known simplification, noted honestly here rather than glossed over.
