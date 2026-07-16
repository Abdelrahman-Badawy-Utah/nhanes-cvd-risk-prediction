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
- `LBXTC`, `LBXHSCRP` (cholesterol, CRP)
- `SMQ020`, `SMQ040` (raw smoking questionnaire responses) and a derived
  `smoking_status` column (0 = never smoked, 1 = former smoker,
  2 = current smoker)
- `cvd` (a 0/1 column indicating cardiovascular disease status, built
  from the NHANES questions on angina, coronary heart disease,
  congestive heart failure, heart attack, and stroke)

## How this extract was built

This extract was built using R and the `nhanesA` package, pulling and
merging the relevant components (demographics, dietary interview, body
measurements, blood pressure, cholesterol, CRP, smoking questionnaire,
and the medical conditions questionnaire) across three survey cycles,
then combining them into the single flat table the notebook expects.

Smoking status is derived from two NHANES questions: `SMQ020` ("smoked
at least 100 cigarettes in life") and `SMQ040` ("do you now smoke
cigarettes"). Participants are classified as never smokers (answered no
to `SMQ020`), former smokers (answered yes to `SMQ020` but "not at all"
to `SMQ040`), or current smokers (answered yes to `SMQ020` and "every
day" or "some days" to `SMQ040`).

**A note on survey weights:** NHANES is a complex survey design with its
own sampling weights, intended to make estimates representative of the
full U.S. population. This project, like the paper that inspired it,
does **not** apply survey weighting -- results describe patterns within
this specific sample of survey respondents, not necessarily
weighted/generalizable estimates for the full U.S. population. This is a
known simplification, noted honestly here rather than glossed over.
