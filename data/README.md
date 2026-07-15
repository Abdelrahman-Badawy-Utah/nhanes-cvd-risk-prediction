# Data

This folder is where your NHANES data extract goes.

## Where to get it

This project uses data from the **National Health and Nutrition
Examination Survey (NHANES)**, 2017-2023 cycles, published by the CDC's
National Center for Health Statistics. It's public and free:

- Official source: https://www.cdc.gov/nchs/nhanes

## What file the notebook expects

Place your extracted data file in this folder (e.g.
`data/nhanes_cvd_extract.xlsx`), then update the `FILE_PATH` variable
near the top of `CVD_Risk_Prediction.ipynb` to match your file's name.

The notebook expects one sheet named `Sheet1`, with one row per person
and columns including (at minimum):

- `RIDAGEYR` (age), `RIAGENDR` (sex), `RIDRETH3` (race/ethnicity)
- Dietary variables from the NHANES DR1T* series (protein, carbohydrates,
  vitamins, minerals, etc.)
- `BMXBMI`, `BMXWAIST` (body measurements)
- `SBP1`, `DBP1` (blood pressure)
- `LBXTC`, `LBXHSCRP` (cholesterol, CRP)
- `cvd` (a 0/1 column indicating cardiovascular disease status, built
  from the NHANES questions on angina, coronary heart disease,
  congestive heart failure, heart attack, and stroke)

## How this extract was built

The raw NHANES data is published across many separate files (one per
survey component, per 2-year cycle). This project's data extract was
built by pulling and merging the relevant components (demographics,
dietary interview, body measurements, blood pressure, cholesterol, CRP,
and the medical conditions questionnaire) using R and the `nhanesA`
package, then combining them into the single flat table this notebook
expects.

**A note on survey weights:** NHANES is a complex survey design with its
own sampling weights, intended to make estimates representative of the
full U.S. population. This project, like the paper that inspired it,
does **not** apply survey weighting -- results describe patterns within
this specific sample of survey respondents, not necessarily
weighted/generalizable estimates for the full U.S. population. This is a
known simplification, noted honestly here rather than glossed over.

## A note on committing data to GitHub

NHANES data is public, so committing a modest-sized extract to this repo
(as this project does) is generally fine. If your extract file is very
large (tens of MB+), consider whether GitHub's file size limits apply,
or whether linking to the data source instead (rather than committing
the file itself) makes more sense for your situation.
