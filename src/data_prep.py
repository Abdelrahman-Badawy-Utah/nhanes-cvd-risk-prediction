"""
data_prep.py
============
Everything involved in turning the raw NHANES extract into a clean,
modeling-ready dataset: restricting to adults, handling missing data, and
filling in a small number of missing clinical values.

WHY THIS IS ITS OWN MODULE: separating data preparation from modeling
means you can change how the data is cleaned without touching a single
line of modeling code, and vice versa -- a basic but important software
engineering habit for any project meant to be read/reused by someone else.
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.config import (
    ALL_PREDICTORS, DIETARY_VARS, CLINICAL_VARS, OUTCOME_VAR,
    FAIRNESS_AUDIT_VARS, RANDOM_SEED,
)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load the raw NHANES extract from an Excel file."""
    return pd.read_excel(file_path, sheet_name="Sheet1")


def restrict_to_adults(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only adults (age >= 18), matching the clinical population of
    interest for cardiovascular disease risk."""
    return df[df["RIDAGEYR"] >= 18].copy()


def apply_complete_case_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop anyone missing dietary data or an unknown CVD outcome. We
    don't try to impute an entire day's diet or guess someone's CVD
    status -- that would introduce much bigger assumptions than filling
    in a handful of missing clinical measurements (next step)."""
    required_cols = DIETARY_VARS + [OUTCOME_VAR]
    return df.dropna(subset=required_cols).copy()


def impute_clinical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in the small number of missing clinical values (BMI, waist
    circumference, blood pressure, cholesterol, CRP) using an iterative,
    multivariate imputation approach (conceptually the same idea as
    MICE): each missing value is predicted from a person's other known
    variables, rather than being deleted or filled with a flat average.

    IMPORTANT: this function fits the imputer on the full dataset (all
    predictor columns) BEFORE any train/test split. That is fine and
    standard for imputation specifically, because it's just estimating
    reasonable values for missing measurements, not learning anything
    about the OUTCOME (unlike feature selection or hyperparameter
    tuning, which must never see the test set -- see feature_selection.py
    and modeling.py for where we deliberately restrict those steps to the
    training data only).
    """
    df = df.copy()
    imputer = IterativeImputer(max_iter=10, random_state=RANDOM_SEED)
    df[ALL_PREDICTORS] = imputer.fit_transform(df[ALL_PREDICTORS])
    return df


def prepare_dataset(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """Run the full data preparation pipeline end to end and return one
    clean, analysis-ready DataFrame. Retains the fairness-audit columns
    (sex, race/ethnicity) alongside the predictors and outcome -- these
    are NEVER used as model inputs, only for the subgroup fairness audit
    performed after modeling (see fairness.py)."""
    if verbose:
        print("[data_prep] Loading raw data ...")
    raw = load_raw_data(file_path)
    if verbose:
        print(f"[data_prep] Loaded {len(raw):,} rows.")

    adults = restrict_to_adults(raw)
    if verbose:
        print(f"[data_prep] Restricted to adults (18+): {len(adults):,} rows.")

    complete_case = apply_complete_case_filter(adults)
    if verbose:
        n_dropped = len(adults) - len(complete_case)
        print(f"[data_prep] Dropped {n_dropped:,} rows missing dietary data "
              f"or CVD status. Remaining: {len(complete_case):,} rows.")

    imputed = impute_clinical_variables(complete_case)
    if verbose:
        remaining_na = imputed[ALL_PREDICTORS].isna().sum().sum()
        print(f"[data_prep] Imputed remaining missing clinical values "
              f"(should be 0 missing now: {remaining_na}).")

    # Keep predictors + outcome + fairness-audit columns, drop everything
    # else (survey admin columns, raw MCQ items already summarized in
    # 'cvd', etc.) to keep the dataset focused.
    keep_cols = ALL_PREDICTORS + [OUTCOME_VAR] + FAIRNESS_AUDIT_VARS
    keep_cols = [c for c in keep_cols if c in imputed.columns]
    final_df = imputed[keep_cols].copy()
    final_df[OUTCOME_VAR] = final_df[OUTCOME_VAR].astype(int)

    if verbose:
        outcome_counts = final_df[OUTCOME_VAR].value_counts()
        outcome_pct = final_df[OUTCOME_VAR].value_counts(normalize=True) * 100
        print(f"[data_prep] Final analytic sample: {len(final_df):,} adults")
        print(f"  - No CVD (0): {outcome_counts[0]:,} ({outcome_pct[0]:.1f}%)")
        print(f"  - Has CVD (1): {outcome_counts[1]:,} ({outcome_pct[1]:.1f}%)")

    return final_df
