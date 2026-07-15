"""
fairness.py
===========
Auditing model performance across demographic subgroups (sex, race/
ethnicity, and age band) that were deliberately EXCLUDED from the model's
predictors.

WHY THIS MATTERS: a model can have great overall AUROC while performing
very unevenly across subgroups -- for example, ranking risk well for men
but poorly for women. Health-tech organizations increasingly expect this
kind of audit as standard practice, not an afterthought. This module
never feeds sex or race/ethnicity into the model itself; it only uses
them AFTER the fact to check whether the model's errors are spread
evenly or concentrated in particular groups.

A NOTE ON SCOPE: this is a fairness AUDIT, not a fairness FIX. If this
audit turned up a large gap between subgroups, the appropriate next
steps (reweighting, subgroup-specific thresholds, collecting more
data for an underperforming group, etc.) are a substantial project in
their own right and are called out as future work rather than
implemented here.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

from src.config import SEX_LABELS, RACE_ETHNICITY_LABELS


def _subgroup_metrics(y_true_group, risk_scores_group, predictions_group):
    """Compute AUROC, recall (sensitivity), and specificity for one
    subgroup. Returns NaN for a metric if it can't be computed (e.g. a
    subgroup with only one class present, or too few people)."""
    result = {"n": len(y_true_group)}

    if len(np.unique(y_true_group)) < 2:
        result["AUROC"] = np.nan
        result["Recall (Sensitivity)"] = np.nan
        result["Specificity"] = np.nan
        return result

    result["AUROC"] = roc_auc_score(y_true_group, risk_scores_group)
    result["Recall (Sensitivity)"] = recall_score(y_true_group, predictions_group, zero_division=np.nan)
    tn, fp, fn, tp = confusion_matrix(y_true_group, predictions_group).ravel()
    result["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return result


def _get_risk_scores(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def audit_by_sex(pipeline, X_test, y_test, sex_series):
    """Break down test-set performance by sex (NHANES RIAGENDR: 1=Male,
    2=Female). Returns a small DataFrame, one row per sex."""
    risk_scores = _get_risk_scores(pipeline, X_test)
    predictions = pipeline.predict(X_test)
    y_test_arr = np.asarray(y_test)
    sex_arr = np.asarray(sex_series)

    rows = {}
    for code, label in SEX_LABELS.items():
        mask = sex_arr == code
        if mask.sum() == 0:
            continue
        rows[label] = _subgroup_metrics(
            y_test_arr[mask], risk_scores[mask], predictions[mask]
        )
    return pd.DataFrame(rows).T.round(4)


def audit_by_race_ethnicity(pipeline, X_test, y_test, race_series):
    """Break down test-set performance by race/ethnicity (NHANES
    RIDRETH3). Returns a small DataFrame, one row per group present in
    the test set."""
    risk_scores = _get_risk_scores(pipeline, X_test)
    predictions = pipeline.predict(X_test)
    y_test_arr = np.asarray(y_test)
    race_arr = np.asarray(race_series)

    rows = {}
    for code, label in RACE_ETHNICITY_LABELS.items():
        mask = race_arr == code
        if mask.sum() == 0:
            continue
        rows[label] = _subgroup_metrics(
            y_test_arr[mask], risk_scores[mask], predictions[mask]
        )
    return pd.DataFrame(rows).T.round(4)


def audit_by_age_band(pipeline, X_test, y_test, age_series,
                       bins=(18, 40, 55, 70, 120),
                       labels=("18-39", "40-54", "55-69", "70+")):
    """Break down test-set performance by age band. Note age IS a model
    predictor (unlike sex/race), so this isn't an 'excluded variable'
    audit like the other two -- it's a check for whether performance is
    stable across the age range, since CVD prevalence rises sharply with
    age and a model could quietly perform worse in some age bands."""
    risk_scores = _get_risk_scores(pipeline, X_test)
    predictions = pipeline.predict(X_test)
    y_test_arr = np.asarray(y_test)

    age_bands = pd.cut(pd.Series(np.asarray(age_series)), bins=bins,
                        labels=labels, right=False)
    age_bands_arr = np.asarray(age_bands)

    rows = {}
    for band in labels:
        mask = age_bands_arr == band
        if mask.sum() == 0:
            continue
        rows[band] = _subgroup_metrics(
            y_test_arr[mask], risk_scores[mask], predictions[mask]
        )
    return pd.DataFrame(rows).T.round(4)
