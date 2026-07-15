"""
evaluation.py
=============
Turning trained model pipelines into a full evaluation: the standard
metrics scorecard, ROC curves, bootstrap confidence intervals around
AUROC, and calibration (reliability) analysis.

WHY BOOTSTRAP CONFIDENCE INTERVALS: a single AUROC number (e.g. 0.816)
invites over-interpretation -- is that meaningfully different from
0.811? Bootstrapping the test set thousands of times and recomputing
AUROC each time gives a range of plausible values, which is a much more
honest way to compare models that are close to each other.

WHY CALIBRATION: AUROC only measures whether a model RANKS people
correctly (higher-risk people get higher scores), not whether its actual
predicted probabilities are trustworthy as probabilities. A model can
have great AUROC and still be badly calibrated (e.g. everyone it labels
"70% risk" might really only have a 40% chance of CVD). For any
clinically-oriented model, calibration matters as much as discrimination.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from src.config import RANDOM_SEED

MODEL_COLUMN_ORDER = ["XGBoost", "Random Forest", "Logistic Regression", "LightGBM", "SVM"]


def _get_risk_scores(pipeline, X):
    """Return a continuous risk score in [0, 1]-ish range for ranking
    purposes. Uses predict_proba when available, falls back to
    decision_function (e.g. for our SVM, which we deliberately run
    without probability=True for speed -- see modeling.py)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluate_all_models(trained_models: dict, X_test, y_test):
    """Compute Accuracy / Precision / Recall / Specificity / F1 / AUROC
    for every model on the (untouched) test set. Returns a DataFrame with
    metrics as rows and models as columns."""
    performance_results = {}
    roc_curve_data = {}
    risk_scores_by_model = {}

    for model_name, info in trained_models.items():
        pipeline = info["pipeline"]
        predictions = pipeline.predict(X_test)
        risk_scores = _get_risk_scores(pipeline, X_test)
        risk_scores_by_model[model_name] = risk_scores

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        specificity = tn / (tn + fp)

        performance_results[model_name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall (Sensitivity)": recall_score(y_test, predictions),
            "Specificity": specificity,
            "F1-score": f1_score(y_test, predictions),
            "AUROC": roc_auc_score(y_test, risk_scores),
        }

        fpr, tpr, _ = roc_curve(y_test, risk_scores)
        roc_curve_data[model_name] = (fpr, tpr, performance_results[model_name]["AUROC"])

    performance_table = pd.DataFrame(performance_results).round(4)
    available_order = [m for m in MODEL_COLUMN_ORDER if m in performance_table.columns]
    performance_table = performance_table[available_order]

    return performance_table, roc_curve_data, risk_scores_by_model


def plot_roc_curves(roc_curve_data: dict, output_path: str):
    """Save a single figure comparing ROC curves for every model."""
    plt.figure(figsize=(7, 6))
    for model_name, (fpr, tpr, auc_value) in roc_curve_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUROC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    plt.xlabel("False Positive Rate  (1 - Specificity)")
    plt.ylabel("True Positive Rate  (Recall / Sensitivity)")
    plt.title("ROC Curve Comparison Across Models")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def bootstrap_auroc_confidence_intervals(risk_scores_by_model: dict, y_test,
                                          n_bootstrap: int = 2000,
                                          verbose: bool = True) -> pd.DataFrame:
    """For each model, resample the test set (with replacement) many
    times and recompute AUROC each time. Returns the median AUROC and a
    95% confidence interval (2.5th-97.5th percentile) for each model.

    This is cheap to compute because it re-uses the risk scores already
    predicted once -- no retraining happens here, we're just repeatedly
    re-scoring different random subsets of the SAME predictions.
    """
    if verbose:
        print(f"[evaluation] Bootstrapping AUROC confidence intervals "
              f"({n_bootstrap:,} resamples per model) ...")

    y_test_array = np.asarray(y_test)
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_test_array)

    results = {}
    for model_name, scores in risk_scores_by_model.items():
        scores_array = np.asarray(scores)
        boot_aurocs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            y_boot = y_test_array[idx]
            # Skip the rare resample that happens to contain only one
            # class (AUROC undefined in that case).
            if len(np.unique(y_boot)) < 2:
                continue
            boot_aurocs.append(roc_auc_score(y_boot, scores_array[idx]))

        boot_aurocs = np.array(boot_aurocs)
        results[model_name] = {
            "Median AUROC": np.median(boot_aurocs),
            "95% CI Lower": np.percentile(boot_aurocs, 2.5),
            "95% CI Upper": np.percentile(boot_aurocs, 97.5),
        }

    ci_table = pd.DataFrame(results).T.round(4)
    available_order = [m for m in MODEL_COLUMN_ORDER if m in ci_table.index]
    ci_table = ci_table.loc[available_order]
    return ci_table


def plot_calibration_curves(trained_models: dict, X_test, y_test, output_path: str,
                             n_bins: int = 10):
    """Plot a reliability diagram (predicted probability vs. observed
    frequency) for every model that produces genuine probabilities.
    SVM is excluded here since we deliberately run it without
    probability=True for speed (see modeling.py) -- its raw decision
    scores aren't on a 0-1 probability scale, so a calibration curve for
    it wouldn't be meaningful without a separate calibration step.
    """
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    brier_scores = {}
    for model_name, info in trained_models.items():
        pipeline = info["pipeline"]
        if not hasattr(pipeline, "predict_proba"):
            continue
        probs = pipeline.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
        brier_scores[model_name] = brier_score_loss(y_test, probs)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction with CVD")
    plt.title("Calibration (Reliability) Curves")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return pd.Series(brier_scores, name="Brier Score").round(4)


def calibrate_model(pipeline, X_train, y_train, method: str = "sigmoid"):
    """Wrap a fitted pipeline in a post-hoc calibration layer (Platt
    scaling by default). Returns a new, separately-fitted calibrated
    version -- the original pipeline is left untouched.

    NOTE: CalibratedClassifierCV internally re-fits using cross-
    validation on the data you pass in, so we pass the TRAINING data
    here, never the test set.
    """
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated
