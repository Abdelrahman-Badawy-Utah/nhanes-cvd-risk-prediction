"""
run_analysis.py
================
Main entry point. Running this script end to end:

  1. Loads and cleans the NHANES data
  2. Selects features using RFE (training data only -- no leakage)
  3. Tunes and trains 5 models (hyperparameter search + fresh
     oversampling inside every CV fold -- no leakage)
  4. Evaluates all 5 models: scorecard, ROC curves, bootstrapped AUROC
     confidence intervals, calibration curves
  5. Audits the best model for subgroup performance gaps (sex, race/
     ethnicity, age band)
  6. Interprets the best tree-based model with SHAP + LIME
  7. Saves the best overall model + everything the Streamlit app needs
  8. Writes a plain-English results summary

Usage:
    python run_analysis.py

Before running, set RAW_DATA_PATH in src/config.py to point at your
NHANES extract.
"""

import os
import pickle
import warnings

import pandas as pd

from src.config import (
    RAW_DATA_PATH, RESULTS_DIR, MODELS_DIR, ALL_PREDICTORS, OUTCOME_VAR,
)
from src.data_prep import prepare_dataset
from src.feature_selection import select_features
from src.modeling import split_data, tune_and_train_all_models
from src.evaluation import (
    evaluate_all_models, plot_roc_curves, bootstrap_auroc_confidence_intervals,
    plot_calibration_curves, calibrate_model,
)
from src.fairness import audit_by_sex, audit_by_race_ethnicity, audit_by_age_band
from src.interpret import (
    pick_best_tree_model, compute_shap_values, plot_shap_summary,
    top_shap_features, explain_individual_cases,
)

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("NHANES CARDIOVASCULAR DISEASE RISK PREDICTION -- FULL ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    print("\n[1/8] Preparing data ...")
    df = prepare_dataset(RAW_DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Train/test split (BEFORE feature selection -- see step 3)
    # ------------------------------------------------------------------
    print("\n[2/8] Splitting into train/test sets ...")
    X_all = df[ALL_PREDICTORS]
    y_all = df[OUTCOME_VAR]
    X_train_full, X_test_full, y_train, y_test = split_data(X_all, y_all)
    print(f"Training set: {len(X_train_full):,} | Test set: {len(X_test_full):,}")

    # ------------------------------------------------------------------
    # 3. Feature selection (TRAINING DATA ONLY -- no leakage)
    # ------------------------------------------------------------------
    print("\n[3/8] Selecting features (using training data only) ...")
    selected_features = select_features(X_train_full, y_train, n_features_to_select=20)
    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    # ------------------------------------------------------------------
    # 4. Hyperparameter tuning + training (oversampling inside each CV fold)
    # ------------------------------------------------------------------
    print("\n[4/8] Tuning and training 5 models ...")
    trained_models = tune_and_train_all_models(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluation: scorecard, ROC, bootstrap CI, calibration
    # ------------------------------------------------------------------
    print("\n[5/8] Evaluating all models on the untouched test set ...")
    performance_table, roc_curve_data, risk_scores_by_model = evaluate_all_models(
        trained_models, X_test, y_test
    )
    print("\n===================== MODEL SCORECARD =====================")
    print(performance_table)
    print("=============================================================")
    performance_table.to_csv(os.path.join(RESULTS_DIR, "performance_table.csv"))

    plot_roc_curves(roc_curve_data, os.path.join(RESULTS_DIR, "roc_curve_comparison.png"))

    ci_table = bootstrap_auroc_confidence_intervals(risk_scores_by_model, y_test)
    print("\n=========== BOOTSTRAPPED AUROC (95% CONFIDENCE INTERVALS) ===========")
    print(ci_table)
    print("======================================================================")
    ci_table.to_csv(os.path.join(RESULTS_DIR, "auroc_confidence_intervals.csv"))

    brier_scores = plot_calibration_curves(
        trained_models, X_test, y_test,
        os.path.join(RESULTS_DIR, "calibration_curves_before.png")
    )
    print("\nBrier scores (lower is better; before calibration):")
    print(brier_scores)

    best_model_name = performance_table.loc["AUROC"].idxmax()
    print(f"\nBest model overall by AUROC: {best_model_name}")

    # Post-hoc calibration for the best model (if it produces probabilities)
    best_pipeline = trained_models[best_model_name]["pipeline"]
    if hasattr(best_pipeline, "predict_proba"):
        print(f"\nApplying post-hoc calibration (Platt scaling) to {best_model_name} ...")
        calibrated_best = calibrate_model(best_pipeline, X_train, y_train)
        trained_models[best_model_name]["calibrated_pipeline"] = calibrated_best

    # ------------------------------------------------------------------
    # 6. Fairness / subgroup audit (best model, on the test set)
    # ------------------------------------------------------------------
    print("\n[6/8] Auditing subgroup performance for the best model ...")
    sex_test = df.loc[X_test.index, "RIAGENDR"]
    race_test = df.loc[X_test.index, "RIDRETH3"]
    age_test = df.loc[X_test.index, "RIDAGEYR"]

    sex_audit = audit_by_sex(best_pipeline, X_test, y_test, sex_test)
    race_audit = audit_by_race_ethnicity(best_pipeline, X_test, y_test, race_test)
    age_audit = audit_by_age_band(best_pipeline, X_test, y_test, age_test)

    print("\n--- Performance by sex ---")
    print(sex_audit)
    print("\n--- Performance by race/ethnicity ---")
    print(race_audit)
    print("\n--- Performance by age band ---")
    print(age_audit)

    sex_audit.to_csv(os.path.join(RESULTS_DIR, "fairness_audit_by_sex.csv"))
    race_audit.to_csv(os.path.join(RESULTS_DIR, "fairness_audit_by_race_ethnicity.csv"))
    age_audit.to_csv(os.path.join(RESULTS_DIR, "fairness_audit_by_age_band.csv"))

    # ------------------------------------------------------------------
    # 7. Interpretability: SHAP + LIME for the best TREE-BASED model
    # ------------------------------------------------------------------
    print("\n[7/8] Interpreting the best tree-based model ...")
    best_tree_model_name = pick_best_tree_model(performance_table)
    tree_pipeline = trained_models[best_tree_model_name]["pipeline"]
    print(f"Best tree-based model: {best_tree_model_name}")

    shap_values, X_shap_sample = compute_shap_values(tree_pipeline, X_test)
    plot_shap_summary(
        shap_values, X_shap_sample, best_tree_model_name,
        os.path.join(RESULTS_DIR, "shap_summary_plot.png")
    )
    top_features = top_shap_features(shap_values, X_shap_sample)
    print("\nTop 10 variables by SHAP impact:")
    print(top_features)

    explain_individual_cases(tree_pipeline, X_train, X_test, y_test, output_dir=RESULTS_DIR)

    # ------------------------------------------------------------------
    # 8. Save model artifacts (for the Streamlit app) + summary
    # ------------------------------------------------------------------
    print("\n[8/8] Saving model artifacts and summary ...")
    artifact = {
        "pipeline": best_pipeline,
        "calibrated_pipeline": trained_models[best_model_name].get("calibrated_pipeline"),
        "model_name": best_model_name,
        "selected_features": selected_features,
        "X_train_sample": X_train.sample(n=min(200, len(X_train)), random_state=42),
    }
    with open(os.path.join(MODELS_DIR, "best_model_artifact.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    summary_lines = [
        "NHANES CVD RISK PREDICTION -- RESULTS SUMMARY",
        "=" * 50,
        "",
        f"Analytic sample size: {len(df):,} adults",
        f"Selected features ({len(selected_features)}): {', '.join(selected_features)}",
        "",
        "MODEL SCORECARD:",
        performance_table.to_string(),
        "",
        "BOOTSTRAPPED AUROC 95% CONFIDENCE INTERVALS:",
        ci_table.to_string(),
        "",
        f"Best model overall: {best_model_name}",
        f"Best tree-based model (used for SHAP/LIME): {best_tree_model_name}",
        "",
        "TOP 10 SHAP FEATURES:",
        top_features.to_string(),
        "",
        "FAIRNESS AUDIT -- BY SEX:",
        sex_audit.to_string(),
        "",
        "FAIRNESS AUDIT -- BY RACE/ETHNICITY:",
        race_audit.to_string(),
        "",
        "FAIRNESS AUDIT -- BY AGE BAND:",
        age_audit.to_string(),
    ]
    with open(os.path.join(RESULTS_DIR, "results_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "=" * 80)
    print(f"ALL DONE. Check the '{RESULTS_DIR}' folder for every table and plot.")
    print("=" * 80)


if __name__ == "__main__":
    main()
