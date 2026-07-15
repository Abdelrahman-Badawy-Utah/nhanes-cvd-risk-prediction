"""
interpret.py
============
SHAP (global) and LIME (local, per-person) interpretability for the
best-performing tree-based model.

UNLIKE THE PAPER-REPLICATION VERSION OF THIS PROJECT: there is no forced
inclusion of any particular variable here. Whatever the leakage-safe RFE
step (see feature_selection.py) decides to keep is what gets interpreted
-- if a variable doesn't make the cut, that's treated as a genuine
finding about this dataset, not something to override.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

from src.config import to_readable, RANDOM_SEED

TREE_BASED_MODELS = ["Random Forest", "XGBoost", "LightGBM"]


def pick_best_tree_model(performance_table: pd.DataFrame) -> str:
    """Pick the best-performing TREE-BASED model by AUROC. Restricting
    interpretability to tree models lets us use SHAP's fast, exact
    TreeExplainer instead of the much slower, approximate KernelExplainer
    needed for Logistic Regression / SVM."""
    available = [m for m in TREE_BASED_MODELS if m in performance_table.columns]
    return performance_table.loc["AUROC", available].idxmax()


def _extract_shap_values_for_positive_class(shap_values):
    """Different tree model / SHAP version combinations hand back SHAP
    values in slightly different shapes for binary classification. This
    normalizes all of them down to one number per (person, variable) --
    the contribution toward predicting 'Has CVD'."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def compute_shap_values(pipeline, X_test, sample_size: int = 500):
    """Compute SHAP values for a random sample of the test set (a sample
    keeps this fast on larger test sets while still giving a
    representative global picture)."""
    tree_model = pipeline.named_steps["clf"]
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(tree_model)
    raw_shap_values = explainer.shap_values(X_sample)
    shap_values = _extract_shap_values_for_positive_class(raw_shap_values)

    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, model_name: str, output_path: str):
    """Save a SHAP beeswarm summary plot with plain-English variable
    names."""
    X_sample_readable = X_sample.copy()
    X_sample_readable.columns = to_readable(X_sample_readable.columns)

    plt.figure()
    shap.summary_plot(shap_values, X_sample_readable, show=False)
    plt.title(f"SHAP Summary -- What Drives {model_name}'s Predictions?")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_shap_features(shap_values, X_sample, n: int = 10) -> pd.Series:
    """Return the top-N variables by mean absolute SHAP impact, with
    plain-English names."""
    readable_cols = to_readable(X_sample.columns)
    importance = pd.DataFrame(np.abs(shap_values), columns=readable_cols).mean()
    return importance.sort_values(ascending=False).head(n)


def explain_individual_cases(pipeline, X_train, X_test, y_test,
                              n_examples: int = 3, output_dir: str = "results"):
    """Generate LIME explanations for a handful of test-set individuals
    and save each as its own plot."""
    tree_model = pipeline.named_steps["clf"]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=to_readable(X_train.columns),
        class_names=["No CVD", "Has CVD"],
        mode="classification",
        random_state=RANDOM_SEED,
    )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for i in range(n_examples):
        instance = X_test.iloc[i].values
        true_label = "Has CVD" if y_test.iloc[i] == 1 else "No CVD"

        explanation = lime_explainer.explain_instance(
            instance, tree_model.predict_proba, num_features=8
        )
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f"Example person #{i+1}  (their real status: {true_label})", y=1.05)
        path = os.path.join(output_dir, f"lime_example_case_{i+1}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
