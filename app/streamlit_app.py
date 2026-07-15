"""
streamlit_app.py
================
Interactive CVD risk calculator. Enter a hypothetical person's clinical
and dietary values and see:
  1. Their predicted CVD risk (using the calibrated model, so the
     percentage shown is a genuinely calibrated probability -- see
     evaluation.py / the model card for why this matters)
  2. A SHAP waterfall-style explanation of what pushed THEIR risk up or
     down, personalized to the exact numbers they entered

Run with:
    streamlit run app/streamlit_app.py

IMPORTANT: this is a portfolio/demonstration tool, not a medical device.
See the prominent disclaimer in the app itself and in the model card.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure "src" is importable regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import READABLE_NAMES

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(APP_DIR, "..", "results", "models", "best_model_artifact.pkl")

# Reasonable input ranges for each feature, derived from the training
# data's observed range. (min, max, default, step, unit_label)
FEATURE_INPUT_CONFIG = {
    "RIDAGEYR": (18, 85, 50, 1, "years"),
    "BMXBMI": (15.0, 60.0, 27.0, 0.5, "kg/m^2"),
    "BMXWAIST": (60.0, 165.0, 95.0, 1.0, "cm"),
    "SBP1": (90, 220, 120, 1, "mmHg"),
    "DBP1": (40, 120, 78, 1, "mmHg"),
    "LBXTC": (80, 300, 190, 1, "mg/dL"),
    "LBXHSCRP": (0.0, 35.0, 2.0, 0.1, "mg/L"),
    "DR1TPROT": (0, 260, 75, 1, "g"),
    "DR1TCARB": (0, 750, 220, 5, "g"),
    "DR1TSUGR": (0, 275, 85, 1, "g"),
    "DR1TBCAR": (0, 30000, 2000, 100, "mcg"),
    "DR1TVB6": (0.0, 9.0, 2.0, 0.1, "mg"),
    "DR1TFF": (0, 850, 200, 5, "mcg"),
    "DR1TVC": (0, 370, 80, 5, "mg"),
    "DR1TVK": (0, 2300, 120, 10, "mcg"),
    "DR1TCALC": (50, 3200, 870, 10, "mg"),
    "DR1TMAGN": (50, 950, 280, 5, "mg"),
    "DR1TCOPP": (0.0, 4.0, 1.1, 0.05, "mg"),
    "DR1TPOTA": (50, 6000, 2450, 25, "mg"),
    "DR1TMOIS": (250, 13800, 2650, 50, "g"),
}

FEATURE_GROUPS = {
    "Clinical Measurements": ["RIDAGEYR", "BMXBMI", "BMXWAIST", "SBP1", "DBP1", "LBXTC", "LBXHSCRP"],
    "Dietary Intake (from a 24-hour recall)": [
        "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TBCAR", "DR1TVB6", "DR1TFF",
        "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TMAGN", "DR1TCOPP", "DR1TPOTA", "DR1TMOIS",
    ],
}


@st.cache_resource
def load_artifact():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_shap_explainer(_pipeline):
    """Cached so we don't rebuild the SHAP explainer on every interaction.
    Leading underscore on the parameter tells Streamlit's cache not to
    try to hash the (unhashable) pipeline object itself."""
    tree_model = _pipeline.named_steps["clf"]
    return shap.TreeExplainer(tree_model)


def predict_risk(artifact, input_values: dict):
    """Run the calibrated pipeline on one person's input values and
    return the predicted probability of CVD. Falls back to the
    uncalibrated pipeline if no calibrated version was saved."""
    features = artifact["selected_features"]
    X_input = pd.DataFrame([[input_values[f] for f in features]], columns=features)

    pipeline = artifact.get("calibrated_pipeline") or artifact["pipeline"]
    probability = pipeline.predict_proba(X_input)[:, 1][0]
    return probability, X_input


def explain_prediction(artifact, X_input: pd.DataFrame):
    """Return SHAP values explaining this single prediction."""
    tree_pipeline = artifact["pipeline"]  # SHAP needs the raw tree model
    explainer = get_shap_explainer(tree_pipeline)
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values[0], explainer.expected_value


def plot_individual_explanation(shap_row, base_value, X_input, feature_names_readable):
    """A simple horizontal bar chart of this person's SHAP contributions,
    sorted by magnitude -- deliberately kept simple/dependency-light
    rather than using shap's own waterfall plot renderer."""
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]

    order = np.argsort(np.abs(shap_row))[::-1]
    sorted_values = shap_row[order]
    sorted_names = [feature_names_readable[i] for i in order]

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in sorted_values]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(sorted_values)), sorted_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(sorted_values)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to predicted risk (SHAP value)")
    ax.set_title("What's driving this person's predicted risk?")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="NHANES CVD Risk Calculator", layout="wide")

    st.title("Cardiovascular Disease Risk Calculator")
    st.caption(
        "A portfolio project built on NHANES 2017-2023 data, inspired by "
        "Ahiduzzaman & Hasan (2025), PLoS One."
    )

    st.warning(
        "**This is a demonstration / portfolio project, not a medical "
        "device or diagnostic tool.** It was trained on a research dataset "
        "with known limitations (see the model card in this repository) "
        "and should never be used to make real health decisions. If you "
        "have concerns about your cardiovascular health, please talk to a "
        "doctor.",
        icon="⚠️",
    )

    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error(
            "No trained model found yet. Run `python run_analysis.py` "
            "first to train a model and generate the artifact this app "
            "needs (results/models/best_model_artifact.pkl)."
        )
        return

    st.sidebar.header("Enter the person's information")
    input_values = {}

    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in artifact["selected_features"]]
        if not available:
            continue
        st.sidebar.subheader(group_name)
        for feature in available:
            min_v, max_v, default_v, step_v, unit = FEATURE_INPUT_CONFIG[feature]
            label = f"{READABLE_NAMES.get(feature, feature)}"
            input_values[feature] = st.sidebar.slider(
                label, min_value=min_v, max_value=max_v, value=default_v, step=step_v
            )

    # Any selected feature without a configured slider (shouldn't normally
    # happen, but keeps the app from crashing if selected_features ever
    # includes something not in FEATURE_INPUT_CONFIG) defaults to 0.
    for feature in artifact["selected_features"]:
        if feature not in input_values:
            input_values[feature] = 0

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Predicted Risk")
        probability, X_input = predict_risk(artifact, input_values)
        st.metric("Predicted probability of CVD", f"{probability * 100:.1f}%")
        st.caption(
            f"Model used: {artifact['model_name']} "
            f"(calibrated with Platt scaling so this percentage reflects "
            f"a genuine probability, not just a ranking score)."
        )

        if probability > 0.5:
            st.error("Predicted higher-than-average risk category.")
        elif probability > 0.2:
            st.warning("Predicted moderate risk category.")
        else:
            st.success("Predicted lower-than-average risk category.")

    with col2:
        st.subheader("Why did the model predict this?")
        shap_row, base_value = explain_prediction(artifact, X_input)
        readable_names = [READABLE_NAMES.get(f, f) for f in artifact["selected_features"]]
        fig = plot_individual_explanation(shap_row, base_value, X_input, readable_names)
        st.pyplot(fig)
        st.caption(
            "Red bars push the predicted risk up; green bars push it down. "
            "This explanation is specific to the exact numbers entered on "
            "the left."
        )

    st.divider()
    st.caption(
        "Data source: National Health and Nutrition Examination Survey "
        "(NHANES) 2017-2023, National Center for Health Statistics. "
        "This tool is for educational/portfolio purposes only."
    )


if __name__ == "__main__":
    main()
