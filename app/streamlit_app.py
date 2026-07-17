"""
streamlit_app.py
================
An interactive CVD risk calculator. Enter a hypothetical person's health,
diet, smoking, and diabetes status, and see:
  1. Their predicted CVD risk (a genuinely trustworthy percentage,
     thanks to the calibration fix explained in the notebook)
  2. A plain-English explanation of what pushed THEIR risk up or down

BEFORE RUNNING THIS APP: you must first run the notebook
(CVD_Risk_Prediction.ipynb) once, start to finish. The last cell of the
notebook saves the trained model to app/model/best_model_artifact.pkl --
this app simply loads that file and uses it to make live predictions.

Run with:
    streamlit run app/streamlit_app.py

IMPORTANT: this is a portfolio / demonstration project, not a medical
device. See the big warning banner in the app itself.
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(APP_DIR, "model", "best_model_artifact.pkl")

READABLE_NAMES = {
    "DR1TPROT": "Protein (g)", "DR1TCARB": "Carbohydrates (g)", "DR1TSUGR": "Total Sugars (g)",
    "DR1TFIBE": "Fiber (g)", "DR1TSFAT": "Saturated Fat (g)", "DR1TMFAT": "Monounsaturated Fat (g)",
    "DR1TPFAT": "Polyunsaturated Fat (g)", "DR1TCHOL": "Dietary Cholesterol (mg)",
    "DR1TBCAR": "Beta-Carotene (mcg)", "DR1TCRYP": "Beta-Cryptoxanthin (mcg)", "DR1TLYCO": "Lycopene (mcg)",
    "DR1TVB1": "Thiamin / Vitamin B1 (mg)", "DR1TVB2": "Riboflavin / Vitamin B2 (mg)",
    "DR1TNIAC": "Niacin / Vitamin B3 (mg)", "DR1TVB6": "Vitamin B6 (mg)",
    "DR1TFOLA": "Total Folate (mcg)", "DR1TFF": "Food Folate (mcg)", "DR1TIRON": "Iron (mg)",
    "DR1TCHL": "Total Choline (mg)", "DR1TVB12": "Vitamin B12 (mcg)", "DR1TVC": "Vitamin C (mg)",
    "DR1TVD": "Vitamin D (mcg)", "DR1TATOC": "Vitamin E (mg)", "DR1TVK": "Vitamin K (mcg)",
    "DR1TCALC": "Calcium (mg)", "DR1TPHOS": "Phosphorus (mg)", "DR1TMAGN": "Magnesium (mg)",
    "DR1TZINC": "Zinc (mg)", "DR1TCOPP": "Copper (mg)", "DR1TSODI": "Sodium (mg)",
    "DR1TPOTA": "Potassium (mg)", "DR1TSELE": "Selenium (mcg)", "DR1TMOIS": "Food Moisture / Water Content (g)",
    "BMXBMI": "Body Mass Index (BMI)", "BMXWAIST": "Waist Circumference (cm)",
    "SBP1": "Systolic Blood Pressure (mmHg)", "DBP1": "Diastolic Blood Pressure (mmHg)",
    "LBXTC": "Total Cholesterol (mg/dL)", "HDL": "HDL Cholesterol (mg/dL)",
    "LBXHSCRP": "C-Reactive Protein (CRP, mg/L)", "eGFR": "Estimated Kidney Function (eGFR)",
    "RIDAGEYR": "Age (years)",
    "smoking_status": "Smoking Status",
    "diabetes_status": "Diabetes Status",
}

# Reasonable slider ranges: (min, max, default, step)
NUMERIC_FEATURE_CONFIG = {
    "RIDAGEYR": (18, 85, 50, 1),
    "BMXBMI": (15.0, 60.0, 27.0, 0.5),
    "BMXWAIST": (60.0, 165.0, 95.0, 1.0),
    "SBP1": (90, 220, 120, 1),
    "DBP1": (40, 120, 78, 1),
    "LBXTC": (80, 300, 190, 1),
    "HDL": (10, 130, 53, 1),
    "LBXHSCRP": (0.0, 35.0, 2.0, 0.1),
    "eGFR": (5, 165, 95, 1),
    "DR1TPROT": (0, 260, 75, 1),
    "DR1TCARB": (0, 750, 220, 5),
    "DR1TSUGR": (0, 275, 85, 1),
    "DR1TCHOL": (0, 1000, 280, 5),
    "DR1TBCAR": (0, 30000, 2000, 100),
    "DR1TNIAC": (0.0, 100.0, 22.0, 1.0),
    "DR1TVB2": (0.0, 6.0, 1.9, 0.1),
    "DR1TVB6": (0.0, 9.0, 2.0, 0.1),
    "DR1TFF": (0, 850, 200, 5),
    "DR1TIRON": (0.0, 60.0, 15.0, 0.5),
    "DR1TVB12": (0.0, 30.0, 4.5, 0.1),
    "DR1TVC": (0, 370, 80, 5),
    "DR1TVK": (0, 2300, 120, 10),
    "DR1TCALC": (50, 3200, 870, 10),
    "DR1TMAGN": (50, 950, 280, 5),
    "DR1TCOPP": (0.0, 4.0, 1.1, 0.05),
    "DR1TSODI": (0, 8000, 3200, 50),
    "DR1TPOTA": (50, 6000, 2450, 25),
    "DR1TMOIS": (250, 13800, 2650, 50),
}

# Both smoking status and diabetes status are categorical, handled with
# selectboxes rather than sliders.
SMOKING_OPTIONS = {"Never smoked": 0, "Former smoker": 1, "Current smoker": 2}
DIABETES_OPTIONS = {"No": 0, "Yes": 1}

FEATURE_GROUPS = {
    "Clinical Measurements": [
        "RIDAGEYR", "BMXBMI", "BMXWAIST", "SBP1", "DBP1",
        "LBXTC", "HDL", "LBXHSCRP", "eGFR",
    ],
    "Smoking & Diabetes History": ["smoking_status", "diabetes_status"],
    "Diet (from a single day's recall)": [
        "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TCHOL", "DR1TBCAR", "DR1TNIAC", "DR1TVB2", "DR1TVB6",
        "DR1TFF", "DR1TIRON", "DR1TVB12", "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TMAGN", "DR1TCOPP",
        "DR1TSODI", "DR1TPOTA", "DR1TMOIS",
    ],
}


@st.cache_resource
def load_artifact():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_shap_explainer(_tree_model):
    """The leading underscore tells Streamlit not to try to hash the
    (unhashable) model object when deciding whether to reuse the cache."""
    return shap.TreeExplainer(_tree_model)


def predict_risk(artifact, input_values: dict):
    features = artifact["selected_features"]
    X_input = pd.DataFrame([[input_values[f] for f in features]], columns=features)
    pipeline = artifact.get("calibrated_pipeline") or artifact["pipeline"]
    probability = pipeline.predict_proba(X_input)[:, 1][0]
    return probability, X_input


def explain_prediction(artifact, X_input: pd.DataFrame):
    tree_model = artifact["pipeline"].named_steps["predict"]
    explainer = get_shap_explainer(tree_model)
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values[0]


def plot_individual_explanation(shap_row, feature_names_readable):
    order = np.argsort(np.abs(shap_row))[::-1]
    sorted_values = shap_row[order]
    sorted_names = [feature_names_readable[i] for i in order]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in sorted_values]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(sorted_values)), sorted_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(sorted_values)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Push toward higher risk (right) or lower risk (left)")
    ax.set_title("What's driving this prediction?")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="CVD Risk Calculator", layout="wide")

    st.title("Cardiovascular Disease Risk Calculator")
    st.caption(
        "A portfolio project built on public NHANES health survey data, "
        "inspired by Ahiduzzaman & Hasan (2025), PLoS One."
    )

    st.warning(
        "**This is a demonstration project, not a medical device.** It "
        "was trained on a research dataset with known limitations (see "
        "the model card in this repository) and should never be used to "
        "make real health decisions. If you have concerns about your "
        "cardiovascular health, please talk to a doctor.",
        icon="⚠️",
    )

    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error(
            "No trained model found yet. Open and run `CVD_Risk_Prediction.ipynb` "
            "from start to finish first -- its last step saves the model "
            "file this app needs (app/model/best_model_artifact.pkl)."
        )
        return

    st.sidebar.header("Enter a hypothetical person's information")
    input_values = {}

    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in artifact["selected_features"]]
        if not available:
            continue
        st.sidebar.subheader(group_name)
        for feature in available:
            if feature == "smoking_status":
                label = st.sidebar.selectbox("Smoking status", list(SMOKING_OPTIONS.keys()), index=0)
                input_values[feature] = SMOKING_OPTIONS[label]
            elif feature == "diabetes_status":
                label = st.sidebar.selectbox("Diabetes", list(DIABETES_OPTIONS.keys()), index=0)
                input_values[feature] = DIABETES_OPTIONS[label]
            else:
                min_v, max_v, default_v, step_v = NUMERIC_FEATURE_CONFIG[feature]
                label = READABLE_NAMES.get(feature, feature)
                input_values[feature] = st.sidebar.slider(
                    label, min_value=min_v, max_value=max_v, value=default_v, step=step_v
                )

    for feature in artifact["selected_features"]:
        if feature not in input_values:
            input_values[feature] = 0

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Predicted Risk")
        probability, X_input = predict_risk(artifact, input_values)
        st.metric("Predicted probability of CVD", f"{probability * 100:.1f}%")
        st.caption(f"Model used: {artifact['model_name']} (calibrated, so this is a trustworthy percentage).")

        if probability > 0.5:
            st.error("Higher-than-average predicted risk.")
        elif probability > 0.2:
            st.warning("Moderate predicted risk.")
        else:
            st.success("Lower-than-average predicted risk.")

    with col2:
        st.subheader("Why did the model predict this?")
        shap_row = explain_prediction(artifact, X_input)
        readable_names = [READABLE_NAMES.get(f, f) for f in artifact["selected_features"]]
        fig = plot_individual_explanation(shap_row, readable_names)
        st.pyplot(fig)
        st.caption(
            "Red bars push the predicted risk up; green bars push it down. "
            "This is personalized to the exact numbers entered on the left."
        )

    st.divider()
    st.caption(
        "Data source: National Health and Nutrition Examination Survey "
        "(NHANES) 2015-2023, National Center for Health Statistics. "
        "Educational/portfolio use only."
    )


if __name__ == "__main__":
    main()
