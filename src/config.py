"""
config.py
=========
Central place for every "settings" value used across the project: file
paths, the candidate predictor variables, and the human-readable variable
name lookup. Every other module imports from here so there is exactly ONE
place to change a setting.
"""

import os

# ------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------
# Change RAW_DATA_PATH to point at your NHANES extract before running
# run_analysis.py.
RAW_DATA_PATH = "PUT_YOUR_FILE_PATH_HERE.xlsx"

RESULTS_DIR = "results"
MODELS_DIR = "results/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42

# Number of cross-validation folds used during hyperparameter search.
# Set to 2 to keep total runtime reasonable on modest/single-core
# hardware. If you have more compute available, bumping this to 5 (the
# more conventional choice) will give more stable hyperparameter
# estimates at the cost of roughly 2-3x longer runtime.
CV_FOLDS = 2

# ------------------------------------------------------------------------
# CANDIDATE PREDICTOR VARIABLES
# ------------------------------------------------------------------------
DIETARY_VARS = [
    "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TSFAT", "DR1TMFAT",
    "DR1TPFAT", "DR1TCHOL", "DR1TBCAR", "DR1TCRYP", "DR1TLYCO", "DR1TTHEO",
    "DR1TNIAC", "DR1TVB6", "DR1TFOLA", "DR1TFF", "DR1TIRON", "DR1TVB12",
    "DR1TVC", "DR1TVD", "DR1TATOC", "DR1TVK", "DR1TCALC", "DR1TPHOS",
    "DR1TMAGN", "DR1TZINC", "DR1TCOPP", "DR1TSODI", "DR1TPOTA", "DR1TSELE",
    "DR1TMOIS",
]

CLINICAL_VARS = ["BMXBMI", "BMXWAIST", "SBP1", "DBP1", "LBXTC", "LBXHSCRP"]

DEMOGRAPHIC_VARS = ["RIDAGEYR"]

ALL_PREDICTORS = DIETARY_VARS + CLINICAL_VARS + DEMOGRAPHIC_VARS

# Variables NEVER used as model predictors, but used afterward purely to
# AUDIT the model for fairness across subgroups. Keeping these out of the
# predictor list is a deliberate choice, discussed in the README/model
# card: we don't want the model directly conditioning its risk score on
# sex or race/ethnicity, but we absolutely want to check whether its
# errors are unevenly distributed across these groups.
FAIRNESS_AUDIT_VARS = ["RIAGENDR", "RIDRETH3"]

OUTCOME_VAR = "cvd"

# ------------------------------------------------------------------------
# HUMAN-READABLE VARIABLE NAMES
# ------------------------------------------------------------------------
READABLE_NAMES = {
    "DR1TPROT": "Protein (g)",
    "DR1TCARB": "Carbohydrates (g)",
    "DR1TSUGR": "Total Sugars (g)",
    "DR1TFIBE": "Fiber (g)",
    "DR1TSFAT": "Saturated Fat (g)",
    "DR1TMFAT": "Monounsaturated Fat (g)",
    "DR1TPFAT": "Polyunsaturated Fat (g)",
    "DR1TCHOL": "Dietary Cholesterol (mg)",
    "DR1TBCAR": "Beta-Carotene (mcg)",
    "DR1TCRYP": "Beta-Cryptoxanthin (mcg)",
    "DR1TLYCO": "Lycopene (mcg)",
    "DR1TTHEO": "Theobromine (mg)",
    "DR1TNIAC": "Niacin / Vitamin B3 (mg)",
    "DR1TVB6": "Vitamin B6 (mg)",
    "DR1TFOLA": "Total Folate (mcg)",
    "DR1TFF": "Food Folate (mcg)",
    "DR1TIRON": "Iron (mg)",
    "DR1TVB12": "Vitamin B12 (mcg)",
    "DR1TVC": "Vitamin C (mg)",
    "DR1TVD": "Vitamin D (mcg)",
    "DR1TATOC": "Vitamin E (mg)",
    "DR1TVK": "Vitamin K (mcg)",
    "DR1TCALC": "Calcium (mg)",
    "DR1TPHOS": "Phosphorus (mg)",
    "DR1TMAGN": "Magnesium (mg)",
    "DR1TZINC": "Zinc (mg)",
    "DR1TCOPP": "Copper (mg)",
    "DR1TSODI": "Sodium (mg)",
    "DR1TPOTA": "Potassium (mg)",
    "DR1TSELE": "Selenium (mcg)",
    "DR1TMOIS": "Food Moisture / Water Content (g)",
    "BMXBMI": "Body Mass Index (BMI)",
    "BMXWAIST": "Waist Circumference (cm)",
    "SBP1": "Systolic Blood Pressure (mmHg)",
    "DBP1": "Diastolic Blood Pressure (mmHg)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBXHSCRP": "C-Reactive Protein (CRP, mg/L)",
    "RIDAGEYR": "Age (years)",
}


def to_readable(feature_list):
    """Translate NHANES variable codes into plain-English labels. Falls
    back to the raw code if a variable isn't in the dictionary."""
    return [READABLE_NAMES.get(f, f) for f in feature_list]


# NHANES coding reference for the two fairness-audit variables (used only
# to make audit tables/plots readable, never fed to the model):
SEX_LABELS = {1: "Male", 2: "Female"}
RACE_ETHNICITY_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multiracial",
}
