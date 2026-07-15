"""
modeling.py
===========
Train/test splitting, class-imbalance correction, and hyperparameter
tuning for all 5 models.

TWO DELIBERATE CORRECTNESS FIXES BAKED INTO THIS MODULE
--------------------------------------------------------
1. REAL hyperparameter tuning. Rather than copying literature-guided
   default settings (as the original paper did), every model here goes
   through a randomized search (RandomizedSearchCV) with cross-validation
   on the training data only.

2. Oversampling is done INSIDE each cross-validation fold, not once
   upfront. This matters more than it might look: RandomOverSampler
   works by duplicating existing minority-class rows. If you oversample
   the training data ONCE and then run cross-validation on that already-
   oversampled data, an exact duplicate of the same person can land in
   both the training portion and the held-out validation portion of a
   fold -- letting the model "cheat" by having effectively already seen
   the validation example. This inflates cross-validated performance
   estimates in a way that doesn't reflect real-world generalization
   (we measured this directly during development: CV AUROC came out
   at an implausible 0.99 before this fix). Wrapping the oversampler in
   an imblearn Pipeline means a fresh, independent oversampling happens
   inside each fold, using only that fold's training rows.
"""

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import RANDOM_SEED, CV_FOLDS

warnings.filterwarnings("ignore")


def split_data(X, y, test_size: float = 0.20):
    """80/20 stratified train/test split. The test set returned here is
    never touched again until final evaluation, and is never oversampled."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )


# ---------------------------------------------------------------------------
# Hyperparameter search spaces. Parameter names are prefixed with "clf__"
# because each model is wrapped in a Pipeline (see build_pipeline below),
# and scikit-learn requires that prefix to route settings to the
# classifier step specifically.
# ---------------------------------------------------------------------------
_RAW_PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [3000],
    },
    "Random Forest": {
        "n_estimators": [100, 150, 250],
        "max_depth": [6, 10, 14],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2"],
    },
    "SVM": {
        "C": [0.1, 1, 5],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    },
    "XGBoost": {
        "n_estimators": [150, 250, 350],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [150, 250, 350],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
    },
}

PARAM_DISTRIBUTIONS = {
    model_name: {f"clf__{k}": v for k, v in params.items()}
    for model_name, params in _RAW_PARAM_DISTRIBUTIONS.items()
}

BASE_ESTIMATORS = {
    "Logistic Regression": lambda: LogisticRegression(),
    "Random Forest": lambda: RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
    "SVM": lambda: SVC(probability=False, random_state=RANDOM_SEED, cache_size=1000),
    "XGBoost": lambda: XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_SEED, use_label_encoder=False, n_jobs=1
    ),
    "LightGBM": lambda: LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_jobs=1),
}

# Fewer search iterations for the slower models to keep total runtime
# reasonable on modest hardware (this project was developed/tested on a
# single-core machine).
N_ITER_BY_MODEL = {
    "Logistic Regression": 5,
    "Random Forest": 6,
    "SVM": 5,
    "XGBoost": 8,
    "LightGBM": 8,
}

USES_SCALED_DATA = {
    "Logistic Regression": True,
    "Random Forest": False,
    "SVM": True,
    "XGBoost": False,
    "LightGBM": False,
}

# SVM's RBF kernel scales poorly with sample size (roughly quadratic),
# making a full hyperparameter search on the full training set
# impractically slow. Standard practice for kernel methods is to search
# on a representative subsample, then refit the winning configuration on
# the full training set. Other models are fast enough not to need this.
SEARCH_SUBSAMPLE_SIZE = {
    "SVM": 1500,
}


def build_pipeline(model_name: str) -> ImbPipeline:
    """Build a Pipeline where oversampling (and scaling, if needed) happen
    as explicit PIPELINE STEPS rather than as a one-time preprocessing
    step. This is what makes it safe to run cross-validation over this
    pipeline: scikit-learn/imblearn automatically re-fit every step
    (including the oversampler and scaler) fresh within each fold, using
    only that fold's training rows."""
    steps = [("oversample", RandomOverSampler(random_state=RANDOM_SEED))]
    if USES_SCALED_DATA[model_name]:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", BASE_ESTIMATORS[model_name]()))
    return ImbPipeline(steps)


def tune_and_train_all_models(X_train, y_train, verbose: bool = True):
    """Run RandomizedSearchCV for each of the 5 models on the TRAINING
    data only (oversampling happens fresh inside each CV fold via the
    pipeline -- see module docstring), then refit each pipeline's winning
    configuration on the full training set.

    Note this function takes the RAW (not oversampled, not scaled)
    training data -- the pipeline itself handles both of those steps
    internally, for every model, every fold, every final fit.

    Returns
    -------
    dict of {model_name: {"pipeline": fitted Pipeline,
                           "uses_scaled_data": bool,
                           "best_params": dict,
                           "cv_auroc": float}}
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    trained_models = {}

    for model_name in BASE_ESTIMATORS:
        if verbose:
            print(f"[modeling] Tuning {model_name} "
                  f"({N_ITER_BY_MODEL[model_name]} random search iterations, "
                  f"{CV_FOLDS}-fold CV, oversampling done fresh inside each fold) ...")

        # For slow models, search on a random subsample of the ORIGINAL
        # (still-imbalanced) training data, then refit the winning
        # pipeline on the full training set.
        if model_name in SEARCH_SUBSAMPLE_SIZE:
            n_sub = min(SEARCH_SUBSAMPLE_SIZE[model_name], len(X_train))
            sub_idx = X_train.sample(n=n_sub, random_state=RANDOM_SEED).index
            X_search, y_search = X_train.loc[sub_idx], y_train.loc[sub_idx]
            if verbose:
                print(f"  (Searching on a {n_sub:,}-row subsample for speed; "
                      f"final pipeline is refit on the full training set.)")
        else:
            X_search, y_search = X_train, y_train

        search = RandomizedSearchCV(
            estimator=build_pipeline(model_name),
            param_distributions=PARAM_DISTRIBUTIONS[model_name],
            n_iter=N_ITER_BY_MODEL[model_name],
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        search.fit(X_search, y_search)

        if verbose:
            print(f"  Best CV AUROC: {search.best_score_:.4f}")
            print(f"  Best params: {search.best_params_}")

        # Refit a fresh pipeline with the winning hyperparameters on the
        # FULL (raw, imbalanced) training set. The pipeline's own
        # "oversample" step balances it internally.
        final_pipeline = build_pipeline(model_name)
        final_pipeline.set_params(**search.best_params_)
        final_pipeline.fit(X_train, y_train)

        trained_models[model_name] = {
            "pipeline": final_pipeline,
            "uses_scaled_data": USES_SCALED_DATA[model_name],
            "best_params": search.best_params_,
            "cv_auroc": search.best_score_,
        }

    return trained_models
