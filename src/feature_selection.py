"""
feature_selection.py
=====================
Recursive Feature Elimination (RFE) -- deciding which of the ~37
candidate variables actually help predict CVD.

A NOTE ON A COMMON MISTAKE THIS MODULE DELIBERATELY AVOIDS:
A subtle but real methodological error is running feature selection on
the ENTIRE dataset before splitting into training and test sets. Doing
that lets information from the test set indirectly influence which
variables are chosen -- a form of data leakage. It's an easy mistake to
make (a lot of published tutorials do exactly this), but it means your
test-set performance numbers are a little too optimistic, because the
test set had a small hand in shaping the model before ever being used to
evaluate it.

This module's `select_features()` function takes ONLY training data as
input, by design -- there is no code path that lets it see the test set.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from src.config import RANDOM_SEED


def select_features(X_train, y_train, n_features_to_select: int = 20,
                     step: int = 2, verbose: bool = True):
    """Run RFE using ONLY the training data and return the list of
    variables it kept.

    Parameters
    ----------
    X_train, y_train : the TRAINING split only. Never pass test data here.
    n_features_to_select : how many variables to keep.
    step : how many variables to eliminate at each iteration (larger =
        faster but coarser).

    Returns
    -------
    list of str : the selected column names, in their original order.
    """
    if verbose:
        print(f"[feature_selection] Running RFE on {X_train.shape[1]} "
              f"candidate variables using ONLY the training data "
              f"({len(X_train):,} rows) ...")

    estimator = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
    )
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step,
    )
    selector.fit(X_train, y_train)

    selected = X_train.columns[selector.support_].tolist()

    if verbose:
        print(f"[feature_selection] Kept {len(selected)} variables:")
        for feat in selected:
            print(f"   - {feat}")

    return selected
