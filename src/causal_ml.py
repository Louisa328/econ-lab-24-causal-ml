"""
causal_ml.py
============
Reusable causal inference utilities for Double Machine Learning (DML)
and Causal Forest CATE estimation.

Lab 24 — ECON 5200: Causal Machine Learning & Applied Analytics
Author: Yun
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from typing import Tuple, Optional


# =============================================================================
# Part 1: Manual DML
# =============================================================================

def manual_dml(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_splits: int = 2,
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
) -> float:
    """
    Estimate the Average Treatment Effect (ATE) using manual cross-fitted
    Double Machine Learning (DML) with Random Forest nuisance learners.

    Implements the Frisch-Waugh-Lovell IV-style estimator:

        theta = sum(D_tilde * Y_tilde) / sum(D_tilde^2)

    where Y_tilde = Y - E[Y|X] and D_tilde = D - E[D|X] are computed
    via cross-fitting to avoid Donsker-class bias.

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Outcome variable (continuous).
    D : np.ndarray, shape (n,)
        Treatment variable (binary 0/1 or continuous).
    X : np.ndarray, shape (n, p)
        Covariate matrix. Must not contain Y or D.
    n_splits : int, default=2
        Number of cross-fitting folds. Must be >= 2.
    n_estimators : int, default=200
        Number of trees in each Random Forest nuisance model.
    max_depth : int, default=5
        Maximum tree depth for Random Forest nuisance models.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    float
        Point estimate of the ATE (theta_hat).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n, p = 1000, 10
    >>> X = np.random.normal(size=(n, p))
    >>> D = (X[:, 0] > 0).astype(float)
    >>> Y = 3.0 * D + X[:, 0] + np.random.normal(size=n)
    >>> manual_dml(Y, D, X)  # should be close to 3.0
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    Y_tilde = np.zeros(n)
    D_tilde = np.zeros(n)

    for train_idx, test_idx in kf.split(X):

        # --- Outcome nuisance: E[Y | X] ---
        ml_l = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        ml_l.fit(X[train_idx], Y[train_idx])
        Y_tilde[test_idx] = Y[test_idx] - ml_l.predict(X[test_idx])

        # --- Treatment nuisance: E[D | X] ---
        # Use Classifier for binary D, Regressor for continuous D
        if np.isin(D, [0, 1]).all():
            ml_m = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            ml_m.fit(X[train_idx], D[train_idx])
            D_hat = ml_m.predict_proba(X[test_idx])[:, 1]
        else:
            ml_m = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            ml_m.fit(X[train_idx], D[train_idx])
            D_hat = ml_m.predict(X[test_idx])

        D_tilde[test_idx] = D[test_idx] - D_hat

    # Frisch-Waugh-Lovell IV-style estimator
    theta_hat = np.sum(D_tilde * Y_tilde) / np.sum(D_tilde ** 2)

    return float(theta_hat)


# =============================================================================
# Part 2: CATE by Subgroup
# =============================================================================

def cate_by_subgroup(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    subgroup_col: np.ndarray,
    subgroup_labels: Optional[list] = None,
    n_splits: int = 2,
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Estimate subgroup-level Conditional Average Treatment Effects (CATEs)
    using separate DML regressions within each subgroup.

    For each unique value in `subgroup_col`, fits a cross-fitted DML model
    on the subset of observations belonging to that subgroup and returns
    the within-subgroup ATE as an estimate of the CATE for that group.

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Outcome variable (continuous).
    D : np.ndarray, shape (n,)
        Treatment variable (binary 0/1 or continuous).
    X : np.ndarray, shape (n, p)
        Covariate matrix. Must not contain Y or D.
    subgroup_col : np.ndarray, shape (n,)
        Array of subgroup labels for each observation (e.g. income quartile).
    subgroup_labels : list, optional
        Ordered list of subgroup labels to display. If None, uses
        sorted unique values of subgroup_col.
    n_splits : int, default=2
        Number of cross-fitting folds within each subgroup.
    n_estimators : int, default=200
        Number of trees in each Random Forest nuisance model.
    max_depth : int, default=5
        Maximum tree depth for Random Forest nuisance models.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by subgroup label with columns:
        - cate      : subgroup ATE estimate
        - n_obs     : number of observations in subgroup

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> n, p = 2000, 5
    >>> X = np.random.normal(size=(n, p))
    >>> D = (X[:, 0] > 0).astype(float)
    >>> Y = (3.0 + 2.0 * (X[:, 1] > 0)) * D + X[:, 0] + np.random.normal(size=n)
    >>> groups = np.where(X[:, 1] > 0, 'High', 'Low')
    >>> cate_by_subgroup(Y, D, X, groups)
    """
    if subgroup_labels is None:
        subgroup_labels = sorted(np.unique(subgroup_col))

    results = []

    for label in subgroup_labels:
        mask = subgroup_col == label
        n_obs = mask.sum()

        if n_obs < 50:
            # Too few observations for reliable cross-fitting
            results.append({
                "subgroup": label,
                "cate": np.nan,
                "n_obs": n_obs,
            })
            continue

        cate_est = manual_dml(
            Y=Y[mask],
            D=D[mask],
            X=X[mask],
            n_splits=n_splits,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        results.append({
            "subgroup": label,
            "cate": cate_est,
            "n_obs": int(n_obs),
        })

    df = pd.DataFrame(results).set_index("subgroup")

    return df
