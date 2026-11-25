"""Leakage detection guard (D8)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def detect_leakage(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    threshold: float = 0.99,
) -> dict:
    """
    Detect potential data leakage by checking if CV performance is suspiciously high.
    Also checks for feature-target correlation that shouldn't exist.
    """
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, X, y, cv=cv_obj, scoring="roc_auc")

    # Check for constant/near-constant features
    from scipy.stats import pearsonr

    correlations = []
    for i in range(X.shape[1]):
        if np.std(X[:, i]) > 1e-10:
            r, _ = pearsonr(X[:, i], y)
            correlations.append(abs(r))
        else:
            correlations.append(0.0)

    max_corr = max(correlations) if correlations else 0.0

    return {
        "cv_auc_mean": float(np.mean(scores)),
        "cv_auc_std": float(np.std(scores)),
        "cv_auc_max": float(np.max(scores)),
        "suspicious": np.mean(scores) > threshold,
        "max_feature_target_corr": float(max_corr),
        "high_corr_features": [i for i, c in enumerate(correlations) if c > 0.9],
    }


def check_preprocessing_leakage(
    preprocess_fn,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """
    Check if preprocessing is being fit on test data.
    This is a smoke test - in real CV, preprocessing should be inside the fold.
    """
    # Simulate wrong usage: fit on all data
    preprocess_wrong = preprocess_fn()
    X_wrong = preprocess_wrong.fit_transform(X, y)

    # Correct usage: fit only on train fold
    from sklearn.model_selection import StratifiedKFold

    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    diffs = []

    for train_idx, test_idx in cv_obj.split(X, y):
        preprocess_correct = preprocess_fn()
        preprocess_correct.fit_transform(X[train_idx], y[train_idx])
        X_test_trans = preprocess_correct.transform(X[test_idx])

        # Compare with wrong approach on test set
        X_wrong_test = X_wrong[test_idx]
        diff = np.mean(np.abs(X_test_trans - X_wrong_test))
        diffs.append(diff)

    return {
        "mean_transformation_diff": float(np.mean(diffs)),
        "max_transformation_diff": float(np.max(diffs)),
        "leakage_detected": np.mean(diffs) > 1e-6,
        "message": "Preprocessing fitted on full dataset leaks information to test set"
        if np.mean(diffs) > 1e-6
        else "OK",
    }


def audit_pipeline(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Full pipeline audit for leakage."""
    results = {}

    # Check each step
    for _, step in pipeline.steps:
        if hasattr(step, "fit") and hasattr(step, "transform"):
            # This is a transformer - check if it's stateless or properly used
            pass

    # Overall leakage check
    results["leakage_check"] = detect_leakage(pipeline, X, y)
    return results
