"""Nested cross-validation (D8)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from qednet.evaluation.calibration import calibrate_model, evaluate_calibration
from qednet.evaluation.metrics import compute_flat_metrics as compute_all_metrics
from qednet.evaluation.statistics import bootstrap_ci, delong_roc_test, paired_t_test


def nested_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    preprocess_fn,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
    calibrate: bool = True,
    calibration_method: str = "platt",
    n_bootstrap: int = 2000,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Stratified outer cross-validation with fold-local preprocessing.
    This function does not perform hyperparameter search. When calibration is
    enabled, ``inner_folds`` controls calibration folds on the training portion.
    Returns aggregated metrics and per-fold results.
    """
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have the same number of rows as X and y")
        if calibrate:
            raise ValueError(
                "Grouped CV with calibration is unsupported: calibration must also be group-aware"
            )
        outer_cv = StratifiedGroupKFold(
            n_splits=outer_folds, shuffle=True, random_state=random_state
        )
        splits = outer_cv.split(X, y, groups)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        splits = outer_cv.split(X, y)

    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocess within fold
        preprocess = preprocess_fn()
        X_train_trans = preprocess.fit_transform(X_train, y_train)
        X_test_trans = preprocess.transform(X_test)

        # Fit only on the outer training fold; preprocessing has not seen test data.
        model_fold = clone(model)
        model_fold.fit(X_train_trans, y_train)

        # Calibrate on inner fold training data
        if calibrate:
            model_fold = calibrate_model(
                model_fold, X_train_trans, y_train, method=calibration_method, cv=inner_folds
            )

        y_prob = model_fold.predict_proba(X_test_trans)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(y_test, y_prob, y_pred)
        cal = evaluate_calibration(y_test, y_prob)

        fold_results.append(
            {
                "fold": fold_idx,
                "metrics": metrics,
                "calibration": cal,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    # Aggregate metrics with bootstrap CIs
    agg_metrics = {}
    for metric_name in fold_results[0]["metrics"]:
        point, lower, upper = bootstrap_ci(
            all_y_true,
            all_y_prob,
            lambda yt, yp, name=metric_name: compute_all_metrics(yt, yp)[name],
            n_bootstrap=n_bootstrap,
        )
        agg_metrics[metric_name] = {
            "mean": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "per_fold": [r["metrics"][metric_name] for r in fold_results],
        }

    # Overall calibration
    overall_cal = evaluate_calibration(all_y_true, all_y_prob)

    return {
        "fold_results": fold_results,
        "aggregated_metrics": agg_metrics,
        "overall_calibration": overall_cal,
        "all_predictions": {
            "y_true": all_y_true,
            "y_prob": all_y_prob,
            "y_pred": all_y_pred,
        },
    }


def compare_models(
    model1,
    model2,
    X: np.ndarray,
    y: np.ndarray,
    preprocess_fn,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> dict:
    """Compare two models using nested CV and statistical tests."""
    result1 = nested_cv(
        model1,
        X,
        y,
        preprocess_fn,
        outer_folds,
        inner_folds,
        random_state,
        calibrate=groups is None,
        groups=groups,
    )
    result2 = nested_cv(
        model2,
        X,
        y,
        preprocess_fn,
        outer_folds,
        inner_folds,
        random_state,
        calibrate=groups is None,
        groups=groups,
    )

    y_true = result1["all_predictions"]["y_true"]
    prob1 = result1["all_predictions"]["y_prob"]
    prob2 = result2["all_predictions"]["y_prob"]

    # Per-fold metric differences
    folds1 = np.array([r["metrics"]["auroc"] for r in result1["fold_results"]])
    folds2 = np.array([r["metrics"]["auroc"] for r in result2["fold_results"]])

    # Statistical tests
    delong = delong_roc_test(y_true, prob1, prob2)
    paired = paired_t_test(folds1, folds2)

    return {
        "model1": result1["aggregated_metrics"],
        "model2": result2["aggregated_metrics"],
        "delong_test": delong,
        "paired_t_test": paired,
        "per_fold_diffs": (folds1 - folds2).tolist(),
    }


def run_multiple_seeds(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    preprocess_fn,
    seeds: list[int],
    outer_folds: int = 5,
    inner_folds: int = 3,
    groups: np.ndarray | None = None,
) -> dict:
    """Run nested CV across multiple random seeds."""
    all_results = []
    for seed in seeds:
        model = model_factory(seed)
        result = nested_cv(
            model,
            X,
            y,
            preprocess_fn,
            outer_folds,
            inner_folds,
            seed,
            calibrate=groups is None,
            groups=groups,
        )
        all_results.append(result)

    # Aggregate across seeds
    agg = {}
    for metric in all_results[0]["aggregated_metrics"]:
        means = [r["aggregated_metrics"][metric]["mean"] for r in all_results]
        agg[metric] = {
            "mean": float(np.mean(means)),
            "std": float(np.std(means)),
            "per_seed": means,
        }

    return {
        "per_seed": all_results,
        "aggregated": agg,
    }
