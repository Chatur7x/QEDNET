"""Validation utilities."""

from __future__ import annotations

import numpy as np


def validate_binary_labels(y: np.ndarray) -> np.ndarray:
    """Validate and convert to binary 0/1 labels."""
    y = np.asarray(y).ravel()
    unique = np.unique(y)
    if not set(unique).issubset({0, 1, -1}):
        raise ValueError(f"Labels must be binary, got {unique}")
    # Convert -1/1 to 0/1
    if -1 in unique:
        y = (y == 1).astype(int)
    return y


def validate_features(X: np.ndarray) -> np.ndarray:
    """Validate feature matrix."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("X has no samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X


def validate_train_test_split(X_train, X_test, y_train, y_test):
    """Validate train/test split consistency."""
    X_train = validate_features(X_train)
    X_test = validate_features(X_test)
    y_train = validate_binary_labels(y_train)
    y_test = validate_binary_labels(y_test)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {X_train.shape[1]} vs {X_test.shape[1]}")
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train/y_train size mismatch")
    if X_test.shape[0] != len(y_test):
        raise ValueError("X_test/y_test size mismatch")

    return X_train, X_test, y_train, y_test


def check_class_balance(y: np.ndarray, min_class_ratio: float = 0.01) -> dict:
    """Check class balance and warn if severe imbalance."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    ratios = counts / total
    min_ratio = ratios.min()

    return {
        "class_counts": dict(zip(unique.tolist(), counts.tolist(), strict=True)),
        "class_ratios": dict(zip(unique.tolist(), ratios.tolist(), strict=True)),
        "min_class_ratio": float(min_ratio),
        "is_imbalanced": min_ratio < min_class_ratio,
        "recommendation": "Consider class_weight='balanced' or SMOTE"
        if min_ratio < min_class_ratio
        else "OK",
    }
