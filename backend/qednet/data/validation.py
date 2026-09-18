"""F2 — Data Validation & Leakage Guard.

Schema validation, datatype validation, missing-value analysis, duplicate
detection, target validation, invalid-value detection and preprocessing
leakage protection (train/test contamination checks).

Learned transformers (scaler, imputer, SMOTE, feature selector, PCA) must be
fit ONLY on training folds. ``LeakageGuard`` verifies this at runtime and
fails loudly (raise) — a leakage check failure means TEST FAILURE /
EXPERIMENT INVALID, never a silent continue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("qednet.validation")


class ValidationError(ValueError):
    """Raised when dataset validation fails."""


@dataclass
class ValidationReport:
    """Structured validation outcome (machine-readable)."""
    dataset: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "passed": self.passed,
            "errors": self.errors, "warnings": self.warnings,
            "checks": self.checks, "stats": self.stats,
        }


def validate_frame(df: pd.DataFrame, target: str,
                   dataset_name: str = "dataset") -> ValidationReport:
    """Full validation of a dataframe for binary classification research."""
    rep = ValidationReport(dataset=dataset_name, passed=True)

    # 1. target present ---------------------------------------------------
    if target not in df.columns:
        rep.errors.append(f"Target column '{target}' missing")
        rep.checks["target_present"] = False
        rep.passed = False
        return rep
    rep.checks["target_present"] = True

    # 2. non-empty ---------------------------------------------------------
    rep.checks["non_empty"] = df.shape[0] > 0 and df.shape[1] >= 2
    if not rep.checks["non_empty"]:
        rep.errors.append("Dataset is empty or has no feature columns")
        rep.passed = False
        return rep

    # 3. target is binary / castable ---------------------------------------
    y = df[target]
    uniq = pd.unique(y.dropna())
    castable = set(map(_norm_label, uniq)) <= {0, 1}
    rep.checks["target_binary"] = castable
    if not castable:
        rep.errors.append(
            f"Target '{target}' is not binary (unique values: {list(uniq)[:6]})")
        rep.passed = False

    # 4. datatype inspection ----------------------------------------------
    bad_dtypes = []
    for c in df.columns:
        if c == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad_dtypes.append(f"{c}({df[c].dtype})")
    rep.checks["numeric_features"] = not bad_dtypes
    if bad_dtypes:
        rep.errors.append(
            "Non-numeric feature columns: " + ", ".join(bad_dtypes[:8]))
        rep.passed = False

    # 5. missing values ------------------------------------------------------
    miss = df.isna().sum()
    total_missing = int(miss.sum())
    rep.checks["missing_analysed"] = True
    rep.stats["total_missing"] = total_missing
    rep.stats["missing_fraction"] = float(df.isna().mean().mean())
    high_missing = [c for c, v in miss.items() if v > 0.5 * len(df)]
    if high_missing:
        rep.warnings.append(
            f"Columns >50% missing: {high_missing[:6]} (imputation required)")

    # 6. duplicates -----------------------------------------------------------
    dup = int(df.duplicated().sum())
    rep.checks["duplicates_checked"] = True
    rep.stats["duplicate_rows"] = dup
    if dup > 0:
        rep.warnings.append(f"{dup} duplicate rows detected (will be dropped)")

    # 7. invalid values -------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    nonfinite = int(np.sum(~np.isfinite(X.to_numpy(dtype=float, na_value=np.nan))
                           ) if not bad_dtypes else 0)
    rep.stats["nonfinite_values"] = nonfinite
    rep.checks["values_finite"] = nonfinite == 0
    if nonfinite:
        rep.warnings.append(
            f"{nonfinite} non-finite values (inf/NaN) need imputation")

    # 8. class balance ----------------------------------------------------------
    balance = df[target].dropna().map(_norm_label).value_counts(normalize=True)
    rep.stats["class_balance"] = {str(k): float(v) for k, v in balance.items()}
    minority = float(balance.min())
    if minority < 0.10:
        rep.warnings.append(
            f"Severe class imbalance (minority class {minority:.1%}); "
            "SMOTE inside CV folds is recommended")

    # 9. identifier-like columns (privacy flag) ----------------------------------
    id_cols = [c for c in df.columns
               if any(h in str(c).lower() for h in
                      ("name", "email", "phone", "patient_id", "ssn", "mrn"))]
    if id_cols:
        rep.warnings.append(
            f"Direct-identifier-like columns present: {id_cols} — "
            "must be removed before research use (privacy)")
    rep.stats["identifier_like_columns"] = id_cols

    logger.info("Validation of '%s': passed=%s (%d errors, %d warnings)",
                dataset_name, rep.passed, len(rep.errors), len(rep.warnings))
    return rep


def _norm_label(v: Any) -> int | None:
    """Normalise a binary label to {0, 1}."""
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    try:
        f = float(v)
        if f in (0.0, 1.0):
            return int(f)
    except (TypeError, ValueError):
        pass
    s = str(v).strip().lower()
    if s in ("b", "benign", "negative", "no", "false", "healthy"):
        return 0
    if s in ("m", "malignant", "positive", "yes", "true", "diseased"):
        return 1
    return None


def to_binary_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Cast the target column to strict {0,1} ints; raise if impossible."""
    labels = df[target].map(_norm_label)
    if labels.isna().any():
        bad = df.loc[labels.isna(), target].unique()[:5]
        raise ValidationError(f"Uncastable target labels: {list(bad)}")
    return labels.astype(int)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

LEARNED_TRANSFORMERS = ("scaler", "imputer", "smote", "selector", "pca",
                        "compressor", "encoder_fit")


class LeakageGuard:
    """Runtime guard enforcing fit-on-train-only discipline.

    Every learned transformer in the preprocessing pipeline must call
    ``guard.fit_part(name)`` when fit on training data and ``guard.assert_fitted``
    before ``transform`` is applied to any split. Fitting on data marked as
    validation/test raises ``LeakageError`` immediately.
    """

    def __init__(self) -> None:
        self._fitted_on: Dict[str, Tuple[int, int]] = {}
        self._forbidden: Optional[Tuple[int, int]] = None
        self.events: List[Dict[str, Any]] = []

    def allow_fit(self) -> None:
        """Open a fitting window (training fold context)."""
        self._forbidden = None

    def forbid_fit(self, fold_id: str = "eval") -> None:
        """Close the fitting window: any fit now is contamination."""
        self._forbidden = (id(self), hash(fold_id) & 0xFFFF)
        self._fold_id = fold_id

    def record_fit(self, part: str, n_samples: int) -> None:
        """Record a fit event; raises if fitting is forbidden."""
        if self._forbidden is not None:
            raise LeakageError(
                f"LEAKAGE DETECTED: transformer '{part}' was fit on "
                f"'{getattr(self, '_fold_id', 'eval')}' data "
                f"(fitting is only allowed on training folds). "
                "TEST FAILURE / EXPERIMENT INVALID.")
        self._fitted_on[part] = (n_samples, len(self.events))
        self.events.append({"part": part, "n_samples": n_samples,
                            "stage": "fit"})

    def record_transform(self, part: str, n_samples: int) -> None:
        self.events.append({"part": part, "n_samples": n_samples,
                            "stage": "transform"})

    def audit(self) -> Dict[str, Any]:
        """Audit summary proving every learned part was fit before use."""
        fitted_parts = [e["part"] for e in self.events if e["stage"] == "fit"]
        used_parts = [e["part"] for e in self.events if e["stage"] == "transform"]
        contaminated = [p for p in used_parts if p not in fitted_parts]
        return {
            "fitted_parts": sorted(set(fitted_parts)),
            "transformed_parts": sorted(set(used_parts)),
            "contamination_detected": bool(contaminated),
            "n_fit_events": len(fitted_parts),
            "n_transform_events": len(used_parts),
        }


class LeakageError(RuntimeError):
    """Preprocessing leakage / train-test contamination detected."""


def check_train_test_contamination(X_train: np.ndarray, X_test: np.ndarray,
                                   tol: float = 1e-9) -> Dict[str, Any]:
    """Detect exact duplicate rows shared between train and test splits."""
    tr = {r.tobytes() for r in np.asarray(X_train, dtype=np.float64)}
    dup = sum(1 for r in np.asarray(X_test, dtype=np.float64) if r.tobytes() in tr)
    return {
        "n_test_rows": int(len(X_test)),
        "n_shared_rows": int(dup),
        "contaminated": bool(dup > 0),
        "note": "shared exact rows between train and test splits",
    }
