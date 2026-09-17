"""D8 Evaluation Engine public API."""

from .metrics import (
    compute_all_metrics,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
    expected_calibration_error,
    net_benefit,
    decision_curve_auc,
)
from .statistics import (
    bootstrap_ci,
    delong_roc_test,
    paired_t_test,
    corrected_paired_t_test,
)
from .calibration import (
    PlattCalibrator,
    IsotonicCalibrator,
    calibrate_model,
    evaluate_calibration,
)
from .cross_validation import (
    nested_cv,
    compare_models,
    run_multiple_seeds,
)
from .leakage_guard import (
    detect_leakage,
    check_preprocessing_leakage,
    audit_pipeline,
)
from .bootstrap import bootstrap_statistic, bootstrap_ci_diff
from .decision_curve import decision_curve, net_benefit_at_threshold, clinical_net_benefit

__all__ = [
    "compute_all_metrics",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity",
    "expected_calibration_error",
    "net_benefit",
    "decision_curve_auc",
    "bootstrap_ci",
    "delong_roc_test",
    "paired_t_test",
    "corrected_paired_t_test",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "calibrate_model",
    "evaluate_calibration",
    "nested_cv",
    "compare_models",
    "run_multiple_seeds",
    "detect_leakage",
    "check_preprocessing_leakage",
    "audit_pipeline",
    "bootstrap_statistic",
    "bootstrap_ci_diff",
    "decision_curve",
    "net_benefit_at_threshold",
    "clinical_net_benefit",
]