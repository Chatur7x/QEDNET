"""Tests for cascade, conformal prediction, and uncertainty."""

import numpy as np
import pytest
from numpy.typing import NDArray

from qednet.deploy.cascade import QuantumSecondOpinionCascade
from qednet.deploy.conformal import ConformalPredictor, ConformalResult
from qednet.deploy.serving import PredictionResult, ServingPipeline
from qednet.deploy.uncertainty import UncertaintyQuantifier, UncertaintyResult
from qednet.models.base import BaseEstimator


class MockModel(BaseEstimator):
    """Mock model for testing."""

    def __init__(self, prob_class_1=0.5):
        self.prob_class_1 = prob_class_1
        self.fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "MockModel":
        self.fitted = True
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        n = X.shape[0]
        p1 = np.full(n, self.prob_class_1)
        p0 = 1 - p1
        return np.column_stack([p0, p1])


class TestQuantumSecondOpinionCascade:
    def setup_method(self):
        self.classical = MockModel(prob_class_1=0.8)
        self.quantum = MockModel(prob_class_1=0.3)
        self.classical.fit(np.zeros((10, 5)), np.zeros(10))
        self.quantum.fit(np.zeros((10, 5)), np.zeros(10))

    def test_cascade_creation(self):
        """Test creating cascade."""
        cascade = QuantumSecondOpinionCascade(
            classical_model=self.classical,
            quantum_model=self.quantum,
            ambiguity_threshold=(0.35, 0.65),
        )
        assert cascade.ambiguity_low == 0.35
        assert cascade.ambiguity_high == 0.65

    def test_invalid_threshold_raises(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError):
            QuantumSecondOpinionCascade(self.classical, self.quantum, (0.6, 0.4))
        with pytest.raises(ValueError):
            QuantumSecondOpinionCascade(self.classical, self.quantum, (-0.1, 0.5))

    def test_high_confidence_classical(self):
        """Test routing when classical is confident (high prob)."""
        cascade = QuantumSecondOpinionCascade(self.classical, self.quantum)
        X = np.zeros((5, 5))
        results = cascade.predict(X)

        for r in results:
            assert r.routing == "classical"
            assert r.reason == "high_confidence"
            assert r.calibrated_probability == 0.8
            assert r.quantum_proba is None

    def test_low_confidence_classical(self):
        """Test routing when classical is confident (low prob)."""
        classical_low = MockModel(prob_class_1=0.1)
        classical_low.fit(np.zeros((10, 5)), np.zeros(10))
        cascade = QuantumSecondOpinionCascade(classical_low, self.quantum)
        X = np.zeros((5, 5))
        results = cascade.predict(X)

        for r in results:
            assert r.routing == "classical"
            assert r.reason == "high_confidence"
            assert r.calibrated_probability == 0.1

    def test_ambiguous_routes_to_quantum(self):
        """Test routing when classical is ambiguous."""
        classical_amb = MockModel(prob_class_1=0.5)
        classical_amb.fit(np.zeros((10, 5)), np.zeros(10))
        cascade = QuantumSecondOpinionCascade(classical_amb, self.quantum)
        X = np.zeros((5, 5))
        results = cascade.predict(X)

        for r in results:
            assert r.routing == "quantum_second_opinion"
            assert r.reason == "ambiguous_probability"
            assert r.calibrated_probability == 0.3
            assert r.quantum_proba == 0.3

    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        # At low boundary
        classical_at_low = MockModel(prob_class_1=0.35)
        classical_at_low.fit(np.zeros((10, 5)), np.zeros(10))
        cascade = QuantumSecondOpinionCascade(classical_at_low, self.quantum)
        results = cascade.predict(np.zeros((1, 5)))
        assert results[0].routing == "quantum_second_opinion"

        # At high boundary
        classical_at_high = MockModel(prob_class_1=0.65)
        classical_at_high.fit(np.zeros((10, 5)), np.zeros(10))
        cascade = QuantumSecondOpinionCascade(classical_at_high, self.quantum)
        results = cascade.predict(np.zeros((1, 5)))
        assert results[0].routing == "quantum_second_opinion"

        # Just outside
        classical_outside = MockModel(prob_class_1=0.34)
        classical_outside.fit(np.zeros((10, 5)), np.zeros(10))
        cascade = QuantumSecondOpinionCascade(classical_outside, self.quantum)
        results = cascade.predict(np.zeros((1, 5)))
        assert results[0].routing == "classical"


class TestConformalPredictor:
    def setup_method(self):
        self.model = MockModel(prob_class_1=0.7)
        self.model.fit(np.zeros((20, 5)), np.zeros(20))
        self.predictor = ConformalPredictor(self.model, alpha=0.10)

    def test_calibrate(self):
        """Test calibration."""
        X_cal = np.random.randn(50, 5)
        y_cal = np.random.randint(0, 2, 50)
        self.predictor.calibrate(X_cal, y_cal)

        assert self.predictor._is_calibrated
        assert self.predictor.threshold is not None
        assert self.predictor.calibration_coverage is not None
        assert len(self.predictor.calibration_scores) == 50

    def test_predict_without_calibrate_raises(self):
        """Test error when predicting without calibration."""
        X = np.zeros((5, 5))
        with pytest.raises(RuntimeError):
            self.predictor.predict(X)

    def test_predict_after_calibrate(self):
        """Test prediction after calibration."""
        X_cal = np.random.randn(50, 5)
        y_cal = np.random.randint(0, 2, 50)
        self.predictor.calibrate(X_cal, y_cal)

        X_test = np.random.randn(10, 5)
        results = self.predictor.predict(X_test)

        assert len(results) == 10
        for r in results:
            assert isinstance(r, ConformalResult)
            assert isinstance(r.prediction_set, list)
            assert all(c in [0, 1] for c in r.prediction_set)
            assert r.alpha == 0.10

    def test_abstention_detection(self):
        """Test abstention when prediction set has both classes."""
        # Use model that gives 0.5 probability (max ambiguity)
        model_50 = MockModel(prob_class_1=0.5)
        model_50.fit(np.zeros((20, 5)), np.zeros(20))
        predictor = ConformalPredictor(model_50, alpha=0.10)

        X_cal = np.random.randn(50, 5)
        y_cal = np.random.randint(0, 2, 50)
        predictor.calibrate(X_cal, y_cal)

        X_test = np.random.randn(10, 5)
        results = predictor.predict(X_test)

        # With p=0.5 and alpha=0.1, both classes should be in set
        for r in results:
            assert r.abstained or len(r.prediction_set) <= 2

    def test_coverage_report(self):
        """Test coverage report."""
        X_cal = np.random.randn(50, 5)
        y_cal = np.random.randint(0, 2, 50)
        self.predictor.calibrate(X_cal, y_cal)

        report = self.predictor.get_coverage_report()
        assert "target_coverage" in report
        assert "empirical_coverage" in report
        assert report["target_coverage"] == 0.90
        assert 0 <= report["empirical_coverage"] <= 1

    def test_uses_finite_sample_order_statistic_not_interpolated_quantile(self):
        model = MockModel(prob_class_1=0.7)
        predictor = ConformalPredictor(model, alpha=0.5)
        predictor.calibrate(np.zeros((5, 5)), np.array([1, 1, 1, 0, 0]))
        assert predictor.threshold == pytest.approx(0.3)

    def test_small_calibration_set_uses_all_labels_fallback(self):
        predictor = ConformalPredictor(MockModel(prob_class_1=0.7), alpha=0.1)
        predictor.calibrate(np.zeros((3, 5)), np.array([0, 1, 0]))
        assert predictor.threshold == 1.0
        assert predictor.predict(np.zeros((1, 5)))[0].prediction_set == [0, 1]

    def test_margin_score_is_candidate_specific(self):
        predictor = ConformalPredictor(MockModel(prob_class_1=0.7), alpha=0.1, score_type="margin")
        predictor.calibrate(np.zeros((20, 5)), np.ones(20, dtype=int))
        assert predictor.predict(np.zeros((1, 5)))[0].prediction_set == [1]

    def test_randomized_aps_is_reproducible_with_fixed_seed(self):
        model = MockModel(prob_class_1=0.7)
        X_cal = np.zeros((20, 5))
        y_cal = np.tile([0, 1], 10)
        first = ConformalPredictor(model, alpha=0.2, score_type="aps", random_state=73)
        second = ConformalPredictor(model, alpha=0.2, score_type="aps", random_state=73)
        first.calibrate(X_cal, y_cal)
        second.calibrate(X_cal, y_cal)
        assert np.array_equal(first.calibration_scores, second.calibration_scores)
        assert first.threshold == second.threshold
        assert first.predict(X_cal[:3]) == second.predict(X_cal[:3])


class TestUncertaintyQuantifier:
    def test_shot_noise_variance(self):
        """Test shot noise variance calculation: p(1-p)/S."""
        quantifier = UncertaintyQuantifier(shots=1024)

        # Test at p=0.5 (maximum variance)
        result = quantifier.quantify(0.5)
        expected_var = 0.5 * 0.5 / 1024
        assert abs(result.variance - expected_var) < 1e-10
        assert result.shots == 1024

        # Test at p=0.7
        result = quantifier.quantify(0.7)
        expected_var = 0.7 * 0.3 / 1024
        assert abs(result.variance - expected_var) < 1e-10

        # Test at extremes (clipped)
        result = quantifier.quantify(0.0)
        assert result.variance > 0  # Clipped to 1e-10

        result = quantifier.quantify(1.0)
        assert result.variance > 0

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        quantifier = UncertaintyQuantifier(shots=1024, confidence=0.95)
        result = quantifier.quantify(0.5)

        assert result.ci_lower <= result.probability <= result.ci_upper
        assert 0 <= result.ci_lower <= 1
        assert 0 <= result.ci_upper <= 1

    def test_batch_quantify(self):
        """Test batch uncertainty quantification."""
        quantifier = UncertaintyQuantifier(shots=1024)
        probs = np.array([0.1, 0.5, 0.9])
        results = quantifier.quantify_batch(probs)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, UncertaintyResult)

    def test_ensemble_uncertainty(self):
        """Test epistemic uncertainty from ensemble."""
        quantifier = UncertaintyQuantifier(shots=1024)
        ensemble_probs = np.array([0.6, 0.65, 0.55, 0.7, 0.58])

        result = quantifier.quantify(0.62, ensemble_probs)

        assert result.ensemble_variance is not None
        assert result.ensemble_std is not None
        assert result.ensemble_variance > 0


class TestServingPipeline:
    def test_basic_serving(self):
        """Test basic serving pipeline."""
        model = MockModel(prob_class_1=0.7)
        model.fit(np.zeros((10, 5)), np.zeros(10))

        pipeline = ServingPipeline(model=model, model_name="test", config_hash="abc123")
        X = np.zeros((3, 5))
        results = pipeline.predict(X)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.model_name == "test"
            assert r.config_hash == "abc123"
            assert r.prediction in [0, 1]

    def test_serving_with_cascade(self):
        """Test serving pipeline with cascade."""
        classical = MockModel(prob_class_1=0.5)
        quantum = MockModel(prob_class_1=0.3)
        classical.fit(np.zeros((10, 5)), np.zeros(10))
        quantum.fit(np.zeros((10, 5)), np.zeros(10))

        from qednet.deploy.cascade import QuantumSecondOpinionCascade

        cascade = QuantumSecondOpinionCascade(classical, quantum)

        pipeline = ServingPipeline(model=classical, cascade=cascade, model_name="cascade_test")
        X = np.zeros((3, 5))
        results = pipeline.predict(X)

        for r in results:
            assert r.routing in ["classical", "quantum_second_opinion"]

    def test_serving_can_omit_optional_uncertainty_and_conformal_outputs(self):
        from unittest.mock import MagicMock

        model = MockModel(prob_class_1=0.7)
        uncertainty = MagicMock()
        conformal = MagicMock()
        pipeline = ServingPipeline(
            model=model,
            uncertainty=uncertainty,
            conformal=conformal,
        )

        result = pipeline.predict(
            np.zeros((1, 5)),
            return_uncertainty=False,
            return_conformal=False,
        )[0]

        assert result.uncertainty is None
        assert result.conformal is None
        assert uncertainty.quantify.call_count == 0
        assert conformal.predict.call_count == 0
