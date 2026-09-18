"""F3 — Preprocessing & Feature Compression.

Modular pipeline: imputation -> scaling -> optional class-imbalance handling
(SMOTE inside training folds ONLY) -> feature selection -> dimensionality
reduction / compression to quantum-compatible dimensions.

Every learned step is fit on training data only, verified by the
``LeakageGuard``. The pipeline maps explanations from compressed space back
to ORIGINAL input features (feature attribution propagation through PCA).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from ..data.validation import LeakageGuard, LeakageError
from .guarded import GuardedResampler, GuardedTransformer

logger = logging.getLogger("qednet.preprocess")


class PreprocessPipeline:
    """Leakage-safe preprocessing + compression pipeline.

    fit(X_train, y_train) learns: imputer, scaler, optional selector,
    compressor (PCA by default). transform(X) applies the fitted steps.
    The LeakageGuard records every fit/transform event.
    """

    def __init__(self, compression_method: str = "pca", target_dim: int = 8,
                 selection_k: Optional[int] = None,
                 handle_imbalance: bool = False,
                 guard: Optional[LeakageGuard] = None,
                 smote_random_state: int = 42):
        if compression_method not in ("pca", "mutual_info", "lasso", "identity"):
            raise ValueError(
                f"Unknown compression method '{compression_method}' "
                "(use pca | mutual_info | lasso | identity)")
        self.method = compression_method
        self.target_dim = int(target_dim)
        self.selection_k = selection_k
        self.handle_imbalance = handle_imbalance
        self.guard = guard or LeakageGuard()
        self._smote_rs = smote_random_state

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[SelectKBest] = None
        self.lasso: Optional[LassoCV] = None
        self.compressor: Optional[PCA] = None
        self.selected_features: Optional[List[int]] = None
        self.feature_names: List[str] = []
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "PreprocessPipeline":
        """Fit all learned steps on TRAINING data only."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.input_dim = X.shape[1]
        self.feature_names = list(feature_names or
                                  [f"x{i}" for i in range(X.shape[1])])

        # 1. imputation ------------------------------------------------------
        self.imputer = GuardedTransformer(
            SimpleImputer(strategy="median"), "imputer", self.guard)
        X = self.imputer.fit_transform(X)

        # 2. scaling ----------------------------------------------------------
        self.scaler = GuardedTransformer(
            StandardScaler(), "scaler", self.guard)
        X = self.scaler.fit_transform(X)

        # 3. feature selection (before compression) -----------------------------
        if self.method == "mutual_info":
            k = min(self.selection_k or self.target_dim, X.shape[1])
            self.selector = GuardedTransformer(
                SelectKBest(mutual_info_classif, k=k), "selector", self.guard)
            X = self.selector.fit_transform(X, y)
            self.selected_features = list(
                np.where(self.selector.get_support())[0])
        elif self.method == "lasso":
            self.lasso = GuardedTransformer(
                LassoCV(cv=3, n_alphas=20, max_iter=3000, random_state=0),
                "selector", self.guard)
            self.lasso.fit(X, y)
            coef = np.abs(self.lasso.coef_)
            k = min(self.target_dim, X.shape[1])
            self.selected_features = list(
                np.argsort(coef)[-k:][::-1])
            X = X[:, self.selected_features]

        # 4. compression to quantum-compatible dims ------------------------------
        if self.method == "pca":
            n_comp = min(self.target_dim, X.shape[1], X.shape[0] - 1)
            self.compressor = GuardedTransformer(
                PCA(n_components=n_comp, random_state=0), "pca", self.guard)
            Xc = self.compressor.fit_transform(X)
            self.output_dim = n_comp
            self.explained_variance = list(
                map(float, self.compressor.explained_variance_ratio_))
        else:
            self.output_dim = X.shape[1]
            self.explained_variance = []

        self._fitted = True
        logger.info("Pipeline fit: %d -> %d dims (method=%s, evr=%.3f)",
                    self.input_dim, self.output_dim, self.method,
                    float(np.sum(self.explained_variance or [0])))
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted steps (never fits)."""
        if not self._fitted:
            raise LeakageError("Pipeline transform before fit — invalid state")
        X = np.asarray(X, dtype=np.float64)
        self.guard.record_transform("imputer", X.shape[0])
        X = self.imputer.transform(X)
        self.guard.record_transform("scaler", X.shape[0])
        X = self.scaler.transform(X)
        if self.selector is not None:
            self.guard.record_transform("selector", X.shape[0])
            X = self.selector.transform(X)
        elif self.selected_features is not None:
            self.guard.record_transform("selector", X.shape[0])
            X = X[:, self.selected_features]
        if self.compressor is not None:
            self.guard.record_transform("pca", X.shape[0])
            X = self.compressor.transform(X)
        return X

    def fit_transform(self, X, y, feature_names=None) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)

    # --------------------------------------------------- imbalance handling
    def resample_train(self, X: np.ndarray, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Optional SMOTE resampling — ONLY on the training fold.

        Applied after transform so no learned step sees resampled data twice.
        """
        if not self.handle_imbalance:
            return X, y
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:  # pragma: no cover
            logger.warning("imbalanced-learn not installed; skipping SMOTE")
            return X, y
        sm = GuardedResampler(
            SMOTE(random_state=self._smote_rs, k_neighbors=3),
            "smote", self.guard)
        Xr, yr = sm.fit_resample(X, y)
        logger.info("SMOTE: %d -> %d training samples", len(y), len(yr))
        return Xr, yr

    # ------------------------------------------- explanation back-mapping
    def compressed_to_original_importance(
            self, compressed_importance: np.ndarray) -> np.ndarray:
        """Map per-compressed-feature importance back to ORIGINAL features.

        For PCA: |w_pc| * importance summed over components, then unselected
        features get 0. This keeps SHAP-style explanations at the original
        input-feature level as required (F8).
        """
        imp = np.asarray(compressed_importance, dtype=np.float64)
        if self.method == "pca" and self.compressor is not None:
            # importance over original scaled features
            comps = np.abs(self.compressor.components_)  # (k, d_scaled)
            orig_scaled = comps.T @ imp  # (d_scaled,)
        else:
            orig_scaled = np.zeros(self.input_dim)
            if self.selected_features is not None:
                for pos, fi in enumerate(self.selected_features):
                    if pos < len(imp):
                        orig_scaled[fi] = imp[pos]
            elif self.output_dim == self.input_dim:
                orig_scaled = imp
        # selection stages are identity for the non-selected path
        if self.selector is not None:
            support = self.selector.get_support()
            full = np.zeros(self.input_dim)
            idx = np.where(support)[0]
            for pos, fi in enumerate(idx):
                full[fi] = orig_scaled[pos] if pos < len(orig_scaled) else 0.0
            orig_scaled = full
        return np.abs(orig_scaled)

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "target_dim_requested": self.target_dim,
            "explained_variance_ratio": self.explained_variance,
            "selected_features": self.selected_features,
            "imbalance_handling": "SMOTE (train folds only)"
            if self.handle_imbalance else "none",
            "fitted": self._fitted,
        }
