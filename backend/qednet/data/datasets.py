"""Built-in open biomedical dataset adapters (Tier 1 + Tier 2).

All bundled datasets are public, de-identified benchmark datasets used for
research experimentation only. Restricted-access datasets (TCGA, MIMIC-IV)
are Tier 3 stretch goals and are NOT bundled — the platform works fully
without them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from .ingestion import (DatasetAdapter, DatasetRegistry, RegisteredDataset,
                        fingerprint_frame, profile_frame)

logger = logging.getLogger("qednet.datasets")

RAW_DIR = Path("data/raw")

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]


class BreastCancerAdapter(DatasetAdapter):
    """Wisconsin Diagnostic Breast Cancer (sklearn, 569 x 30, public)."""
    name = "breast_cancer"

    def load(self) -> pd.DataFrame:
        bunch = load_breast_cancer()
        df = pd.DataFrame(bunch.data, columns=[c.replace(" ", "_") for c in bunch.feature_names])
        # sklearn target: 1 = benign, 0 = malignant -> remap to 1 = malignant
        # so that the positive class is the clinically urgent one.
        df["target"] = (1 - bunch.target).astype(int)
        return df

    def target(self) -> str:
        return "target"


class HeartDiseaseAdapter(DatasetAdapter):
    """UCI Heart Disease, Cleveland processed (303 x 13, public).

    Target: 1 = angiographic diameter narrowing > 50% (disease).
    """
    name = "heart_disease"

    def load(self) -> pd.DataFrame:
        path = RAW_DIR / "processed.cleveland.data"
        df = pd.read_csv(path, header=None, names=HEART_COLUMNS,
                         na_values="?")
        df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
        df["thal"] = pd.to_numeric(df["thal"], errors="coerce")
        df["target"] = (df["target"] > 0).astype(int)
        return df

    def target(self) -> str:
        return "target"


class ParkinsonsAdapter(DatasetAdapter):
    """UCI Parkinson's voice recordings (195 x 22, public).

    Target: 1 = Parkinson's (positive class).
    """
    name = "parkinsons"

    def load(self) -> pd.DataFrame:
        path = RAW_DIR / "parkinsons.data"
        df = pd.read_csv(path)
        df = df.drop(columns=["name"])  # subject identifier, not a feature
        return df

    def target(self) -> str:
        return "status"


class PimaDiabetesAdapter(DatasetAdapter):
    """Pima Indian Diabetes (768 x 8, public). Target: 1 = diabetes."""
    name = "pima_diabetes"

    def load(self) -> pd.DataFrame:
        path = RAW_DIR / "pima-indians-diabetes.csv"
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "target"]
        df = pd.read_csv(path, header=None, names=cols)
        # Physiologically impossible zeros are treated as missing values.
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness",
                           "Insulin", "BMI"]
        df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
        return df

    def target(self) -> str:
        return "target"


BUILTIN_ADAPTERS: List[DatasetAdapter] = [
    BreastCancerAdapter(), HeartDiseaseAdapter(),
    ParkinsonsAdapter(), PimaDiabetesAdapter(),
]


def get_builtin_adapter(name: str) -> DatasetAdapter:
    for a in BUILTIN_ADAPTERS:
        if a.name == name:
            return a
    raise KeyError(
        f"Unknown builtin dataset '{name}'. Available: "
        f"{[a.name for a in BUILTIN_ADAPTERS]}")


def register_builtins(registry: DatasetRegistry) -> None:
    """Register/refresh all bundled datasets with profiles and fingerprints."""
    for adapter in BUILTIN_ADAPTERS:
        try:
            df = adapter.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load builtin '%s': %s", adapter.name, exc)
            continue
        target = adapter.target()
        profile = profile_frame(df, target)
        rec = RegisteredDataset(
            name=adapter.name, source="builtin", target=target,
            sha256=fingerprint_frame(df),
            n_samples=profile.n_samples, n_features=profile.n_features,
            profile=profile.to_dict(),
            validation_status="pending",
            notes="public benchmark dataset (research use only)",
        )
        if not registry.exists(adapter.name) or \
                registry.get(adapter.name).get("sha256") != rec.sha256:
            registry.register(rec)
            logger.info("Registered builtin dataset '%s' (%d x %d)",
                        adapter.name, profile.n_samples, profile.n_features)


def load_dataset(name: str, registry: DatasetRegistry) -> pd.DataFrame:
    """Load a registered dataset (builtin adapter or uploaded CSV)."""
    meta = registry.get(name)
    if meta["source"] == "builtin":
        return get_builtin_adapter(name).load()
    from .ingestion import load_dataframe
    return load_dataframe(name, registry)


DATASET_NOTES: Dict[str, str] = {
    "breast_cancer": (
        "Wisconsin Diagnostic Breast Cancer — 569 samples, 30 features. "
        "Positive class (1) = malignant. Tier 1 (mandatory)."),
    "heart_disease": (
        "UCI Heart Disease (Cleveland) — 303 samples, 13 features. "
        "Positive class (1) = diameter narrowing > 50%. Tier 1 (mandatory)."),
    "parkinsons": (
        "UCI Parkinson's (voice) — 195 samples, 22 features. "
        "Positive class (1) = Parkinson's. Small-data regime. Tier 1."),
    "pima_diabetes": (
        "Pima Indian Diabetes — 768 samples, 8 features. "
        "Positive class (1) = diabetes. Tier 2 (recommended)."),
}
