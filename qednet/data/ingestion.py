"""Dataset ingestion and registry (D2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .validation import DATASET_METADATA, DATASET_SCHEMAS


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    metadata: dict
    config_hash: str
    groups: np.ndarray | None = None


class DatasetRegistry:
    """Registry for supported datasets with reproducible loading."""

    _loaders = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._loaders[name] = func
            return func

        return decorator

    @classmethod
    def load(cls, name: str, **kwargs) -> DatasetBundle:
        name = name.lower()
        if name not in cls._loaders:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._loaders.keys())}")
        return cls._loaders[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._loaders.keys())


def _hash_config(**kwargs) -> str:
    """Generate a deterministic hash of the loading configuration."""
    cfg_str = str(sorted(kwargs.items()))
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:16]


def _validate_and_split(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    groups: np.ndarray | None = None,
    excluded_features: tuple[str, ...] = (),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
]:
    """Validate dataframe against schema and split."""
    sch = schema.remove_columns([target_col]) if target_col in schema.columns else schema
    sch.validate(df.drop(columns=[target_col]))
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = [c for c in df.columns if c != target_col and c not in excluded_features]
    stratify_arr = y if stratify else None
    X = df[feature_names].values
    split_arrays = [X, y]
    if groups is not None:
        split_arrays.append(np.asarray(groups))
    split = train_test_split(
        *split_arrays,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arr,
    )
    X_train, X_test, y_train, y_test = split[:4]
    groups_ordered = np.hstack([split[4], split[5]]) if groups is not None else None
    return X_train, X_test, y_train, y_test, feature_names, groups_ordered


def _load_breast_cancer_local(**kwargs) -> DatasetBundle:
    """Load from sklearn (clean, no missing values)."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = (df["target"] == 0).astype(int)  # 1=malignant, 0=benign
    df = df.drop(columns=["target"])

    def _sklearn_to_schema(c: str) -> str:
        c = c.replace("mean concave points", "concave_points_mean")
        c = c.replace("mean fractal dimension", "fractal_dimension_mean")
        if c.startswith("mean "):
            return f"{c[5:].replace(' ', '_')}_mean"
        if c.endswith(" error"):
            return f"{c[:-6].replace(' ', '_')}_se"
        if c.startswith("worst "):
            return f"{c[6:].replace(' ', '_')}_worst"
        return c

    df = df.rename(columns={c: _sklearn_to_schema(c) for c in df.columns if c != "diagnosis"})
    schema = DATASET_SCHEMAS["breast_cancer"]
    X_train, X_test, y_train, y_test, feature_names, _ = _validate_and_split(
        df, schema, "diagnosis", **kwargs
    )
    return DatasetBundle(
        X=np.vstack([X_train, X_test]),
        y=np.hstack([y_train, y_test]),
        feature_names=feature_names,
        target_name="diagnosis",
        metadata=DATASET_METADATA["breast_cancer"],
        config_hash=_hash_config(**kwargs),
    )


def _load_heart_disease_local(**kwargs) -> DatasetBundle:
    """Load UCI Heart Disease (Cleveland). No synthetic fallback — real data only."""
    for path in [
        "data/raw/heart_cleveland.data",
        "data/heart_disease.csv",
        "data/heart.csv",
    ]:
        if Path(path).exists():
            break
    else:
        raise FileNotFoundError(
            "Heart disease data missing. Run: python scripts/download_datasets.py"
        )
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    df = df.dropna().reset_index(drop=True)  # 6 rows with missing ca/thal
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    for c in ["ca", "thal"]:
        df[c] = df[c].astype(int)
    schema = DATASET_SCHEMAS["heart_disease"]
    X_train, X_test, y_train, y_test, feature_names, _ = _validate_and_split(
        df, schema, "target", **kwargs
    )
    return DatasetBundle(
        X=np.vstack([X_train, X_test]),
        y=np.hstack([y_train, y_test]),
        feature_names=feature_names,
        target_name="target",
        metadata=DATASET_METADATA["heart_disease"],
        config_hash=_hash_config(**kwargs),
    )


def _load_parkinsons_local(**kwargs) -> DatasetBundle:
    """Load UCI Parkinson's (voice). No synthetic fallback — real data only."""
    for path in [
        "data/raw/parkinsons.data",
        "data/parkinsons.csv",
    ]:
        if Path(path).exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "Parkinson's data missing. Run: python scripts/download_datasets.py"
        )
    if "name" not in df.columns:
        raise ValueError("Parkinson's dataset is missing the subject identifier column 'name'")
    groups = df["name"].astype(str).str.extract(r"(S\d+)", expand=False)
    if groups.isna().any():
        raise ValueError("Could not extract subject IDs from every Parkinson's row")
    schema = DATASET_SCHEMAS["parkinsons"]
    X_train, X_test, y_train, y_test, feature_names, groups_ordered = _validate_and_split(
        df, schema, "status", groups=groups.to_numpy(), excluded_features=("name",), **kwargs
    )
    return DatasetBundle(
        X=np.vstack([X_train, X_test]),
        y=np.hstack([y_train, y_test]),
        feature_names=feature_names,
        target_name="status",
        metadata=DATASET_METADATA["parkinsons"],
        config_hash=_hash_config(**kwargs),
        groups=groups_ordered,
    )


def _load_diabetes_local(**kwargs) -> DatasetBundle:
    """Load Pima Indians Diabetes. No synthetic fallback — real data only."""
    for path in [
        "data/raw/pima_diabetes.csv",
        "data/diabetes.csv",
    ]:
        if Path(path).exists():
            break
    else:
        raise FileNotFoundError("Diabetes data missing. Run: python scripts/download_datasets.py")
    cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    # documented Pima quirk: 0 = missing in these physiological columns
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
    schema = DATASET_SCHEMAS["diabetes"]
    X_train, X_test, y_train, y_test, feature_names, _ = _validate_and_split(
        df, schema, "Outcome", **kwargs
    )
    return DatasetBundle(
        X=np.vstack([X_train, X_test]),
        y=np.hstack([y_train, y_test]),
        feature_names=feature_names,
        target_name="Outcome",
        metadata=DATASET_METADATA["diabetes"],
        config_hash=_hash_config(**kwargs),
    )


# Register all loaders
DatasetRegistry.register("breast_cancer")(_load_breast_cancer_local)
DatasetRegistry.register("heart_disease")(_load_heart_disease_local)
DatasetRegistry.register("parkinsons")(_load_parkinsons_local)
DatasetRegistry.register("diabetes")(_load_diabetes_local)


# De-identification hook (stub for production use)
def deidentify(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """Remove or hash direct identifiers. Extend for production."""
    out = df.copy()
    for col in id_columns:
        if col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
                )
            else:
                out = out.drop(columns=[col])
    return out
