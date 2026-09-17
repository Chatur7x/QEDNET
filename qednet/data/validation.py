"""Dataset schemas and validation (D2)."""

from __future__ import annotations

import pandera as pa
from pandera import Column, Check, DataFrameSchema


# Wisconsin Breast Cancer (UCI)
BREAST_CANCER_SCHEMA = DataFrameSchema(
    {
        "radius_mean": Column(float, Check.greater_than(0)),
        "texture_mean": Column(float, Check.greater_than(0)),
        "perimeter_mean": Column(float, Check.greater_than(0)),
        "area_mean": Column(float, Check.greater_than(0)),
        "smoothness_mean": Column(float, Check.between(0, 1)),
        "compactness_mean": Column(float, Check.greater_than_or_equal_to(0)),
        "concavity_mean": Column(float, Check.greater_than_or_equal_to(0)),
        "concave_points_mean": Column(float, Check.between(0, 1)),
        "symmetry_mean": Column(float, Check.between(0, 1)),
        "fractal_dimension_mean": Column(float, Check.between(0, 1)),
        "radius_se": Column(float, Check.greater_than_or_equal_to(0)),
        "texture_se": Column(float, Check.greater_than_or_equal_to(0)),
        "perimeter_se": Column(float, Check.greater_than_or_equal_to(0)),
        "area_se": Column(float, Check.greater_than_or_equal_to(0)),
        "smoothness_se": Column(float, Check.greater_than_or_equal_to(0)),
        "compactness_se": Column(float, Check.greater_than_or_equal_to(0)),
        "concavity_se": Column(float, Check.greater_than_or_equal_to(0)),
        "concave_points_se": Column(float, Check.greater_than_or_equal_to(0)),
        "symmetry_se": Column(float, Check.greater_than_or_equal_to(0)),
        "fractal_dimension_se": Column(float, Check.greater_than_or_equal_to(0)),
        "radius_worst": Column(float, Check.greater_than(0)),
        "texture_worst": Column(float, Check.greater_than(0)),
        "perimeter_worst": Column(float, Check.greater_than(0)),
        "area_worst": Column(float, Check.greater_than(0)),
        "smoothness_worst": Column(float, Check.between(0, 1)),
        "compactness_worst": Column(float, Check.greater_than_or_equal_to(0)),
        "concavity_worst": Column(float, Check.greater_than_or_equal_to(0)),
        "concave_points_worst": Column(float, Check.between(0, 1)),
        "symmetry_worst": Column(float, Check.between(0, 1)),
        "fractal_dimension_worst": Column(float, Check.between(0, 1)),
    },
    strict=True,
    coerce=True,
)

# UCI Heart Disease (Cleveland) - 14 columns including target
HEART_DISEASE_SCHEMA = DataFrameSchema(
    {
        "age": Column(int, Check.between(20, 100)),
        "sex": Column(int, Check.isin([0, 1])),
        "cp": Column(int, Check.isin([1, 2, 3, 4])),  # Cleveland coding is 1-based
        "trestbps": Column(int, Check.between(80, 200)),
        "chol": Column(int, Check.greater_than(100)),
        "fbs": Column(int, Check.isin([0, 1])),
        "restecg": Column(int, Check.isin([0, 1, 2])),
        "thalach": Column(int, Check.between(60, 220)),
        "exang": Column(int, Check.isin([0, 1])),
        "oldpeak": Column(float, Check.greater_than_or_equal_to(0)),
        "slope": Column(int, Check.isin([1, 2, 3])),  # Cleveland coding is 1-based
        "ca": Column(int, Check.isin([0, 1, 2, 3, 4])),
        "thal": Column(int, Check.isin([3, 6, 7])),  # Cleveland coding: 3 normal, 6 fixed, 7 reversible
        "target": Column(int, Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)

# UCI Parkinson's - 22 features + target
PARKINSONS_SCHEMA = DataFrameSchema(
    {
        "name": Column(str),  # patient identifier
        "MDVP:Fo(Hz)": Column(float, Check.greater_than(0)),
        "MDVP:Fhi(Hz)": Column(float, Check.greater_than(0)),
        "MDVP:Flo(Hz)": Column(float, Check.greater_than(0)),
        "MDVP:Jitter(%)": Column(float, Check.between(0, 1)),
        "MDVP:Jitter(Abs)": Column(float, Check.greater_than_or_equal_to(0)),
        "MDVP:RAP": Column(float, Check.between(0, 1)),
        "MDVP:PPQ": Column(float, Check.between(0, 1)),
        "Jitter:DDP": Column(float, Check.between(0, 1)),
        "MDVP:Shimmer": Column(float, Check.greater_than_or_equal_to(0)),
        "MDVP:Shimmer(dB)": Column(float, Check.greater_than_or_equal_to(0)),
        "Shimmer:APQ3": Column(float, Check.between(0, 1)),
        "Shimmer:APQ5": Column(float, Check.between(0, 1)),
        "MDVP:APQ": Column(float, Check.between(0, 1)),
        "Shimmer:DDA": Column(float, Check.between(0, 1)),
        "NHR": Column(float, Check.greater_than_or_equal_to(0)),
        "HNR": Column(float, Check.greater_than(0)),
        "RPDE": Column(float, Check.between(0, 1)),
        "DFA": Column(float, Check.between(0, 1)),
        "spread1": Column(float),
        "spread2": Column(float),
        "D2": Column(float, Check.greater_than(0)),
        "PPE": Column(float, Check.between(0, 1)),
        "status": Column(int, Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)

# Pima Indian Diabetes
DIABETES_SCHEMA = DataFrameSchema(
    {
        "Pregnancies": Column(int, Check.between(0, 20)),
        "Glucose": Column(float, Check.between(0, 300), nullable=True),
        "BloodPressure": Column(float, Check.between(0, 200), nullable=True),
        "SkinThickness": Column(float, Check.between(0, 100), nullable=True),
        "Insulin": Column(float, Check.between(0, 900), nullable=True),
        "BMI": Column(float, Check.between(0, 70), nullable=True),
        "DiabetesPedigreeFunction": Column(float, Check.greater_than_or_equal_to(0)),
        "Age": Column(int, Check.between(20, 100)),
        "Outcome": Column(int, Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)

DATASET_SCHEMAS = {
    "breast_cancer": BREAST_CANCER_SCHEMA,
    "heart_disease": HEART_DISEASE_SCHEMA,
    "parkinsons": PARKINSONS_SCHEMA,
    "diabetes": DIABETES_SCHEMA,
}

DATASET_METADATA = {
    "breast_cancer": {
        "name": "Wisconsin Breast Cancer",
        "source": "UCI ML Repository",
        "n_features": 30,
        "target": "diagnosis",
        "target_map": {"M": 1, "B": 0},
        "task": "binary_classification",
    },
    "heart_disease": {
        "name": "UCI Heart Disease (Cleveland)",
        "source": "UCI ML Repository",
        "n_features": 13,
        "target": "target",
        "task": "binary_classification",
    },
    "parkinsons": {
        "name": "UCI Parkinson's Disease",
        "source": "UCI ML Repository",
        "n_features": 22,
        "target": "status",
        "task": "binary_classification",
    },
    "diabetes": {
        "name": "Pima Indian Diabetes",
        "source": "UCI ML Repository",
        "n_features": 8,
        "target": "Outcome",
        "task": "binary_classification",
    },
}