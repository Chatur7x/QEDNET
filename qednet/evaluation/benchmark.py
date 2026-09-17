"""Benchmark utilities (D8)."""

from __future__ import annotations

import numpy as np
from typing import Callable

from qednet.evaluation import nested_cv, compare_models
from qednet.preprocess import make_preprocess_pipeline, PreprocessConfig
from qednet.models import (
    VQC, QSVM, ReuploadClassifier,
    LogisticRegressionModel, RandomForestModel,
    XGBoostModel, RBFSVMModel, ParameterMatchedMLP,
)


def get_default_models(n_qubits: int = 8, n_features: int = 16, n_params_target: int = 100, seed: int = 42):
    """Get standard model zoo for benchmarking."""
    return {
        "logistic_regression": LogisticRegressionModel(random_state=seed),
        "random_forest": RandomForestModel(random_state=seed),
        "xgboost": XGBoostModel(random_state=seed),
        "rbf_svm": RBFSVMModel(random_state=seed),
        "parameter_matched_mlp": ParameterMatchedMLP(
            n_params_target=n_params_target, input_dim=n_features, random_state=seed
        ),
        "vqc": VQC(n_qubits=n_qubits, n_layers=3, seed=seed),
        "qsvm": QSVM(n_qubits=n_qubits, seed=seed),
        "reupload": ReuploadClassifier(n_qubits=n_qubits, n_layers=3, seed=seed),
    }


def run_benchmark(
    dataset_name: str,
    models: dict,
    preprocess_config: PreprocessConfig,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seeds: list[int] = [42],
) -> dict:
    """Run full benchmark on a dataset."""
    from qednet.data import DatasetRegistry

    bundle = DatasetRegistry.load(dataset_name)
    X, y = bundle.X, bundle.y

    preprocess_fn = lambda: make_preprocess_pipeline(preprocess_config)

    results = {}
    for name, model in models.items():
        print(f"Running {name}...")
        if hasattr(model, "n_qubits"):  # Quantum model
            model.n_qubits = min(model.n_qubits, preprocess_config.compress_dim)
        result = nested_cv(model, X, y, preprocess_fn, outer_folds, inner_folds, seeds[0])
        results[name] = result

    return {
        "dataset": dataset_name,
        "results": results,
    }


def pairwise_comparison(
    results: dict,
    metric: str = "auroc",
) -> dict:
    """Pairwise comparison of all models on a metric."""
    model_names = list(results.keys())
    n = len(model_names)
    comparisons = {}

    for i in range(n):
        for j in range(i + 1, n):
            name1, name2 = model_names[i], model_names[j]
            folds1 = np.array([r["metrics"][metric] for r in results[name1]["fold_results"]])
            folds2 = np.array([r["metrics"][metric] for r in results[name2]["fold_results"]])

            from qednet.evaluation.statistics import paired_t_test
            comp = paired_t_test(folds1, folds2)
            comp["model1"] = name1
            comp["model2"] = name2
            comparisons[f"{name1}_vs_{name2}"] = comp

    return comparisons