import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qednet.models import (
    LogisticRegressionModel,
    MLPClassifier,
    ParameterMatchedMLP,
    QSVM,
    RBFSVMModel,
    RandomForestModel,
    ReuploadClassifier,
    VQC,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def load_dataset(name):
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if name == "heart_disease":
        import pandas as pd
        frame = pd.read_csv(ROOT / "data/raw/heart_cleveland.data", header=None, na_values="?").dropna()
        return frame.iloc[:, :13].to_numpy(dtype=float), (frame.iloc[:, 13].to_numpy() > 0).astype(int)
    if name == "parkinsons":
        import pandas as pd
        frame = pd.read_csv(ROOT / "data/raw/parkinsons.data")
        return frame.drop(columns=["name", "status"]).to_numpy(dtype=float), frame["status"].to_numpy(dtype=int)
    if name == "diabetes":
        import pandas as pd
        frame = pd.read_csv(ROOT / "data/raw/pima_diabetes.csv", header=None)
        return frame.iloc[:, :-1].to_numpy(dtype=float), frame.iloc[:, -1].to_numpy(dtype=int)
    raise ValueError(f"unknown dataset {name}")


def make_model(name, seed, input_dim):
    if name == "logistic_regression":
        return LogisticRegressionModel(random_state=seed)
    if name == "random_forest":
        return RandomForestModel(random_state=seed)
    if name == "rbf_svm":
        return RBFSVMModel(random_state=seed)
    if name == "mlp_default":
        return MLPClassifier(early_stopping=False, max_iter=500, random_state=seed)
    if name == "parameter_matched_mlp_vqc":
        if input_dim == 4:
            return ParameterMatchedMLP(n_params_target=12, input_dim=input_dim, hidden_dims=(1, 2), random_state=seed)
        if input_dim == 8:
            return ParameterMatchedMLP(n_params_target=12, input_dim=input_dim, hidden_dims=(1,), random_state=seed)
        raise ValueError(f"no measured within-10-percent VQC parameter-matched architecture for input_dim={input_dim}")
    if name == "parameter_matched_mlp_reupload":
        if input_dim == 4:
            return ParameterMatchedMLP(n_params_target=15, input_dim=input_dim, hidden_dims=(2, 1), random_state=seed)
        if input_dim == 13:
            return ParameterMatchedMLP(n_params_target=16, input_dim=input_dim, hidden_dims=(1,), random_state=seed)
        raise ValueError(f"no measured within-10-percent Reupload parameter-matched architecture for input_dim={input_dim}")
    if name == "qsvm":
        return QSVM(n_qubits=4, encoding="zz", device="default.qubit", shots=None, random_state=seed)
    if name == "vqc":
        return VQC(n_qubits=4, n_layers=1, device="default.qubit", shots=None, seed=seed)
    if name == "reupload":
        return ReuploadClassifier(n_qubits=4, n_layers=1, device="default.qubit", shots=None, seed=seed)
    raise ValueError(f"unknown model {name}")


def make_preprocessor(feature_setting, seed):
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if feature_setting == "A":
        steps.extend([
            ("select", SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=seed), k=8)),
            ("pca", PCA(n_components=4, random_state=seed)),
        ])
    elif feature_setting != "B":
        raise ValueError(f"unknown feature setting {feature_setting}")
    return Pipeline(steps)


def run(args):
    manifest = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    item = manifest["datasets"][args.dataset]
    split_name = "grouped_folds" if args.split_variant == "grouped" else "random_folds"
    if args.split_variant == "grouped" and args.dataset != "parkinsons":
        raise ValueError("grouped split variant exists only for Parkinson's")
    split = item[split_name][args.fold]
    X_full, y_full = load_dataset(args.dataset)
    # The manifest stores source row IDs separately from local positions. Source IDs
    # may have gaps after cleaning (heart disease); only local positions index X_full.
    sample_rows = np.asarray(item["sample_local_row_indices"], dtype=int)
    if sample_rows.size != item["n"] or sample_rows.min() < 0 or sample_rows.max() >= len(X_full):
        raise ValueError("sample_local_row_indices are not valid positions in the loaded dataset")
    X = X_full[sample_rows]
    y = y_full[sample_rows]
    train_idx = np.asarray(split["train_sample_indices"], dtype=int)
    test_idx = np.asarray(split["test_sample_indices"], dtype=int)
    preprocessor = make_preprocessor(args.feature_setting, args.seed)
    prep_started = time.perf_counter()
    X_train = preprocessor.fit_transform(X[train_idx], y[train_idx])
    X_test = preprocessor.transform(X[test_idx])
    preprocess_seconds = time.perf_counter() - prep_started
    model = make_model(args.model, args.seed, X_train.shape[1])
    fit_started = time.perf_counter()
    model.fit(X_train, y[train_idx])
    train_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    probability = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
    predict_seconds = time.perf_counter() - predict_started
    prediction = (probability >= 0.5).astype(int)
    metadata = model.metadata() if hasattr(model, "metadata") else {}
    train_classes, train_counts = np.unique(y[train_idx], return_counts=True)
    test_classes, test_counts = np.unique(y[test_idx], return_counts=True)
    row = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "fold": args.fold,
        "feature_setting": args.feature_setting,
        "split_variant": args.split_variant,
        "run_type": args.run_type,
        "n_samples": int(len(y)),
        "input_features": int(X.shape[1]),
        "model_input_features": int(X_train.shape[1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_class_counts": {str(int(k)): int(v) for k, v in zip(train_classes, train_counts)},
        "test_class_counts": {str(int(k)): int(v) for k, v in zip(test_classes, test_counts)},
        "n_train_subjects": split["train_subject_count"],
        "n_test_subjects": split["test_subject_count"],
        "subject_overlap_count": len(split["overlap_subject_ids"]),
        "preprocess_seconds": preprocess_seconds,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "auc": float(roc_auc_score(y[test_idx], probability)),
        "accuracy": float(accuracy_score(y[test_idx], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], prediction)),
        "f1": float(f1_score(y[test_idx], prediction, zero_division=0)),
        "trainable_parameters": metadata.get("n_params", metadata.get("n_params_approx", getattr(model, "n_params", None))),
        "metadata": metadata,
        "status": "completed",
    }
    print(json.dumps(row, separators=(",", ":"), default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--feature-setting", required=True, choices=("A", "B"))
    parser.add_argument("--split-variant", default="random", choices=("random", "grouped"))
    parser.add_argument("--run-type", default="benchmark")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

