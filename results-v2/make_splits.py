import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SAMPLE_SEED = 42
FOLD_SEED = 42
N_SPLITS = 3


def load_all():
    X, y = load_breast_cancer(return_X_y=True)
    datasets = {"breast_cancer": (np.asarray(X, dtype=float), np.asarray(y, dtype=int), None, np.arange(len(y)).tolist())}

    heart = pd.read_csv(ROOT / "data/raw/heart_cleveland.data", header=None, na_values="?")
    heart_clean = heart.dropna()
    heart_y = (heart_clean.iloc[:, 13].to_numpy() > 0).astype(int)
    datasets["heart_disease"] = (heart_clean.iloc[:, :13].to_numpy(dtype=float), heart_y, None, heart_clean.index.astype(int).tolist())

    park = pd.read_csv(ROOT / "data/raw/parkinsons.data")
    park_groups = park["name"].str.extract(r"phon_R\d+_(S\d+)_\d+")[0]
    if park_groups.isna().any():
        raise ValueError("Parkinson subject ID regex did not match every name value")
    park_y = park["status"].to_numpy(dtype=int)
    park_X = park.drop(columns=["name", "status"]).to_numpy(dtype=float)
    datasets["parkinsons"] = (park_X, park_y, park_groups.to_numpy(), park.index.astype(int).tolist())

    diabetes = pd.read_csv(ROOT / "data/raw/pima_diabetes.csv", header=None)
    diabetes_y = diabetes.iloc[:, -1].to_numpy(dtype=int)
    diabetes_X = diabetes.iloc[:, :-1].to_numpy(dtype=float)
    datasets["diabetes"] = (diabetes_X, diabetes_y, None, diabetes.index.astype(int).tolist())
    return datasets


def make_fold(train_local, test_local, source_indices, subject_ids):
    return {
        "train_sample_indices": train_local.astype(int).tolist(),
        "test_sample_indices": test_local.astype(int).tolist(),
        "train_source_row_indices": [int(source_indices[i]) for i in train_local],
        "test_source_row_indices": [int(source_indices[i]) for i in test_local],
        "train_subject_ids": [str(subject_ids[i]) for i in train_local] if subject_ids is not None else None,
        "test_subject_ids": [str(subject_ids[i]) for i in test_local] if subject_ids is not None else None,
        "train_subject_count": int(len(set(subject_ids[train_local]))) if subject_ids is not None else None,
        "test_subject_count": int(len(set(subject_ids[test_local]))) if subject_ids is not None else None,
        "overlap_subject_ids": sorted(set(subject_ids[train_local]) & set(subject_ids[test_local])) if subject_ids is not None else [],
    }


def main():
    result = {
        "sample_seed": SAMPLE_SEED,
        "fold_seed": FOLD_SEED,
        "n_rows_per_dataset": 80,
        "n_splits": N_SPLITS,
        "fit_seeds": [42, 1729, 2024],
        "datasets": {},
    }
    for name, (X, y, groups, source_rows) in load_all().items():
        sample_local, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=80, random_state=SAMPLE_SEED).split(X, y))
        X_sample, y_sample = X[sample_local], y[sample_local]
        sample_groups = groups[sample_local] if groups is not None else None
        source_sample_rows = [source_rows[i] for i in sample_local]
        class_values, class_counts = np.unique(y_sample, return_counts=True)
        item = {
            "n": int(len(y_sample)),
            "input_features": int(X_sample.shape[1]),
            "class_counts": {str(int(k)): int(v) for k, v in zip(class_values, class_counts)},
            "sample_local_row_indices": [int(i) for i in sample_local],
            "sample_source_row_indices": [int(i) for i in source_sample_rows],
            "sample_labels": y_sample.astype(int).tolist(),
            "sample_subject_ids": sample_groups.astype(str).tolist() if sample_groups is not None else None,
            "sample_subject_record_counts": {},
            "random_folds": [],
            "grouped_folds": None,
        }
        if sample_groups is not None:
            vals, counts = np.unique(sample_groups, return_counts=True)
            item["sample_subject_record_counts"] = {str(v): int(c) for v, c in zip(vals, counts)}
        random_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
        for train_idx, test_idx in random_cv.split(X_sample, y_sample):
            item["random_folds"].append(make_fold(train_idx, test_idx, source_sample_rows, sample_groups))
        if sample_groups is not None:
            grouped_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
            grouped_folds = []
            for train_idx, test_idx in grouped_cv.split(X_sample, y_sample, sample_groups):
                fold = make_fold(train_idx, test_idx, source_sample_rows, sample_groups)
                if fold["overlap_subject_ids"]:
                    raise AssertionError(f"subject overlap in grouped fold: {fold['overlap_subject_ids']}")
                grouped_folds.append(fold)
            item["grouped_folds"] = grouped_folds
        result["datasets"][name] = item
        print(f"dataset={name} n={item['n']} input_features={item['input_features']} class_counts={item['class_counts']}")
        for kind in ("random_folds", "grouped_folds"):
            folds = item[kind]
            if folds is None:
                continue
            for fold_ix, fold in enumerate(folds):
                overlap = len(fold["overlap_subject_ids"])
                print(f"  split={kind} fold={fold_ix} n_train={len(fold['train_sample_indices'])} n_test={len(fold['test_sample_indices'])} train_subjects={fold['train_subject_count']} test_subjects={fold['test_subject_count']} overlap={overlap}")
        if sample_groups is not None:
            print(f"  sampled_subject_record_counts={item['sample_subject_record_counts']}")
    target = OUT / "splits.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"SAVED={target}")


if __name__ == "__main__":
    main()
