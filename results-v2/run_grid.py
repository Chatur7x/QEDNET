import argparse
import json
import sys
from pathlib import Path

from runner import OUT, append_json, run_fold


DATASETS = ("breast_cancer", "heart_disease", "parkinsons", "diabetes")
CLASSICAL_BASE = ("logistic_regression", "random_forest", "rbf_svm", "mlp_default")
PARAMETER_MATCHED = ("parameter_matched_mlp_vqc", "parameter_matched_mlp_reupload")
QUANTUM = ("qsvm", "vqc", "reupload")
SEEDS = (42, 1729, 2024)
FOLDS = (0, 1, 2)


def existing_rows(quantum):
    path = OUT / ("quantum_raw.json" if quantum else "classical_raw.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def has_record(rows, dataset, model, seed, fold, setting, split_variant):
    return any(
        r.get("run_type") == "benchmark"
        and r.get("dataset") == dataset
        and r.get("model") == model
        and r.get("seed") == seed
        and r.get("fold") == fold
        and r.get("feature_setting") == setting
        and r.get("split_variant") == split_variant
        for r in rows
    )


def add_not_measured(dataset, seed, fold, setting, split_variant, model):
    manifest = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    item = manifest["datasets"][dataset]
    split_key = "grouped_folds" if split_variant == "grouped" else "random_folds"
    split = item[split_key][fold]
    row = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "fold": fold,
        "feature_setting": setting,
        "split_variant": split_variant,
        "run_type": "benchmark",
        "n_samples": item["n"],
        "input_features": item["input_features"],
        "model_input_features": item["input_features"],
        "n_train": len(split["train_sample_indices"]),
        "n_test": len(split["test_sample_indices"]),
        "train_class_counts": None,
        "test_class_counts": None,
        "n_train_subjects": split["train_subject_count"],
        "n_test_subjects": split["test_subject_count"],
        "subject_overlap_count": len(split["overlap_subject_ids"]),
        "train_seconds": None,
        "predict_seconds": None,
        "auc": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "f1": None,
        "trainable_parameters": None,
        "status": "not_measured",
        "notes": "NOT MEASURED: setting B has no quantum model row to compare with this parameter-matched variant, and the real ParameterMatchedMLP architecture does not meet the 10-percent target parameter band for this dataset/model pair; see the measured architecture audit.",
    }
    append_json(OUT / "classical_raw.json", row)
    print(json.dumps({"saved": [dataset, setting, split_variant, model, seed, fold], "status": "not_measured"}), flush=True)


def should_skip(rows, dataset, model, seed, fold, setting, split_variant):
    if has_record(rows, dataset, model, seed, fold, setting, split_variant):
        print(json.dumps({"skipped_existing": [dataset, setting, split_variant, model, seed, fold]}), flush=True)
        return True
    return False


def run_classical():
    models = CLASSICAL_BASE + PARAMETER_MATCHED
    for dataset in DATASETS:
        split_variants = ("random", "grouped") if dataset == "parkinsons" else ("random",)
        for split_variant in split_variants:
            for setting in ("A", "B"):
                for seed in SEEDS:
                    for fold in FOLDS:
                        for model in models:
                            rows = existing_rows(False)
                            if should_skip(rows, dataset, model, seed, fold, setting, split_variant):
                                continue
                            if model in PARAMETER_MATCHED and setting == "B":
                                matchable = (dataset == "heart_disease" and model == "parameter_matched_mlp_reupload") or (dataset == "diabetes" and model == "parameter_matched_mlp_vqc")
                                if not matchable:
                                    add_not_measured(dataset, seed, fold, setting, split_variant, model)
                                    continue
                            result = run_fold(dataset, model, seed, fold, setting, split_variant, 900, "benchmark")
                            print(json.dumps({"saved": [dataset, setting, split_variant, model, seed, fold], "status": result.get("status"), "parent_elapsed_seconds": result.get("parent_elapsed_seconds")}), flush=True)


def run_quantum():
    for dataset in DATASETS:
        split_variants = ("random", "grouped") if dataset == "parkinsons" else ("random",)
        for split_variant in split_variants:
            for seed in SEEDS:
                for fold in FOLDS:
                    for model in QUANTUM:
                        rows = existing_rows(True)
                        if should_skip(rows, dataset, model, seed, fold, "A", split_variant):
                            continue
                        result = run_fold(dataset, model, seed, fold, "A", split_variant, 900, "benchmark")
                        print(json.dumps({"saved": [dataset, "A", split_variant, model, seed, fold], "status": result.get("status"), "parent_elapsed_seconds": result.get("parent_elapsed_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=("classical", "quantum"))
    args = parser.parse_args()
    if args.suite == "classical":
        run_classical()
    else:
        run_quantum()


if __name__ == "__main__":
    main()
