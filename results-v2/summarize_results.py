import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

OUT = Path(__file__).resolve().parent
SUMMARY_COLUMNS = [
    "dataset", "feature_setting", "model", "seed", "fold", "status", "auc", "accuracy",
    "balanced_accuracy", "f1", "train_seconds", "predict_seconds", "timeout_elapsed_seconds",
    "n_train", "n_test", "trainable_parameters", "notes",
]
PAIRED_COLUMNS = [
    "dataset", "split_variant", "seed", "fold", "best_quantum_model", "best_quantum_auc",
    "best_classical_model", "best_classical_auc", "difference",
]
METRICS = ("auc", "accuracy", "balanced_accuracy", "f1", "train_seconds")
CLASSICAL_BASE = ("logistic_regression", "random_forest", "rbf_svm", "mlp_default")
PM_MODELS = ("parameter_matched_mlp_vqc", "parameter_matched_mlp_reupload")
QUANTUM_MODELS = ("qsvm", "vqc", "reupload")


def load_rows(filename):
    path = OUT / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    if value is None or value == "":
        return ""
    return str(value)


def metric_summary(rows, metric):
    values = [float(r[metric]) for r in rows if r.get("status") == "completed" and r.get(metric) is not None]
    if not values:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def fmt_stat(stat):
    if stat["n"] == 0:
        return "NOT MEASURED (no completed folds)"
    if stat["sd"] is None:
        return f'{stat["mean"]} (SD NOT MEASURED; n={stat["n"]})'
    return f'{stat["mean"]} ± {stat["sd"]} (n={stat["n"]})'


def paired_stats(values):
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ttest_p": None, "wilcoxon_p": None}
    result = {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "ttest_p": float(ttest_1samp(values, popmean=0.0).pvalue) if len(values) > 1 else None,
        "wilcoxon_p": float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        if len(values) >= 6 and any(v != 0 for v in values) else (1.0 if len(values) >= 6 else None),
    }
    return result


def ptext(value):
    return "NOT MEASURED" if value is None else str(value)


def main():
    classical_all = load_rows("classical_raw.json")
    quantum_all = load_rows("quantum_raw.json")
    classical = [r for r in classical_all if r.get("run_type") == "benchmark"]
    quantum = [r for r in quantum_all if r.get("run_type") == "benchmark"]
    manifest = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))

    # Fold-level CSV: every benchmark record, with the split variant and sample
    # details encoded in notes because the requested schema has no split column.
    summary_rows = []
    for row in classical + quantum:
        notes = [
            f"split_variant={row.get('split_variant', 'random')}",
            f"n_samples={row.get('n_samples', 'NOT MEASURED')}",
            f"input_features={row.get('input_features', 'NOT MEASURED')}",
            f"model_input_features={row.get('model_input_features', 'NOT MEASURED')}",
            f"subject_overlap_count={row.get('subject_overlap_count', 'NOT MEASURED')}",
        ]
        if row.get("notes"):
            notes.append(str(row["notes"]))
        if row.get("metadata"):
            notes.append("metadata=" + json.dumps(row["metadata"], sort_keys=True, separators=(",", ":")))
        summary_rows.append({
            "dataset": row.get("dataset"),
            "feature_setting": row.get("feature_setting"),
            "model": row.get("model"),
            "seed": row.get("seed"),
            "fold": row.get("fold"),
            "status": row.get("status"),
            "auc": number(row.get("auc")),
            "accuracy": number(row.get("accuracy")),
            "balanced_accuracy": number(row.get("balanced_accuracy")),
            "f1": number(row.get("f1")),
            "train_seconds": number(row.get("train_seconds")),
            "predict_seconds": number(row.get("predict_seconds")),
            "timeout_elapsed_seconds": number(row.get("parent_elapsed_seconds")) if row.get("status") == "timeout" else "",
            "n_train": number(row.get("n_train")),
            "n_test": number(row.get("n_test")),
            "trainable_parameters": number(row.get("trainable_parameters")),
            "notes": "; ".join(notes),
        })
    # Keep explicit NOT MEASURED rows (setting B ParameterMatchedMLP) from raw.
    for row in classical_all:
        if row.get("run_type") == "benchmark" and row.get("status") == "not_measured":
            if not any(
                s["dataset"] == row.get("dataset") and s["model"] == row.get("model")
                and s["seed"] == row.get("seed") and s["fold"] == row.get("fold")
                and s["feature_setting"] == row.get("feature_setting")
                and f"split_variant={row.get('split_variant', 'random')}" in s["notes"]
                for s in summary_rows
            ):
                summary_rows.append({
                    "dataset": row.get("dataset"), "feature_setting": row.get("feature_setting"),
                    "model": row.get("model"), "seed": row.get("seed"), "fold": row.get("fold"),
                    "status": "not_measured", "auc": "", "accuracy": "", "balanced_accuracy": "",
                    "f1": "", "train_seconds": "", "predict_seconds": "", "timeout_elapsed_seconds": "",
                    "n_train": number(row.get("n_train")), "n_test": number(row.get("n_test")),
                    "trainable_parameters": "", "notes": "; ".join([
                        f"split_variant={row.get('split_variant', 'random')}",
                        f"n_samples={row.get('n_samples', 'NOT MEASURED')}",
                        f"input_features={row.get('input_features', 'NOT MEASURED')}",
                        str(row.get("notes", "NOT MEASURED: no reason recorded")),
                    ]),
                })
    summary_rows.sort(key=lambda r: (r["dataset"], r["feature_setting"], r["notes"].split(";")[0], int(r["seed"]), int(r["fold"]), r["model"]))
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Paired best-of-per-fold differences. A winner can change from fold to fold.
    paired_rows = []
    group_keys = sorted({
        (r.get("dataset"), r.get("split_variant", "random"), r.get("seed"), r.get("fold"))
        for r in quantum if r.get("feature_setting") == "A" and r.get("status") == "completed"
    })
    for dataset, split_variant, seed, fold in group_keys:
        qrows = [r for r in quantum if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("seed") == seed and r.get("fold") == fold and r.get("feature_setting") == "A" and r.get("status") == "completed"]
        crows = [r for r in classical if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("seed") == seed and r.get("fold") == fold and r.get("feature_setting") == "A" and r.get("status") == "completed"]
        if not qrows or not crows:
            continue
        qb = max(qrows, key=lambda r: float(r["auc"]))
        cb = max(crows, key=lambda r: float(r["auc"]))
        paired_rows.append({
            "dataset": dataset, "split_variant": split_variant, "seed": seed, "fold": fold,
            "best_quantum_model": qb["model"], "best_quantum_auc": str(qb["auc"]),
            "best_classical_model": cb["model"], "best_classical_auc": str(cb["auc"]),
            "difference": str(float(qb["auc"]) - float(cb["auc"])),
        })
    paired_rows.sort(key=lambda r: (r["dataset"], r["split_variant"], int(r["seed"]), int(r["fold"])))
    paired_columns = PAIRED_COLUMNS
    with (OUT / "paired_differences.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=paired_columns)
        writer.writeheader()
        writer.writerows(paired_rows)

    # Table groups use the same direct sample but keep the two Parkinson
    # split variants distinct in every display.
    display_groups = [(name, "random") for name in DATASET_ORDER]
    display_groups.insert(3, ("parkinsons", "grouped"))
    analysis = {"tables": [], "paired_best": [], "parameter_matched": [], "run_counts": {}, "splits": {}}
    for dataset in DATASET_ORDER:
        item = manifest["datasets"][dataset]
        analysis["splits"][dataset] = {
            "n": item["n"], "input_features": item["input_features"],
            "class_counts": item["class_counts"],
            "fold_sizes_random": [[len(f["train_sample_indices"]), len(f["test_sample_indices"])] for f in item["random_folds"]],
            "fold_sizes_grouped": [[len(f["train_sample_indices"]), len(f["test_sample_indices"])] for f in item["grouped_folds"]] if item["grouped_folds"] else None,
            "group_audit": [{"fold": i, "n_train_subjects": f["train_subject_count"], "n_test_subjects": f["test_subject_count"], "overlap": len(f["overlap_subject_ids"])} for i, f in enumerate(item["grouped_folds"] or [])],
        }
    for dataset, split_variant in display_groups:
        for setting in ("A", "B"):
            rows = [r for r in classical + quantum if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("feature_setting") == setting]
            models = sorted({r.get("model") for r in rows})
            table = {"dataset": dataset, "split_variant": split_variant, "feature_setting": setting, "models": []}
            for model in models:
                mrows = [r for r in rows if r.get("model") == model]
                completed = [r for r in mrows if r.get("status") == "completed"]
                table["models"].append({
                    "model": model,
                    "status_counts": {status: sum(r.get("status") == status for r in mrows) for status in ("completed", "timeout", "error", "not_measured")},
                    "metrics": {metric: metric_summary(mrows, metric) for metric in METRICS},
                    "parameter_counts": sorted({r.get("trainable_parameters") for r in completed if r.get("trainable_parameters") is not None}),
                })
            analysis["tables"].append(table)

    for dataset, split_variant in display_groups:
        keys = sorted({(r.get("seed"), r.get("fold")) for r in quantum if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("feature_setting") == "A" and r.get("status") == "completed"})
        differences = [float(r["difference"]) for r in paired_rows if r["dataset"] == dataset and r["split_variant"] == split_variant]
        stats = paired_stats(differences)
        analysis["paired_best"].append({"dataset": dataset, "split_variant": split_variant, **stats})
        if dataset == "parkinsons" or dataset in DATASET_ORDER:
            for pm_model in PM_MODELS:
                for q_model in QUANTUM_MODELS:
                    deltas = []
                    for seed, fold in keys:
                        pm = [r for r in classical if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("feature_setting") == "A" and r.get("model") == pm_model and r.get("seed") == seed and r.get("fold") == fold and r.get("status") == "completed"]
                        q = [r for r in quantum if r.get("dataset") == dataset and r.get("split_variant", "random") == split_variant and r.get("feature_setting") == "A" and r.get("model") == q_model and r.get("seed") == seed and r.get("fold") == fold and r.get("status") == "completed"]
                        if pm and q:
                            deltas.append(float(pm[0]["auc"]) - float(q[0]["auc"]))
                    analysis["parameter_matched"].append({"dataset": dataset, "split_variant": split_variant, "pm_model": pm_model, "quantum_model": q_model, **paired_stats(deltas)})

    analysis["run_counts"] = {
        "classical_benchmark": {status: sum(r.get("status") == status for r in classical) for status in ("completed", "timeout", "error", "not_measured")},
        "quantum_benchmark": {status: sum(r.get("status") == status for r in quantum) for status in ("completed", "timeout", "error", "not_measured")},
        "quantum_timing_pilots_excluded": sum(r.get("run_type") != "benchmark" for r in quantum_all),
    }

    # Validate headers and print the first three data records from every CSV.
    csv_audit = {}
    for filename, expected in (("summary.csv", SUMMARY_COLUMNS), ("paired_differences.csv", PAIRED_COLUMNS)):
        with (OUT / filename).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            actual = reader.fieldnames
            all_rows = list(reader)
        first_rows = all_rows[:3]
        row_shapes_valid = all(len(row) == len(expected) and None not in row for row in all_rows)
        semantic_valid = True
        if filename == "summary.csv":
            source_rows = classical + quantum
            for row in all_rows:
                split_variant = row["notes"].split("split_variant=", 1)[1].split(";", 1)[0]
                match = [r for r in source_rows if r.get("dataset") == row["dataset"] and r.get("model") == row["model"] and str(r.get("seed")) == row["seed"] and str(r.get("fold")) == row["fold"] and r.get("feature_setting") == row["feature_setting"] and r.get("split_variant", "random") == split_variant]
                if len(match) != 1:
                    semantic_valid = False
                    break
                src = match[0]
                checks = {
                    "status": str(src.get("status")),
                    "auc": number(src.get("auc")),
                    "accuracy": number(src.get("accuracy")),
                    "balanced_accuracy": number(src.get("balanced_accuracy")),
                    "f1": number(src.get("f1")),
                    "train_seconds": number(src.get("train_seconds")),
                    "predict_seconds": number(src.get("predict_seconds")),
                    "timeout_elapsed_seconds": number(src.get("parent_elapsed_seconds")) if src.get("status") == "timeout" else "",
                    "n_train": number(src.get("n_train")),
                    "n_test": number(src.get("n_test")),
                    "trainable_parameters": number(src.get("trainable_parameters")),
                }
                if any(row[key] != value for key, value in checks.items()):
                    semantic_valid = False
                    break
            semantic_valid = semantic_valid and len(all_rows) == len(source_rows)
        else:
            for row in all_rows:
                src = [r for r in paired_rows if r["dataset"] == row["dataset"] and r["split_variant"] == row["split_variant"] and str(r["seed"]) == row["seed"] and str(r["fold"]) == row["fold"]]
                if len(src) != 1 or any(row[key] != str(src[0][key]) for key in PAIRED_COLUMNS):
                    semantic_valid = False
                    break
            semantic_valid = semantic_valid and len(all_rows) == len(paired_rows)
        valid = actual == expected and row_shapes_valid and semantic_valid
        csv_audit[filename] = {"expected_columns": expected, "actual_columns": actual, "row_count": len(all_rows), "three_sample_rows": first_rows, "row_shape_valid": row_shapes_valid, "semantic_values_match_source": semantic_valid, "column_validation_pass": valid}
        if not valid:
            raise AssertionError(f"CSV header/row validation failed: {filename}")
        print(f"CSV={filename} COLUMNS={actual}")
        print(json.dumps(first_rows, indent=2))
        print(f"CSV={filename} COLUMN_VALIDATION=PASS")
    analysis["csv_audit"] = csv_audit
    (OUT / "analysis_summary.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("RUN_COUNTS=" + json.dumps(analysis["run_counts"], sort_keys=True))
    print(f"SUMMARY_ROWS={len(summary_rows)} PAIRED_ROWS={len(paired_rows)}")
    print(f"SAVED={OUT / 'summary.csv'}")
    print(f"SAVED={OUT / 'paired_differences.csv'}")
    print(f"SAVED={OUT / 'analysis_summary.json'}")


DATASET_ORDER = ("breast_cancer", "heart_disease", "parkinsons", "diabetes")


if __name__ == "__main__":
    main()
