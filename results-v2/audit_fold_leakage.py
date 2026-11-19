import json
from pathlib import Path

out = Path(__file__).resolve().parent
source = (out / "run_fold.py").read_text(encoding="utf-8")
manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
checks = {
    "preprocessor_fit_transform_train_only": "preprocessor.fit_transform(X[train_idx], y[train_idx])" in source,
    "test_preprocessor_transform_only": "preprocessor.transform(X[test_idx])" in source,
    "model_fit_train_only": "model.fit(X_train, y[train_idx])" in source,
    "selector_and_pca_are_in_fold_local_preprocessor": "steps.extend([" in source and "SelectKBest" in source and "PCA(n_components=4" in source,
}
grouped = manifest["datasets"]["parkinsons"]["grouped_folds"]
grouped_audit = [{
    "fold": ix,
    "train_subject_count": fold["train_subject_count"],
    "test_subject_count": fold["test_subject_count"],
    "overlap_count": len(fold["overlap_subject_ids"]),
} for ix, fold in enumerate(grouped)]
checks["all_grouped_parkinsons_folds_have_zero_subject_overlap"] = all(row["overlap_count"] == 0 for row in grouped_audit)
random_folds = manifest["datasets"]["parkinsons"]["random_folds"]
random_audit = [{"fold": ix, "overlap_count": len(fold["overlap_subject_ids"])} for ix, fold in enumerate(random_folds)]
result = {"checks": checks, "passed": all(checks.values()), "grouped_parkinsons": grouped_audit, "random_parkinsons": random_audit}
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
