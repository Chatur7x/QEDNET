import json
import subprocess
import sys
from pathlib import Path

from runner import run_child

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
command = [
    sys.executable,
    str(OUT / "run_fold.py"),
    "--dataset", "breast_cancer",
    "--model", "logistic_regression",
    "--seed", "42",
    "--fold", "0",
    "--feature-setting", "A",
    "--split-variant", "random",
]
result = run_child(command, 900)
if result["status"] != "completed":
    print(json.dumps({"command": command, "run": result, "passed": False}, indent=2))
    raise SystemExit(2)
current = json.loads(result["stdout"].strip().splitlines()[-1])
baseline_path = ROOT / "results/shared80_classical_raw.json"
baseline_rows = json.loads(baseline_path.read_text(encoding="utf-8"))
matches = [r for r in baseline_rows if r.get("dataset") == "breast_cancer" and r.get("model") == "logistic_regression" and r.get("seed") == 42 and r.get("fold") == 0]
if len(matches) != 1:
    print(json.dumps({"command": command, "baseline_matches": len(matches), "passed": False}, indent=2))
    raise SystemExit(3)
baseline = matches[0]
record = {
    "command": command,
    "status": result["status"],
    "elapsed_seconds": result["elapsed_seconds"],
    "current": {"auc": current["auc"], "accuracy": current["accuracy"], "balanced_accuracy": current["balanced_accuracy"], "n_train": current["n_train"], "n_test": current["n_test"]},
    "prior_record": {"auc": baseline["auc"], "accuracy": baseline["accuracy"], "balanced_accuracy": baseline["balanced_accuracy"], "n_train": baseline["n_train"], "n_test": baseline["n_test"]},
    "auc_exact_match": current["auc"] == baseline["auc"],
    "accuracy_exact_match": current["accuracy"] == baseline["accuracy"],
    "balanced_accuracy_exact_match": current["balanced_accuracy"] == baseline["balanced_accuracy"],
    "split_size_match": current["n_train"] == baseline["n_train"] and current["n_test"] == baseline["n_test"],
}
record["passed"] = all(record[k] for k in ("auc_exact_match", "accuracy_exact_match", "balanced_accuracy_exact_match", "split_size_match"))
(OUT / "sanity_check_reconciled.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
if not record["passed"]:
    raise SystemExit(1)
