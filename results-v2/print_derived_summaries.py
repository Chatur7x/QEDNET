import json
import statistics
from pathlib import Path

out = Path(__file__).resolve().parent
analysis = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
print("PAIRED_BEST_OF_FOLD=" + json.dumps(analysis["paired_best"], separators=(",", ":"), allow_nan=False))
print("PARAMETER_MATCHED=" + json.dumps(analysis["parameter_matched"], separators=(",", ":"), allow_nan=False))
repeats = [json.loads(line) for line in (out / "repeat_qsvm_timing.raw.txt").read_text(encoding="utf-8").splitlines() if line.startswith("{")]
fit_times = [r["fit_seconds"] for r in repeats if r["status"] == "completed"]
timing = {
    "dataset": "heart_disease", "model": "qsvm", "seed": 42, "fold": 0,
    "n_samples": 80, "n_train": 53, "n_test": 27,
    "completed_repeats": len(fit_times), "fit_seconds_each": fit_times,
    "min_fit_seconds": min(fit_times), "median_fit_seconds": statistics.median(fit_times),
    "max_fit_seconds": max(fit_times),
}
print("QSVM_REPEAT_SUMMARY=" + json.dumps(timing, separators=(",", ":"), allow_nan=False))
