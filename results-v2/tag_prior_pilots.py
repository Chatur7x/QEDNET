import json
from pathlib import Path

out = Path(__file__).resolve().parent
path = out / "quantum_raw.json"
rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
changed = 0
for row in rows:
    if row.get("dataset") == "heart_disease" and row.get("model") in {"qsvm", "vqc", "reupload"} and row.get("fold") == 0 and "run_type" not in row:
        row["run_type"] = "timing_gate_two_stage"
        row["sample_protocol"] = "80-row draw from fixed 150-row sample; timing pilot before final direct-sampling manifest"
        changed += 1
path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps({"file": str(path), "rows_tagged": changed, "total_rows": len(rows)}, indent=2))
