import json
from pathlib import Path

out = Path(__file__).resolve().parent
path = out / "classical_raw.json"
rows = json.loads(path.read_text(encoding="utf-8"))
replace_keys = {
    ("heart_disease", "parameter_matched_mlp_reupload"),
    ("diabetes", "parameter_matched_mlp_vqc"),
}
removed = [
    row for row in rows
    if (row.get("dataset"), row.get("model")) in replace_keys
    and row.get("feature_setting") == "B"
    and row.get("status") == "not_measured"
    and row.get("run_type") == "benchmark"
]
remaining = [row for row in rows if row not in removed]
(out / "pmmlp_not_measured_superseded.json").write_text(json.dumps(removed, indent=2), encoding="utf-8")
path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
print(json.dumps({"removed_placeholder_rows": len(removed), "remaining_rows": len(remaining), "saved_superseded_records": str(out / "pmmlp_not_measured_superseded.json")}, indent=2))
