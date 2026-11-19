import json
from pathlib import Path

path = Path(__file__).resolve().parent / "quantum_raw.json"
rows = json.loads(path.read_text(encoding="utf-8"))
changed = 0
for row in rows:
    if row.get("model") == "qsvm" and row.get("trainable_parameters") is None:
        row["trainable_parameters"] = 0
        changed += 1
path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps({"updated_qsvm_rows": changed, "total_quantum_rows": len(rows)}, indent=2))
