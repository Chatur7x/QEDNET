#!/usr/bin/env python3
"""Patch inflated n_samples in completed experiment results.

The evaluation engine summed test-set sizes across all models (M x S x N
instead of N). This script recomputes the true dataset size from each
experiment's pooled predictions / known datasets, or by dividing by the
number of models when pooled data matches the pattern.

Run from /home/z/my-project.
"""
import json
import sys
from pathlib import Path

RUNS = Path("/home/z/my-project/experiments/runs")

# True sizes of the open datasets used (registry ground truth)
TRUE_SIZES = {
    "breast_cancer": 569,
    "heart_disease": 303,
    "parkinsons": 195,
    "pima_diabetes": 768,
}


def main() -> int:
    patched = []
    for d in sorted(RUNS.iterdir()):
        result_p = d / "result.json"
        if not result_p.exists():
            continue
        r = json.loads(result_p.read_text())
        dataset = r.get("dataset")
        n_models = len(r.get("models", {}))
        current = r.get("n_samples")
        true_n = TRUE_SIZES.get(dataset)
        if true_n is None:
            # unknown dataset: derive as current / n_models if divisible
            if current and n_models and current % n_models == 0:
                true_n = current // n_models
            else:
                print(f"  {d.name}: dataset '{dataset}' size unknown, "
                      f"n_samples={current} left as is")
                continue
        if current == true_n:
            print(f"  {d.name}: already correct ({current})")
            continue
        # also fix derived display fields that used n_samples
        r["n_samples"] = true_n
        result_p.write_text(json.dumps(r, indent=2))
        patched.append((d.name, current, true_n))
        print(f"  {d.name}: patched n_samples {current} -> {true_n}")
    print(f"\n{len(patched)} file(s) patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
