"""Quick real benchmark: breast_cancer, logistic vs tiny VQC, 2-fold CV, 1 seed.

Usage: python scripts/quick_benchmark.py
Proves F1-F7 end to end with real numbers. Not a fixture.
"""

from __future__ import annotations

import json
import time

from qednet.data import DatasetRegistry
from qednet.evaluation.benchmark import make_preprocess_pipeline, nested_cv
from qednet.models import VQC
from qednet.models.classical import LogisticRegressionModel
from qednet.preprocess import PreprocessConfig


def main() -> None:
    import numpy as np

    bundle = DatasetRegistry.load("breast_cancer")
    X, y = bundle.X, bundle.y
    # smoke-test subsample: 100 stratified rows for speed
    rng = np.random.RandomState(0)
    keep = np.concatenate([
        rng.choice(np.where(y == c)[0], 50, replace=False) for c in (0, 1)
    ])
    rng.shuffle(keep)
    X, y = X[keep], y[keep]
    print(f"dataset: {bundle.metadata['name']} X={X.shape} pos_rate={y.mean():.3f}", flush=True)

    pre_cfg = PreprocessConfig(select_k=8, compress_dim=4)
    pre_fn = lambda: make_preprocess_pipeline(pre_cfg)

    models = {
        "logistic": LogisticRegressionModel(),
        "vqc_tiny": VQC(n_qubits=4, n_layers=1, max_iter=3, seed=42),
    }
    out = {}
    for name, model in models.items():
        t0 = time.time()
        res = nested_cv(model, X, y, pre_fn, outer_folds=2, random_state=42,
                        calibrate=False, n_bootstrap=50)
        dt = time.time() - t0
        agg = res["aggregated_metrics"]
        summ = {k: round(v["mean"], 4) for k, v in agg.items()}
        print(f"{name}: {json.dumps(summ)} ({dt:.1f}s)", flush=True)
        out[name] = {"metrics": summ, "seconds": round(dt, 1)}

    with open("quick_benchmark.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote quick_benchmark.json")


if __name__ == "__main__":
    main()
