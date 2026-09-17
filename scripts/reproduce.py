"""Reproduce a benchmark from a YAML config. Real run, no fixtures.

python scripts/reproduce.py --config configs/experiments/breast_cancer_vqc.yaml \
    --compare logistic_regression xgboost vqc --seeds 42 --folds 2
"""

from __future__ import annotations

import argparse
import json

from qednet.data import DatasetRegistry
from qednet.evaluation.benchmark import make_preprocess_pipeline, nested_cv
from qednet.preprocess import PreprocessConfig
from qednet.train.config import ExperimentConfig, create_model, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--compare", nargs="+", default=["logistic_regression", "vqc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    seeds = args.seeds or cfg.training.seeds or [cfg.seed]
    folds = args.folds or cfg.training.outer_folds

    bundle = DatasetRegistry.load(cfg.dataset.name)
    X, y = bundle.X, bundle.y
    pre_cfg = PreprocessConfig(**{k: v for k, v in cfg.preprocess.__dict__.items()})
    print(f"{cfg.dataset.name}: X={X.shape} folds={folds} seeds={seeds}", flush=True)

    out: dict = {"dataset": cfg.dataset.name, "models": {}}
    for mtype in args.compare:
        per_seed = []
        for s in seeds:
            from dataclasses import replace
            mc = replace(cfg.model, type=mtype)
            nq = min(pre_cfg.compress_dim, 8)
            model = create_model(mc, nq, X.shape[1], s)
            r = nested_cv(model, X, y, lambda: make_preprocess_pipeline(pre_cfg),
                          outer_folds=folds, random_state=s, calibrate=False,
                          n_bootstrap=args.bootstrap)
            a = r["aggregated_metrics"]["auroc"]
            per_seed.append({"mean": a["mean"], "ci_lower": a["ci_lower"], "ci_upper": a["ci_upper"]})
            print(f"  {mtype} seed={s} auroc={a['mean']:.4f}", flush=True)
        import numpy as np
        means = [p["mean"] for p in per_seed]
        out["models"][mtype] = {"mean": float(np.mean(means)), "std": float(np.std(means)),
                                "per_seed": per_seed}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
