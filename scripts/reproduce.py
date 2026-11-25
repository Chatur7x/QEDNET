"""Run reproducible CV benchmarks with the repository's real estimators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from qednet.data import DatasetRegistry
from qednet.evaluation.benchmark import (
    get_default_models,
    make_preprocess_pipeline,
    nested_cv,
)
from qednet.preprocess import PreprocessConfig


def _load_config(path: str) -> tuple[dict, PreprocessConfig]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    dataset = config.get("dataset", {})
    preprocess_values = config.get("preprocess")
    if preprocess_values is None:
        preprocess_values = {
            "compress_method": dataset.get("compression", "pca"),
            "compress_dim": dataset.get("target_dim", 8),
            "random_state": dataset.get("random_state", config.get("seed", 42)),
        }
    return config, PreprocessConfig(**preprocess_values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--compare", nargs="+", default=["logistic_regression", "vqc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--out", default="reports/benchmark.json")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    config, preprocess_config = _load_config(args.config)
    training = config.get("training", {})
    seed_config = training.get("seeds", config.get("seed", 42))
    if isinstance(seed_config, int):
        seed_config = [seed_config]
    seeds = args.seeds or seed_config or [42]
    folds = args.folds or training.get("outer_folds", 5)
    inner_folds = training.get("inner_folds", 3)
    dataset_name = config.get("dataset", {}).get("name", "breast_cancer")
    model_config = config.get("model", {})
    n_qubits = min(model_config.get("n_qubits", 8), preprocess_config.compress_dim)

    bundle = DatasetRegistry.load(dataset_name)
    x, y = bundle.X, bundle.y
    available = get_default_models(
        n_qubits=n_qubits, n_features=preprocess_config.compress_dim, seed=seeds[0]
    )
    unsupported = [name for name in args.compare if name not in available]
    if unsupported:
        parser.error(f"unsupported model(s): {unsupported}; available: {sorted(available)}")

    print(f"{dataset_name}: X={x.shape} folds={folds} seeds={seeds}", flush=True)
    output: dict = {"dataset": dataset_name, "models": {}}
    for model_name in args.compare:
        per_seed = []
        for seed in seeds:
            model = get_default_models(
                n_qubits=n_qubits, n_features=preprocess_config.compress_dim, seed=seed
            )[model_name]
            result = nested_cv(
                model,
                x,
                y,
                lambda: make_preprocess_pipeline(preprocess_config),
                outer_folds=folds,
                inner_folds=inner_folds,
                random_state=seed,
                calibrate=False,
                n_bootstrap=args.bootstrap,
                groups=bundle.groups,
            )
            auroc = result["aggregated_metrics"]["auroc"]
            per_seed.append({
                "seed": seed,
                "mean": auroc["mean"],
                "ci_lower": auroc["ci_lower"],
                "ci_upper": auroc["ci_upper"],
            })
            print(f"  {model_name} seed={seed} auroc={auroc['mean']:.4f}", flush=True)
        means = [entry["mean"] for entry in per_seed]
        output["models"][model_name] = {
            "mean": float(np.mean(means)),
            "std": float(np.std(means)),
            "per_seed": per_seed,
        }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
