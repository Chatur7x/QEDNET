#!/usr/bin/env python3
"""Reproduce a QED-Net 2.0 experiment from its configuration.

    python scripts/reproduce.py --config configs/experiments/breast_cancer.yaml
    python scripts/reproduce.py --config ... --full   # 10 seeds, inner tuning on

The reproduction process regenerates the relevant benchmark output from the
stored configuration, seeds and package versions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce a QED-Net experiment")
    p.add_argument("--config", required=True, help="experiment YAML path")
    p.add_argument("--full", action="store_true",
                   help="run the full research protocol (10 seeds)")
    p.add_argument("--exp-id", default=None, help="override experiment id")
    p.add_argument("--no-explain", action="store_true")
    args = p.parse_args()

    from qednet.config import load_config
    from qednet.data.datasets import register_builtins
    from qednet.data.ingestion import DatasetRegistry
    from qednet.train.runner import run_experiment
    from qednet.tracking.store import ExperimentStore

    cfg = load_config(ROOT / args.config if not args.config.startswith("/") else args.config)
    if args.full:
        cfg.training.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.exp_id:
        cfg.experiment_id = args.exp_id
    elif args.full:
        cfg.experiment_id = cfg.experiment_id + "_full"

    import os
    os.chdir(ROOT)

    store = ExperimentStore()
    reg = DatasetRegistry()
    register_builtins(reg)
    result = run_experiment(cfg, store, reg,
                            run_explainability=not args.no_explain)

    print("\n=== REPRODUCTION SUMMARY ===")
    print(f"experiment: {cfg.experiment_id}")
    print(f"dataset:    {result['dataset']}")
    print(f"protocol:   {result['protocol']['outer_folds']}-fold outer x "
          f"{result['protocol']['inner_folds']}-fold inner, seeds {cfg.training.seeds}")
    for k, m in sorted(result["models"].items(),
                       key=lambda kv: -(kv[1]["summary"]["auroc"]["mean"] or 0)):
        print(f"  {k:<22} AUROC {m['summary']['auroc']['mean']:.4f} "
              f"+/- {m['summary']['auroc']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
