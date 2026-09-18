"""QED-Net 2.0 command-line interface.

All commands emit STRICT JSON on stdout (logs go to stderr) so the Next.js
API layer and scripts can consume them deterministically.

    python -m qednet.cli list-datasets
    python -m qednet.cli validate --dataset breast_cancer
    python -m qednet.cli upload --csv path.csv --target col --name mydata
    python -m qednet.cli run-experiment --config configs/experiments/x.yaml
    python -m qednet.cli list-experiments
    python -m qednet.cli get-experiment --id exp_001
    python -m qednet.cli explain --id exp_001 [--model vqc]
    python -m qednet.cli certificate --id exp_001
    python -m qednet.cli predict --id exp_001 --input-json '{"x": [...]}'
    python -m qednet.cli cascade-eval --id exp_001
    python -m qednet.cli env
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np

# silence noisy third-party loggers for clean JSON stdout
for noisy in ("qednet",):
    logging.getLogger(noisy).addHandler(logging.NullHandler())

ROOT = Path(__file__).resolve().parents[2]


def _stdout_json(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _setup(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stderr,
                        format="%(name)s %(levelname)s %(message)s")
    if verbose:
        for name in ("qednet", ):
            logging.getLogger(name).setLevel(logging.INFO)


def _registry():
    from .data.datasets import register_builtins
    from .data.ingestion import DatasetRegistry
    reg = DatasetRegistry()
    register_builtins(reg)
    return reg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_list_datasets(_args) -> None:
    reg = _registry()
    _stdout_json({"datasets": reg.list_datasets()})


def cmd_validate(args) -> None:
    from .data.datasets import load_dataset
    from .data.validation import validate_frame
    reg = _registry()
    df = load_dataset(args.dataset, reg)
    target = reg.get(args.dataset)["target"]
    rep = validate_frame(df, target, args.dataset)
    _stdout_json(rep.to_dict())


def cmd_upload(args) -> None:
    from .data.ingestion import DatasetRegistry, ingest_csv
    reg = DatasetRegistry()
    try:
        rec = ingest_csv(args.csv, target=args.target, dataset_name=args.name)
    except Exception as exc:  # noqa: BLE001
        _stdout_json({"ok": False, "error": str(exc)})
        return
    # validate right away
    from .data.validation import validate_frame
    import pandas as pd
    df = pd.read_csv(args.csv)
    rep = validate_frame(df, rec.target, rec.name)
    rec.validation_status = "passed" if rep.passed else "failed"
    reg.register(rec)
    _stdout_json({"ok": rep.passed, "dataset": rec.to_dict(),
                  "validation": rep.to_dict()})


def cmd_run_experiment(args) -> None:
    from .config import load_config
    from .tracking.store import ExperimentStore
    from .train.runner import run_experiment
    cfg = load_config(args.config)
    if args.exp_id:
        cfg.experiment_id = args.exp_id
    store = ExperimentStore()
    reg = _registry()
    try:
        result = run_experiment(cfg, store, reg,
                                run_explainability=not args.no_explain)
        _stdout_json({
            "ok": True,
            "experiment_id": cfg.experiment_id,
            "dataset": result["dataset"],
            "runtime_s": result["total_runtime_s"],
            "models": {k: round(v["summary"]["auroc"]["mean"] or 0, 4)
                       for k, v in result["models"].items()},
            "evidence_classification":
                (result.get("ablation") or {}).get("entanglement_contribution"),
            "certificate": _cert_summary(cfg.experiment_id, store),
            "result_path": str(store.runs_dir / cfg.experiment_id /
                               "result.json"),
        })
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc(file=sys.stderr)
        _stdout_json({"ok": False, "error": str(exc),
                      "error_type": type(exc).__name__})


def _cert_summary(exp_id, store):
    try:
        cert = store.load_json(exp_id, "certificate")
        return {
            "classification": cert.get("final_evidence_classification"),
            "rationale": cert.get("final_evidence_rationale"),
        }
    except FileNotFoundError:
        return None


def cmd_list_experiments(_args) -> None:
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    _stdout_json({"experiments": store.list_experiments()})


def cmd_get_experiment(args) -> None:
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    try:
        result = store.get_result(args.id)
        _stdout_json({"ok": True, "result": result})
    except FileNotFoundError as exc:
        _stdout_json({"ok": False, "error": str(exc)})


def cmd_explain(args) -> None:
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    try:
        data = store.load_json(args.id, "explainability")
        if args.model and args.model in data:
            data = {args.model: data[args.model]}
        _stdout_json({"ok": True, "explainability": data})
    except FileNotFoundError as exc:
        _stdout_json({"ok": False, "error": str(exc)})


def cmd_certificate(args) -> None:
    from .certificate.certificate import certificate_to_markdown
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    try:
        cert = store.load_json(args.id, "certificate")
        _stdout_json({"ok": True, "certificate": cert,
                      "markdown": certificate_to_markdown(cert)})
    except FileNotFoundError as exc:
        _stdout_json({"ok": False, "error": str(exc)})


def cmd_predict(args) -> None:
    """Single-sample prediction through the F12 cascade with F11 output."""
    from .cascade.cascade import cascade_predict_single
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    try:
        x = json.loads(args.input_json)
        if isinstance(x, dict):
            x = x.get("x") or x.get("features") or x.get("values")
        x = [float(v) for v in x]
        pipe = store.load_binary(args.id, "pipeline")
        models = store.load_binary(args.id, "models")
        conformal = store.load_binary(args.id, "conformal")
        from .config import load_config
        cfg_path = store.runs_dir / args.id / "config.yaml"
        band_lo, band_hi = 0.35, 0.65
        screening_key = "logistic_regression"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            band_lo = cfg.cascade.low_confidence_min
            band_hi = cfg.cascade.low_confidence_max
            screening_key = cfg.cascade.screening_model
        quantum_key = next((k for k in ("vqc", "reupload", "hybrid_qnn",
                                        "qsvm") if k in models), None)
        if quantum_key is None or screening_key not in models:
            _stdout_json({"ok": False,
                          "error": "experiment lacks cascade model pair"})
            return
        from .uncertainty.abstention import AbstentionRule
        rule = AbstentionRule(band_low=band_lo, band_high=band_hi,
                              conformal=conformal)
        out = cascade_predict_single(
            np.array(x), models[screening_key], models[quantum_key], pipe,
            band_low=band_lo, band_high=band_hi, abstention=rule)
        out["experiment_id"] = args.id
        out["abstention_rule"] = rule.describe()
        out["ok"] = True
        _stdout_json(out)
    except FileNotFoundError as exc:
        _stdout_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        _stdout_json({"ok": False, "error": str(exc),
                      "error_type": type(exc).__name__})


def cmd_cascade_eval(args) -> None:
    from .tracking.store import ExperimentStore
    store = ExperimentStore()
    try:
        result = store.get_result(args.id)
        _stdout_json({"ok": True, "cascade": result.get("cascade")})
    except FileNotFoundError as exc:
        _stdout_json({"ok": False, "error": str(exc)})


def cmd_env(_args) -> None:
    """Environment report: versions of core packages.

    Reports `ok: false` when any CRITICAL package is missing so the health
    endpoint surfaces a degraded backend honestly instead of silently
    returning null versions. Critical = required to run experiments/predict.
    Optional (imblearn, mlflow) degrade features but don't invalidate the core.
    """
    import platform
    critical = ("numpy", "sklearn", "pandas", "xgboost", "shap",
                "pennylane", "yaml")
    mods = {}
    for mod in ("numpy", "sklearn", "pandas", "xgboost", "shap", "pennylane",
                "imblearn", "mlflow", "yaml", "autograd"):
        try:
            m = __import__(mod)
            mods[mod] = getattr(m, "__version__", "present")
        except ImportError:
            mods[mod] = None
    missing_critical = [m for m in critical if mods.get(m) is None]
    from . import RESEARCH_STATUS, __version__
    _stdout_json({
        "qednet_version": __version__,
        "python": platform.python_version(),
        "packages": mods,
        "critical_missing": missing_critical,
        "ok": not missing_critical,
        "research_status": RESEARCH_STATUS,
    })


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="qednet")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-datasets").set_defaults(func=cmd_list_datasets)

    sp = sub.add_parser("validate")
    sp.add_argument("--dataset", required=True)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("upload")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--target", default=None)
    sp.add_argument("--name", default=None)
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("run-experiment")
    sp.add_argument("--config", required=True)
    sp.add_argument("--exp-id", default=None)
    sp.add_argument("--no-explain", action="store_true")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_run_experiment)

    sub.add_parser("list-experiments").set_defaults(func=cmd_list_experiments)

    sp = sub.add_parser("get-experiment")
    sp.add_argument("--id", required=True)
    sp.set_defaults(func=cmd_get_experiment)

    sp = sub.add_parser("explain")
    sp.add_argument("--id", required=True)
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("certificate")
    sp.add_argument("--id", required=True)
    sp.set_defaults(func=cmd_certificate)

    sp = sub.add_parser("predict")
    sp.add_argument("--id", required=True)
    sp.add_argument("--input-json", required=True)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("cascade-eval")
    sp.add_argument("--id", required=True)
    sp.set_defaults(func=cmd_cascade_eval)

    sub.add_parser("env").set_defaults(func=cmd_env)

    args = p.parse_args(argv)
    _setup(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
