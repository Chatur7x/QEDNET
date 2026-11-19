import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def kill_tree(pid):
    """Use Windows taskkill to terminate a process and its descendants."""
    killer = subprocess.Popen(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = killer.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        killer.kill()
        stdout, stderr = killer.communicate()
        raise RuntimeError("taskkill did not finish")
    return {"returncode": killer.returncode, "stdout": stdout, "stderr": stderr}


def run_child(command, timeout_seconds, cwd=ROOT):
    """Run one child; on timeout kill its process tree and measure in parent."""
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        elapsed = time.perf_counter() - started
        return {
            "status": "completed" if process.returncode == 0 else "error",
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "kill": None,
        }
    except subprocess.TimeoutExpired:
        kill = kill_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        elapsed = time.perf_counter() - started
        return {
            "status": "timeout",
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "kill": kill,
        }


def self_test(timeout_seconds=5, sleep_seconds=30):
    cmd = [
        sys.executable,
        str(OUT / "timeout_sleep_child.py"),
        "--seconds",
        str(sleep_seconds),
    ]
    result = run_child(cmd, timeout_seconds)
    passed = (
        result["status"] == "timeout"
        and result["elapsed_seconds"] < sleep_seconds
        and "SLEEP_COMPLETED" not in result["stdout"]
    )
    record = {
        "test": "parent_process_tree_timeout",
        "command": cmd,
        "configured_timeout_seconds": timeout_seconds,
        "child_sleep_seconds": sleep_seconds,
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "child_returncode": result["returncode"],
        "child_stdout": result["stdout"],
        "child_stderr": result["stderr"],
        "taskkill": result["kill"],
        "passed": passed,
    }
    (OUT / "timeout_selftest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit(1)


def append_json(path, record):
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = []
    key = (record.get("dataset"), record.get("feature_setting"), record.get("split_variant"), record.get("model"), record.get("seed"), record.get("fold"), record.get("run_type", "benchmark"))
    if any((r.get("dataset"), r.get("feature_setting"), r.get("split_variant"), r.get("model"), r.get("seed"), r.get("fold"), r.get("run_type", "benchmark")) == key for r in rows):
        return
    rows.append(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    with path.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())


def run_fold(dataset, model, seed, fold, feature_setting, split_variant, timeout_seconds=900, run_type="benchmark"):
    is_quantum = model in {"qsvm", "vqc", "reupload"}
    raw_path = OUT / ("quantum_raw.json" if is_quantum else "classical_raw.json")
    safe_run_type = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_type)
    stem = f"{dataset}__{feature_setting}__{split_variant}__{model}__seed{seed}__fold{fold}__{safe_run_type}"
    stdout_path = OUT / "fold_logs" / f"{stem}.stdout.txt"
    stderr_path = OUT / "fold_logs" / f"{stem}.stderr.txt"
    command_path = OUT / "fold_logs" / f"{stem}.command.json"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(OUT / "run_fold.py"),
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--fold", str(fold),
        "--feature-setting", feature_setting,
        "--split-variant", split_variant,
        "--run-type", run_type,
    ]
    command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
    result = run_child(command, timeout_seconds)
    stdout_path.write_text(result["stdout"], encoding="utf-8")
    stderr_path.write_text(result["stderr"], encoding="utf-8")
    if result["status"] == "completed":
        child_record = json.loads(result["stdout"].strip().splitlines()[-1])
        record = {**child_record, "status": "completed", "parent_elapsed_seconds": result["elapsed_seconds"], "timeout_seconds": timeout_seconds, "child_command_file": str(command_path.relative_to(ROOT))}
    else:
        record = {
            "dataset": dataset, "model": model, "seed": seed, "fold": fold,
            "feature_setting": feature_setting, "split_variant": split_variant,
            "run_type": run_type,
            "status": result["status"], "parent_elapsed_seconds": result["elapsed_seconds"],
            "timeout_seconds": timeout_seconds, "returncode": result["returncode"],
            "child_command_file": str(command_path.relative_to(ROOT)),
            "error": result["stderr"], "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
        }
        if result["kill"] is not None:
            record["taskkill"] = result["kill"]
    append_json(raw_path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--feature-setting", choices=("A", "B"))
    parser.add_argument("--split-variant", default="random")
    parser.add_argument("--run-type", default="benchmark")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (args.dataset, args.model, args.seed, args.fold, args.feature_setting)
    if any(value is None for value in required):
        parser.error("dataset, model, seed, fold, and feature-setting are required outside self-test")
    result = run_fold(args.dataset, args.model, args.seed, args.fold, args.feature_setting, args.split_variant, args.timeout, args.run_type)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
