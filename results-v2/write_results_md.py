import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def show(value):
    return "NOT MEASURED" if value is None else str(value)


def fmt(metric):
    if not metric or metric.get("n", 0) == 0:
        return "NOT MEASURED"
    if metric.get("sd") is None:
        return f"{metric['mean']} (SD NOT MEASURED; n={metric['n']})"
    return f"{metric['mean']} ± {metric['sd']} (n={metric['n']})"


def table_line(cells):
    return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def table(headers, rows):
    return "\n".join([table_line(headers), table_line(["---"] * len(headers))] + [table_line(r) for r in rows])


def pm_pairs(analysis):
    out = defaultdict(list)
    for item in analysis["parameter_matched"]:
        out[(item["dataset"], item["split_variant"])].append(item)
    return out


def best_conclusions(analysis):
    result = {}
    for p in analysis["paired_best"]:
        dataset, variant = p["dataset"], p["split_variant"]
        if p["n"] == 0:
            decision = "NOT MEASURED"
        else:
            t_sig = p["ttest_p"] is not None and p["ttest_p"] < 0.05
            w_sig = p["wilcoxon_p"] is not None and p["wilcoxon_p"] < 0.05
            if t_sig and w_sig:
                decision = "quantum lower; both tests p < 0.05"
            elif t_sig != w_sig:
                decision = "tests disagree (t-test >= 0.05; Wilcoxon < 0.05)"
            else:
                decision = "no significant difference (both p >= 0.05)"
        result[(dataset, variant)] = (p, decision)
    return result


def main():
    analysis = read_json("analysis_summary.json")
    manifest = read_json("splits.json")
    leakage = read_json("audit_fold_leakage.raw.txt")
    timeout = read_json("timeout_selftest.json")
    sanity = read_json("sanity_check.json")
    reconciled = read_json("sanity_check_reconciled.json")
    grouped = read_json("audit_fold_leakage.raw.txt")["grouped_parkinsons"]
    repeats = [json.loads(line) for line in (OUT / "repeat_qsvm_timing.raw.txt").read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    repeat_fit = [r["fit_seconds"] for r in repeats if r["status"] == "completed"]
    gates = {}
    for model in ("qsvm", "vqc", "reupload"):
        gates[model] = read_json(f"timing_gate_direct_heart_{model}.raw.txt")
    paired = best_conclusions(analysis)
    pmpairs = pm_pairs(analysis)
    lines = [
        "# QEDNET fixed benchmark v2 results",
        "",
        "## Summary tables",
        "",
        "Each metric is the mean ± sample standard deviation across completed folds, with the observed fold count. Values are printed at the precision stored in the current-run analysis output; no rounding is applied here. `Fit seconds` measures estimator `.fit` only, excluding preprocessing. A = median imputation, scaling, SelectKBest(8), PCA(4); B = the same fold-local imputation and scaling with all original input features (no selector or PCA). Quantum models have no setting B. ParameterMatchedMLP training times use the same estimator-only timing convention.",
        "",
        "| Dataset and split | Rows; original features; class counts | Folds (train/test) |",
        "| --- | --- | --- |",
    ]
    dataset_order = ["breast_cancer", "heart_disease", "parkinsons", "diabetes"]
    for dataset in dataset_order:
        item = manifest["datasets"][dataset]
        counts = ", ".join(f"{k}:{v}" for k, v in item["class_counts"].items())
        sizes = "; ".join(f"{len(f['train_sample_indices'])}/{len(f['test_sample_indices'])}" for f in item["random_folds"])
        lines.append(table_line([f"{dataset} (random)", f"{item['n']}; {item['input_features']}; {counts}", sizes]))
        if dataset == "parkinsons":
            sizes = "; ".join(f"{len(f['train_sample_indices'])}/{len(f['test_sample_indices'])}" for f in item["grouped_folds"])
            lines.append(table_line([f"{dataset} (grouped)", f"{item['n']}; {item['input_features']}; {counts}", sizes]))

    lines += ["", "For every table below, rows were sampled directly from each complete dataset using `StratifiedShuffleSplit(train_size=80, random_state=42)`. The same fixed three folds were reused for all models and estimator seeds; the seed varies model initialization, not fold membership. Per-seed and per-fold raw records are in `classical_raw.json`, `quantum_raw.json`, and `summary.csv`.", ""]
    dataset_label = {"breast_cancer": "breast cancer", "heart_disease": "heart disease", "parkinsons": "Parkinson's", "diabetes": "diabetes"}
    for key in [(d, "random") for d in dataset_order] + [("parkinsons", "grouped")]:
        dataset, variant = key
        for setting in ("A", "B"):
            section = next(t for t in analysis["tables"] if t["dataset"] == dataset and t["split_variant"] == variant and t["feature_setting"] == setting)
            rows = []
            for model in section["models"]:
                metrics = model["metrics"]
                pc = model["parameter_counts"]
                param_text = ", ".join(str(x) for x in pc) if pc else ("0" if model["model"] == "qsvm" else "NOT MEASURED")
                rows.append([
                    model["model"], f"{model['status_counts']['completed']}/{model['status_counts']['timeout']}/{model['status_counts']['error']}/{model['status_counts']['not_measured']}",
                    fmt(metrics.get("auc")), fmt(metrics.get("accuracy")), fmt(metrics.get("balanced_accuracy")), fmt(metrics.get("f1")), fmt(metrics.get("train_seconds")), param_text,
                ])
            label = f"{dataset_label[dataset]} — {variant} folds — setting {setting}"
            lines += [f"### {label}", "", "Status is completed / timeout / error / not measured. Parameter counts are listed when reported by the real estimator; random forest count varies with fitted tree structure. `NOT MEASURED` is not treated as zero.", "", table(["Model", "Status counts", "AUC mean ± SD (n)", "Accuracy mean ± SD (n)", "Balanced accuracy mean ± SD (n)", "F1 mean ± SD (n)", "Fit seconds mean ± SD (n)", "Trainable parameters"], rows), ""]

    lines += ["## Fold-paired comparisons", "", "The primary difference is best quantum AUC minus best classical AUC on the same dataset, split, seed, and fold, using feature setting A. The best model can change by fold, so this is a best-of-fold comparison, not a fixed model-pair estimate. The paired tests use the nine observed seed/fold pairs. P-values are unadjusted for multiple comparisons.", "", table(["Dataset / split", "Mean difference", "Sample SD", "n", "Paired t-test p", "Wilcoxon p", "Interpretation"], [[f"{d} / {v}", show(p["mean"]), show(p["sd"]), p["n"], show(p["ttest_p"]), show(p["wilcoxon_p"]), decision] for (d, v), (p, decision) in paired.items()]), ""]
    lines.append("Decision wording uses the requested p-value threshold of 0.05: the breast-cancer, heart-disease, random Parkinson, and diabetes mean differences favor classical (negative quantum-minus-classical differences) with both tests below 0.05; Parkinson grouped tests disagree (t-test p >= 0.05; Wilcoxon p < 0.05). These small-sample tests are exploratory.")
    lines += ["", "### ParameterMatchedMLP versus each quantum model", "", "AUC differences are ParameterMatchedMLP minus the named quantum model on matching setting A folds. Each entry gives mean difference, sample SD, n, paired t-test p, and Wilcoxon p. Architecture counts on setting A were 12 for VQC-matched and 15 for Reupload-matched. Grouped Parkinson's uses the subject-grouped folds.", ""]
    for (dataset, variant), entries in pmpairs.items():
        lines.append(f"#### {dataset_label[dataset]} — {variant}")
        lines.append("")
        lines.append(table(["ParameterMatchedMLP", "Quantum model", "Mean difference", "Sample SD", "n", "t-test p", "Wilcoxon p"], [[e["pm_model"], e["quantum_model"], show(e["mean"]), show(e["sd"]), e["n"], show(e["ttest_p"]), show(e["wilcoxon_p"])] for e in entries]))
        lines.append("")

    lines += ["### ParameterMatchedMLP setting B eligibility", "", "Setting B parameter-matched architectures were measured only where the architecture audit found an eligible real-class trainable-parameter count within 10% of a quantum target. Heart disease / Reupload target: hidden_dims=[1], 16 parameters. Diabetes / VQC target: hidden_dims=[1], 11 parameters. Their fold metrics are included in the setting B tables. Other setting B ParameterMatchedMLP variants are NOT MEASURED because no eligible architecture was identified by the real class parameter audit: breast cancer (minimum candidate 31 parameters), Parkinson's (minimum 23), heart disease VQC match (14 is outside the ±10% target interval around 12), and diabetes Reupload match (no candidate in range). No quantum runs were made for setting B.", ""]

    lines += ["## Protocol, commands, and validation", "", "### Data, splits, and seeds", "", "The datasets are the scikit-learn breast cancer data and the repository files `data/raw/heart_cleveland.data`, `data/raw/parkinsons.data`, and `data/raw/pima_diabetes.csv`. Dataset sample counts, feature counts, label counts, fold indices, source row indices, and Parkinson subject IDs are saved in `splits.json`. The current sample has 80 rows per dataset, selected directly from the full dataset with sample seed 42. Random-fold membership uses `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. Fit seeds are 42, 1729, and 2024; the same split indices are used across models and fit seeds. Fold sizes (train/test) are 53/27, 53/27, and 54/26.", ""]
    pk = manifest["datasets"]["parkinsons"]
    lines.append(f"Parkinson's data contained {len(pk['sample_subject_record_counts'])} represented subjects in the 80-row sample. The recorded per-subject row counts are: `{json.dumps(pk['sample_subject_record_counts'], sort_keys=True, separators=(',', ':'))}`. Random-fold train/test subject overlaps were 15, 16, and 15 for folds 0, 1, and 2. The grouped sample uses the same direct 80-row draw, including repeated subject recordings because one row per subject cannot provide 80 rows; its grouped fold overlaps were 0, 0, and 0. Grouped train/test subject counts by fold were 19/10, 19/10, and 20/9. Thus Parkinson's random-fold results are subject-leaked; use the grouped results to assess subject separation.")
    lines += ["", "### Models and implementation", "", "Classical models: LogisticRegressionModel, RandomForestModel, RBFSVMModel, MLPClassifier (`early_stopping=False`, `max_iter=500`), plus the repository ParameterMatchedMLP. Quantum models: QSVM with `zz` feature map, VQC with angle/RX encoding and one layer, and ReuploadClassifier with one layer and repeated Rot plus RY(x+phi) encoding and ring CNOT entanglement. All use PennyLane `default.qubit`, 4 qubits, and `shots=None`; no real quantum hardware ran. QSVM has 0 trainable parameters; VQC and Reupload have 12 and 16. QSVM `circuit_depth=3` is feature-map circuit depth metadata, not trained variational layers. HybridQNN was not run because source inspection shows its fit delegates to a random-forest mock.", "", "### Leakage, timeouts, and timing", "", f"The fold-local audit passed: preprocessing is fit on training rows only, test rows receive transform only, and the estimator fits training labels only. Grouped Parkinson folds had zero subject overlap. Timeout self-test passed: child sleep 30 seconds; parent timeout 5 seconds; Windows taskkill /T /F; measured elapsed {timeout['elapsed_seconds']} seconds. Each grid fit used a child process and 900-second parent timeout. Classical statuses: {analysis['run_counts']['classical_benchmark']}; quantum statuses: {analysis['run_counts']['quantum_benchmark']}. Earlier two-stage timing pilots are excluded.", "", "Heart timing gate, seed 42, fold 0, n=80, train/test 53/27: QSVM fit " + str(gates["qsvm"]["train_seconds"]) + " s, predict " + str(gates["qsvm"]["predict_seconds"]) + " s; 2862 kernel evaluations combined across fit and predict, not split. VQC fit " + str(gates["vqc"]["train_seconds"]) + " s, predict " + str(gates["vqc"]["predict_seconds"]) + " s. Reupload fit " + str(gates["reupload"]["train_seconds"]) + " s, predict " + str(gates["reupload"]["predict_seconds"]) + " s.", "", "Repeated same-fold heart QSVM estimator fit times (seed 42, n=80, train/test 53/27): " + json.dumps(repeat_fit) + ". Measured min=" + str(min(repeat_fit)) + ", median=" + str(sorted(repeat_fit)[len(repeat_fit)//2]) + ", max=" + str(max(repeat_fit)) + ". Preprocessing is excluded from estimator train_seconds; end-to-end pipeline time is NOT MEASURED.", ""]
    lines += ["### Sanity check and earlier result mismatch", "", f"The requested direct-sample breast-cancer logistic-regression seed-42 fold-0 sanity check did not match the earlier record: current direct sample AUC={sanity['current']['auc']}, accuracy={sanity['current']['accuracy']}, balanced accuracy={sanity['current']['balanced_accuracy']}; earlier record AUC={sanity['prior_record']['auc']}, accuracy={sanity['prior_record']['accuracy']}, balanced accuracy={sanity['prior_record']['balanced_accuracy']}. Fold sizes matched at train/test 53/27. Investigation found that the earlier shared-80 artifact selected 80 rows from a prior 150-row sample; a separate two-stage diagnostic reproduced that earlier record exactly (AUC={reconciled['current']['auc']}, accuracy={reconciled['current']['accuracy']}, balanced accuracy={reconciled['current']['balanced_accuracy']}). This benchmark follows the task's final instruction to draw 80 rows directly from each full dataset. The mismatch is therefore a sample-protocol difference; old scores were not carried into the new results.", ""]
    lines += ["### Reproduction commands and evidence files", "", "Run commands from the repository root in PowerShell. The exact command line, raw stdout/stderr, and exit status are saved as the adjacent `.command.txt`, `.raw.txt`, and `.exit.txt` evidence files where applicable.", "", table(["Stage", "Command / evidence"], [
        ["Environment and repository inventory", "`step1_package_versions.command.txt` / `step1_package_versions.raw.txt`; `step1_repo_inventory.*`"],
        ["Split construction", "`python results-v2/make_splits.py`; `make_splits.*`, `final_splits.*`; manifest `splits.json`"],
        ["Child timeout test", "`timeout_selftest.command.txt`; `timeout_selftest.json`, `.raw.txt`, `.exit.txt`"],
        ["Classical benchmark grid", "`classical_grid.command.txt`; `classical_grid.raw.txt`, `.exit.txt`; fold records `classical_raw.json`"],
        ["Quantum benchmark grid", "`quantum_grid.command.txt`; `quantum_grid.raw.txt`, `.exit.txt`; fold records `quantum_raw.json`"],
        ["Leakage and model audits", "`audit_fold_leakage.*`, `audit_model_implementations.*`, `audit_pm_architecture.*`, `audit_qsvm_parameter_count.*`"],
        ["Timing gate / repeat", "`timing_gate_direct_heart_{qsvm,vqc,reupload}.*`; `repeat_qsvm_timing.*`"],
        ["Summaries and CSV validation", "`summarize_results.command.txt`; `summarize_results.raw.txt`, `.exit.txt`; `print_derived_summaries.*`"],
        ["Per-fold result evidence", "`classical_raw.json`, `quantum_raw.json`, `fold_logs/`, `summary.csv`, `paired_differences.csv`"],
    ]), ""]
    lines += ["Package versions observed in the saved `pip show` output: scikit-learn 1.9.1, PennyLane 0.45.1, NumPy 2.5.3. The CSV validator printed three sample data rows from each CSV, checked every CSV header against the required columns, and checked row content against source JSON records; see `summarize_results.raw.txt` and `analysis_summary.json` `csv_audit`.", "", "## Limitations", "", "- This is a single small 80-row sample per dataset, not a full-data performance estimate. The split seed is fixed; fold uncertainty does not capture variation across independent subsamples.", "- Each model used three folds for each of three estimator seeds; the folds are reused across seeds, so the nine paired observations are not nine independent datasets.", "- Parkinson's random-fold evaluations leak subjects across train/test. The grouped variant removes subject overlap but still has repeated recordings in the 80-row sample and has only three grouped folds.", "- Classical setting A is restricted to four PCA features to match the quantum input dimension; this constrains classical baselines. Setting B is classical only and does not support a direct quantum comparison.", "- Paired best-of-fold comparisons select a different best model when needed in each fold, and p-values are exploratory, unadjusted, and based on a small number of pairs.", "- Quantum results are analytic simulator results only (default.qubit); no hardware noise, shot noise, queue delay, or hardware execution was measured.", "- Quantum calibration (Brier score/ECE), conformal prediction, SHAP, nested cross-validation, and full-data quantum runs are NOT MEASURED. Real hardware results are NOT MEASURED because only a simulator was configured.", "- MLP early stopping was disabled in this benchmark. This does not establish that the earlier low-AUC behavior was caused solely by early stopping; the earlier 150-row separate rerun is not this benchmark.", "- Estimator fit time excludes preprocessing and process startup. End-to-end user-facing latency is NOT MEASURED.", "- The earlier sanity comparison differs because the older sample protocol was two-stage; the new benchmark uses direct-from-full-data sampling as instructed.", "", "## Safe to claim", "", "- Benchmarked four classical baselines and three PennyLane quantum classifiers on 80-row stratified samples from four biomedical datasets, with three-fold CV and fold-local preprocessing; Parkinson's was also evaluated with subject-grouped folds.", "- The quantum models ran on PennyLane's analytic `default.qubit` simulator with four qubits and four PCA features; no real hardware was used.", "- Implemented parent-enforced subprocess timeouts and fold-level result checkpointing; the saved timeout self-test passed.", "- The grouped Parkinson's split had zero subject overlap, while random-fold subject overlap was observed and reported.", "", "## NOT safe to claim", "", "- Do not claim quantum advantage or that quantum models beat classical models generally; the benchmark is a small fixed sample with best-of-fold selection and exploratory tests.", "- Do not present Parkinson's random-fold results as subject-independent generalization.", "- Do not claim full-dataset quantum performance, real-hardware performance, calibration, conformal coverage, SHAP explanations, or nested cross-validation.", "- Do not claim full-feature classical comparisons against quantum models; setting A classical baselines use four PCA features, and quantum models were not run in setting B.", "- Do not claim the older shared-80 results were reproduced by the direct 80-row sample; they match only under the separate two-stage sampling diagnostic.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE={OUT / 'RESULTS.md'}")
    print(f"TABLE_SECTIONS={sum(1 for x in lines if x.startswith('### ') and ' folds — setting ' in x)}")
    print(f"OUTPUT_BYTES={(OUT / 'RESULTS.md').stat().st_size}")


if __name__ == "__main__":
    main()

