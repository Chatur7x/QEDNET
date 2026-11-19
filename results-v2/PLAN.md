# QEDNET fixed benchmark v2 — Step 1 plan

## Repository and environment read

- Work branch: `results-v2` (created for this task; no commit will be made).
- Package versions from the current `pip show` command: scikit-learn `1.9.1`, PennyLane `0.45.1`, NumPy `2.5.3`. Raw command output is in `step1_package_versions.raw.txt`.
- Datasets found: scikit-learn breast cancer, `data/raw/heart_cleveland.data`, `data/raw/parkinsons.data`, and `data/raw/pima_diabetes.csv`.
- Real classes found: LogisticRegression, RandomForest, RBF SVM, MLP, ParameterMatchedMLP, QSVM, VQC, ReuploadClassifier, and HybridQNN. The requested benchmark excludes HybridQNN.
- No model or pipeline source files were edited. Older `results/` artifacts remain read-only; their numbers were not used as benchmark results, except the required sanity comparison.

## Parkinson's sampling constraint

The current data audit found `32` unique subject IDs, while the requested Parkinson's sample has `80` rows. A sample with at most one row per subject is therefore impossible for that sample size. The grouped Parkinson's sample will include repeated recordings per subject, record its actual rows-per-subject counts, and use subject-grouped folds with zero train/test subject overlap.

## Planned files under `results-v2/`

Step-one evidence already created:

- `step1_branch.command.txt`, `step1_branch.raw.txt`, `step1_branch.exit.txt`
- `step1_package_versions.command.txt`, `step1_package_versions.raw.txt`, `step1_package_versions.exit.txt`
- `step1_repo_inventory.py`, `step1_repo_inventory.command.txt`, `step1_repo_inventory.raw.txt`, `step1_repo_inventory.exit.txt`
- `step1_parkinsons_subject_audit.py`, `step1_parkinsons_subject_audit.command.txt`, `step1_parkinsons_subject_audit.raw.txt`, `step1_parkinsons_subject_audit.exit.txt`
- `PLAN.md`

Planned after approval:

- `run_fold.py` — dispatches to the repository's real estimator classes and emits one fold result.
- `runner.py` — parent-controlled subprocess launcher, process-tree timeout termination, immediate JSON checkpointing, and stdout/stderr capture.
- `timeout_sleep_child.py` and timeout self-test command/output/status files.
- `make_splits.py`, `splits.json`, and split-audit command/raw files.
- `classical_raw.json`, `quantum_raw.json`, per-fold logs, and command/raw/exit files for all runs.
- `summary.csv`, `paired_differences.csv`, their three-row/header audits, and summary command/raw files.
- `RESULTS.md` with measured outcomes, timeout and NOT MEASURED reasons, methods, limitations, and safe/unsafe resume claims.

No data files or large outputs will be committed.

## Run plan and gates

- First, validate that the parent kills a fake sleeping child at the configured test timeout. If this fails, fix only `results-v2/runner.py` and repeat the check; do not start model grids until it passes.
- The direct 80-row sanity fold did not match the older `results/shared80_classical_raw.json` record because that older sample was drawn from a 150-row sample. The user directed continuation after being told about this discrepancy. A two-stage diagnostic reproduced the old row, but all final benchmark runs followed the task's explicit direct-from-full-data 80-row sample rule. Both sanity outputs are retained.
- Generate one shared direct 80-row split manifest for all datasets, plus subject-grouped Parkinson's folds. Save local and source row indices, subject IDs, class labels, and per-subject sample counts.
- Run both classical feature settings and seeds, keeping all preprocessing fit within each training fold. The MLP variant used `early_stopping=False`; ParameterMatchedMLP used the real repository class and its fitted metadata records actual parameter counts.
- Run the heart-disease timing gate on one fold per quantum model before the quantum grid. All three measured fit times passed the requested threshold; the exact per-model logs are saved under `timing_gate_direct_heart_*.raw.txt`.
- Run quantum fits only in child processes; checkpoint every completed or timed-out fold. Apply grouped Parkinson's splits to every requested classical and quantum model.
- Build summary and paired-difference tables only from measured fold rows. Omit timeouts from averages, include paired-test p-values, and label results with insufficient completed pairs NOT MEASURED.
- Run the five-repeat heart QSVM timing measurement and report each observed fit time plus measured min/median/max.
- Print three rows for each created CSV and check every field against its header and source records.

## Execution status

The user approved continuing after the sanity discrepancy was explained. The classical and quantum grids, grouped Parkinson's runs, and five-repeat QSVM timing run have been executed. Their current-run outputs, commands, raw fold logs, and summaries are retained under `results-v2/`. No runtime estimates were used.
