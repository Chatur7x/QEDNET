from pathlib import Path

path = Path(__file__).resolve().parent / "RESULTS.md"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "The paired tests use the nine observed seed/fold pairs. P-values are unadjusted for multiple comparisons.",
    "The paired tests use nine correlated pairs: the same three folds are reused across three estimator seeds. Therefore, paired t-test and Wilcoxon p-values are exploratory and are not independent-sample evidence; p-values are also unadjusted for multiple comparisons.",
)
text = text.replace(
    "Classical statuses: {'completed': 468, 'timeout': 0, 'error': 0, 'not_measured': 72}; quantum statuses: {'completed': 135, 'timeout': 0, 'error': 0, 'not_measured': 0}. Earlier two-stage timing pilots are excluded.",
    "Classical benchmark statuses after deduplication: 468 completed, 0 timeout, 0 error, 72 not measured. Quantum benchmark statuses after deduplication and corrected fold-0 replacements: 135 completed, 0 timeout, 0 error, 0 not measured. Earlier timing pilots and repeats are excluded from score counts.",
)
text = text.replace(
    "Repeated same-fold heart QSVM estimator fit times (seed 42, n=80, train/test 53/27):",
    "Repeated same-fold heart QSVM estimator fit times read from `timing_repeats_heart_qsvm.json` only (seed 42, n=80, train/test 53/27):",
)
text += """

## Audit corrections (raw-record review)

The raw quantum audit found one error record: `heart_disease / random / qsvm / seed 42 / fold 0 / setting A`, with no AUC. The exception was `IndexError: index 301 is out of bounds for axis 0 with size 297`. `heart_cleveland.data` contains 303 source rows; dropping rows with missing values leaves 297 model rows, while the largest retained original source index is 301. Source row IDs are labels for auditing and can have gaps after cleaning; indexing the 297-row cleaned array by those IDs raises the error. The fix in `run_fold.py` uses the manifest's `sample_local_row_indices` for array indexing and validates the positions. The sample and split were not changed.

`quantum_raw.json` contains 146 records and three duplicate keys (11 additional raw rows across those keys). The collisions are heart QSVM, VQC, and Reupload at seed 42, fold 0, setting A; they include non-benchmark timing pilots and QSVM repeat measurements. The audit and every excluded collision with its AUC are in `audit_raw_records.raw.txt` and `dedup_rule.md`. Score CSVs are rebuilt from benchmark-grid rows only, keeping the first completed record for each requested key. Corrected reruns for all three conflicting model keys are in `quantum_rerun_heart_fold0.json` and replace the corresponding benchmark rows.

The corrected fold AUCs did not change versus the retained benchmark values: QSVM `0.5166666666666666`, VQC `0.7111111111111111`, and Reupload `0.4166666666666667`. Earlier two-stage pilot AUCs differ for VQC (`0.6166666666666667`) and Reupload (`0.48333333333333334`); those came from a different sample protocol and are excluded from benchmark summaries.

Status counts above describe the deduplicated benchmark set, not every raw record or the NOT MEASURED placeholder rows in the CSV. The raw error remains documented and is not counted as a benchmark-grid error after the corrected fold rerun. `summary.csv` and `paired_differences.csv` were rebuilt; their required columns and first three data rows were printed and checked in `deduplicate_rebuild.raw.txt`.
"""
path.write_text(text, encoding="utf-8")
print("UPDATED=results-v2/RESULTS.md")
print(f"PAIRED_EXPLORATORY_LABEL={'nine correlated pairs' in text}")
print(f"DEDUP_SECTION={'## Audit corrections (raw-record review)' in text}")
