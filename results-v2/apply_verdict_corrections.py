from pathlib import Path
p=Path('results-v2/RESULTS_FINAL.md')
s=p.read_text(encoding='utf-8')
old='QSVM fit 6.996467499999881 s, predict 7.0287773999998535 s; 2862 kernel evaluations combined across fit and predict, not split.'
new='QSVM fit 6.996467499999881 s, predict 7.0287773999998535 s; the implementation-level counter records 1431 kernel evaluations during fit and 1431 during prediction (2862 combined). Fit count is 53 × 54 / 2 from the symmetric train kernel; prediction count is 27 × 53 test/train pairs. The counter resets in fit and continues accumulating through prediction.'
assert old in s
s=s.replace(old,new)
marker='These nine pairs per row reuse the same three folds across three seeds, so they are not independent. The p-values are exploratory, unadjusted for multiple comparisons, and should not be read as confirmation that classical models outperform quantum models in general.'
assert marker in s
method='\n\nWith SciPy 1.18.1, the approximate Wilcoxon method was used for the breast-cancer/random, diabetes/random, Parkinson random, and Parkinson grouped best-of-fold comparisons because their absolute paired differences contain ties. It was also approximate for breast-cancer/random ParameterMatchedMLP-Reupload versus QSVM and ParameterMatchedMLP-VQC versus QSVM; heart-disease/random Reupload-matched ParameterMatchedMLP versus Reupload (zero paired differences); Parkinson grouped Reupload-matched versus QSVM; and Parkinson random Reupload-matched versus QSVM and VQC, plus VQC-matched versus QSVM. The other 24 comparisons used the exact method. These method choices explain the Wilcoxon p-value changes listed below.'
s=s.replace(marker,marker+method)
# Add explicit sample-specific Parkinson resume correction under Safe to claim.
safe='- The grouped Parkinson\'s split had zero subject overlap, while random-fold subject overlap was observed and reported.'
assert safe in s
resume='\n- For this 80-row Parkinson\'s benchmark, setting A RandomForest mean AUC was 0.8634920634920635 (sample SD 0.09222733607382565; n=9) on random folds and 0.7067460317460317 (sample SD 0.2754061883729273; n=9) on subject-grouped folds. The random folds leaked subjects; the grouped folds had zero subject overlap. Do not describe the earlier 0.92-to-0.69 result as this 80-row benchmark; that was a separate full-data result and is not re-established by these files.'
s=s.replace(safe,safe+resume)
s += '\n\nAdditional corrections: the sample-specific Parkinson\'s RandomForest AUCs and QSVM timing-gate kernel counts above were calculated from `summary.csv` and `qednet/models/quantum/qsvm.py`; command output is in `verify_followup_claims.raw.txt`. The Wilcoxon method rows and tie/zero reasons are in `recompute_paired_tests_final.raw.txt`. No model code or benchmark records were changed for these report edits.\n'
p.write_text(s,encoding='utf-8')
print('UPDATED=RESULTS_FINAL.md')
print('PARKINSON_AUC_CORRECTION_PRESENT='+str('0.8634920634920635' in s and '0.7067460317460317' in s))
print('WILCOXON_METHOD_EXPLANATION_PRESENT='+str('absolute paired differences contain ties' in s and 'zero paired differences' in s))
print('QSVM_SPLIT_COUNTS_PRESENT='+str('1431 kernel evaluations during fit and 1431 during prediction' in s))
