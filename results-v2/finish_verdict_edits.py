from pathlib import Path
p=Path('results-v2/RESULTS_FINAL.md')
s=p.read_text(encoding='utf-8')
old='The saved `run_fold.py` is untracked: `run_fold_git_history.raw.txt` shows no tracked log or diff and `git ls-files --error-unmatch` fails. The current source is consistent with a code-side index correction; the exact historical before/after code change is NOT ESTABLISHED from git history. The data file and `splits.json` are present in the saved evidence, and the row audit explains the failure, but no tracked snapshot exists to conclusively attribute what changed between failure and rerun.'
new='`run_fold.py` is now committed on `results-v2` as `3a17fad` (`Record results-v2 fold runner`); the commit records the current file. Before that commit the file was untracked, so the prior source snapshot and exact historical before/after code change remain NOT ESTABLISHED. The current source uses local sample positions and bounds-checks them. The data file and `splits.json` audit explain why source row 301 was invalid as a position, but the saved git history cannot conclusively establish when that code change was made.'
assert old in s
s=s.replace(old,new)
s=s.replace("Do not describe the earlier 0.92-to0.69 result as this 80-row benchmark; that was a separate full-data result and is not re-established by these files.", "Do not describe the earlier full-data Parkinson's result as this 80-row benchmark; its exact source values are not re-established by these files.")
old2='The same three folds were reused across three estimator seeds, so these nine pairs are correlated. The paired p-values are exploratory and unadjusted.'
new2='The same three folds were reused across three estimator seeds, so these nine pairs are correlated. The paired p-values are exploratory and unadjusted. The reason for the 11 Wilcoxon p-value changes is method selection: SciPy 1.18.1 used the approximate method when paired absolute differences had ties, or when one comparison had zero differences; it used the exact method for the other comparisons. The affected comparisons and their tie/zero reasons are listed in `recompute_paired_tests_final.raw.txt`.'
assert old2 in s
s=s.replace(old2,new2)
s=s.replace('Additional corrections: the sample-specific Parkinson\'s RandomForest AUCs and QSVM timing-gate kernel counts above were calculated from `summary.csv` and `qednet/models/quantum/qsvm.py`; command output is in `verify_followup_claims.raw.txt`. The Wilcoxon method rows and tie/zero reasons are in `recompute_paired_tests_final.raw.txt`. No model code or benchmark records were changed for these report edits.', 'Additional corrections: Parkinson\'s setting A RandomForest AUC is `0.8634920634920635` on random folds and `0.7067460317460317` on grouped folds (source: `summary.csv`; full means and SDs above). The heart timing-gate kernel counts are 1431 fit and 1431 prediction evaluations (source: `qednet/models/quantum/qsvm.py` and the measured 53/27 fold; audit output in `verify_followup_claims.raw.txt`). SciPy used the approximate Wilcoxon method for 11 comparisons with tied absolute differences or zero differences; 24 comparisons used exact. No model fits or benchmark runs were performed for these report edits.')
p.write_text(s,encoding='utf-8')
print('UPDATED=RESULTS_FINAL.md')
print('COMMIT_RECORDED='+str('3a17fad' in s))
print('UNSUPPORTED_OLD_AUC_PHRASE_PRESENT='+str('0.92-to0.69' in s))
print('AFFECTED_WILCOXON_REASON_PRESENT='+str('11 Wilcoxon p-value changes' in s))
