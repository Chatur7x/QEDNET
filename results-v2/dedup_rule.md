# Deduplication rule

For score summaries, include only the final grid rows tagged `run_type=benchmark`. Retain the first completed record for each requested key `(dataset, split_variant, model, seed, fold, feature_setting)`. If no completed record exists, retain the first non-completed row. Earlier timing pilots and repeated timing runs are not grid records; they remain in the raw audit but are excluded from scores. The corrected fold-0 reruns replace the matching benchmark records after verification on the same sample/fold.

## Duplicate-key records not used in the final score set

- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `0`; status `error`; AUC `None`; run_type `timing_gate_two_stage`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `3`; status `completed`; AUC `0.5166666666666666`; run_type `timing_gate_direct`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `141`; status `completed`; AUC `0.5166666666666666`; run_type `heart_qsvm_timing_repeat_1`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `142`; status `completed`; AUC `0.5166666666666666`; run_type `heart_qsvm_timing_repeat_2`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `143`; status `completed`; AUC `0.5166666666666666`; run_type `heart_qsvm_timing_repeat_3`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `144`; status `completed`; AUC `0.5166666666666666`; run_type `heart_qsvm_timing_repeat_4`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> raw index `145`; status `completed`; AUC `0.5166666666666666`; run_type `heart_qsvm_timing_repeat_5`; excluded from score set
- key `('heart_disease', 'random', 'vqc', 42, 0, 'A')` -> raw index `1`; status `completed`; AUC `0.6166666666666667`; run_type `timing_gate_two_stage`; excluded from score set
- key `('heart_disease', 'random', 'vqc', 42, 0, 'A')` -> raw index `4`; status `completed`; AUC `0.7111111111111111`; run_type `timing_gate_direct`; excluded from score set
- key `('heart_disease', 'random', 'reupload', 42, 0, 'A')` -> raw index `2`; status `completed`; AUC `0.48333333333333334`; run_type `timing_gate_two_stage`; excluded from score set
- key `('heart_disease', 'random', 'reupload', 42, 0, 'A')` -> raw index `5`; status `completed`; AUC `0.4166666666666667`; run_type `timing_gate_direct`; excluded from score set
- key `('heart_disease', 'random', 'qsvm', 42, 0, 'A')` -> original benchmark record replaced; old AUC `0.5166666666666666`; corrected rerun AUC `0.5166666666666666`
- key `('heart_disease', 'random', 'vqc', 42, 0, 'A')` -> original benchmark record replaced; old AUC `0.7111111111111111`; corrected rerun AUC `0.7111111111111111`
- key `('heart_disease', 'random', 'reupload', 42, 0, 'A')` -> original benchmark record replaced; old AUC `0.4166666666666667`; corrected rerun AUC `0.4166666666666667`

## Corrected AUC check

- qsvm: prior benchmark AUC `0.5166666666666666`; rerun AUC `0.5166666666666666`; changed `False`.
- vqc: prior benchmark AUC `0.7111111111111111`; rerun AUC `0.7111111111111111`; changed `False`.
- reupload: prior benchmark AUC `0.4166666666666667`; rerun AUC `0.4166666666666667`; changed `False`.
