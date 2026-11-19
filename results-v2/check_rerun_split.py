import json
from pathlib import Path
p=Path('results-v2')
m=json.loads((p/'splits.json').read_text(encoding='utf-8'))['datasets']['heart_disease']
r=json.loads((p/'quantum_rerun_heart_fold0.json').read_text(encoding='utf-8-sig'))[0]
f=m['random_folds'][0]
print(f"manifest_sample_n={m['n']} local_sample_positions_count={len(m['sample_local_row_indices'])} max_local_position={max(m['sample_local_row_indices'])}")
print(f"manifest_source_sample_rows_count={len(m['sample_source_row_indices'])} source_301_present={301 in m['sample_source_row_indices']}")
print(f"fold=0 train_samples={len(f['train_sample_indices'])} test_samples={len(f['test_sample_indices'])}")
print(f"fold0_train_source_indices_count={len(f['train_source_row_indices'])} fold0_test_source_indices_count={len(f['test_source_row_indices'])}")
print(f"rerun_key=({r['dataset']},{r['split_variant']},{r['model']},seed={r['seed']},fold={r['fold']},setting={r['feature_setting']}) n={r['n_samples']} train={r['n_train']} test={r['n_test']}")
missing=[k for k in ('sample_source_row_indices','train_source_row_indices','test_source_row_indices','train_sample_indices','test_sample_indices') if k not in r]
print(f"rerun_missing_index_fields={missing}")
print('exact_source_row_and_fold_index_match=NOT ESTABLISHED from rerun record (index arrays are absent)')
