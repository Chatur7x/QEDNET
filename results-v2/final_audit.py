import csv,json,collections,statistics
from pathlib import Path
out=Path('results-v2')
q=json.loads((out/'quantum_deduplicated.json').read_text(encoding='utf-8'))
c=json.loads((out/'classical_raw.json').read_text(encoding='utf-8'))
cs=collections.Counter(r.get('status') for r in c if r.get('run_type')=='benchmark')
qs=collections.Counter(r.get('status') for r in q)
print('CLASSICAL_DEDUP_STATUS_COUNTS='+json.dumps(dict(cs),sort_keys=True))
print('QUANTUM_DEDUP_STATUS_COUNTS='+json.dumps(dict(qs),sort_keys=True))
print('COMBINED_COMPLETED='+str(cs['completed']+qs['completed']))
for name,expected in [('summary.csv',['dataset','feature_setting','model','seed','fold','status','auc','accuracy','balanced_accuracy','f1','train_seconds','predict_seconds','timeout_elapsed_seconds','n_train','n_test','trainable_parameters','notes']),('paired_differences.csv',['dataset','split_variant','seed','fold','best_quantum_model','best_quantum_auc','best_classical_model','best_classical_auc','difference'])]:
 with (out/name).open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f); rows=list(reader); cols=reader.fieldnames
  print(f'CSV={name} ROWS={len(rows)} COLUMNS_MATCH={cols==expected} ALL_ROWS_SHAPE={all(None not in r and len(r)==len(expected) for r in rows)}')
  print('FIRST_3_ROWS='+json.dumps(rows[:3],separators=(',',':')))
timing=json.loads((out/'timing_repeats_heart_qsvm.json').read_text(encoding='utf-8'))
fit=timing['estimator_fit_seconds']
print('TIMING_FILE_STATS='+json.dumps({'fit_seconds':fit,'min':min(fit),'median':statistics.median(fit),'max':max(fit)},separators=(',',':')))
for model in ('qsvm','vqc','reupload'):
 r=next(r for r in q if r['dataset']=='heart_disease' and r['model']==model and r['seed']==42 and r['fold']==0 and r['feature_setting']=='A')
 print(f"FINAL_AUC model={model} auc={r['auc']} train_seconds={r['train_seconds']}")
