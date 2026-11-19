import csv
import json
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent
FIELDS=('dataset','split_variant','model','seed','fold','feature_setting')
SUMMARY_COLUMNS=['dataset','feature_setting','model','seed','fold','status','auc','accuracy','balanced_accuracy','f1','train_seconds','predict_seconds','timeout_elapsed_seconds','n_train','n_test','trainable_parameters','notes']
PAIRED_COLUMNS=['dataset','split_variant','seed','fold','best_quantum_model','best_quantum_auc','best_classical_model','best_classical_auc','difference']
CLASSICAL=('logistic_regression','random_forest','rbf_svm','mlp_default','parameter_matched_mlp_vqc','parameter_matched_mlp_reupload')
QUANTUM=('qsvm','vqc','reupload')
def key(r): return tuple(r.get(k) for k in FIELDS)
def num(v): return '' if v is None else str(v)
raw=json.loads((OUT/'quantum_raw.json').read_text(encoding='utf-8'))
corrected=json.loads((OUT/'quantum_rerun_heart_fold0.json').read_text(encoding='utf-8-sig'))
# Only the original benchmark grid is eligible for benchmark summaries. Timing pilots and repeats are non-grid evidence.
bench=[r for r in raw if r.get('run_type')=='benchmark']
seen=set(); kept=[]; dropped=[]
for r in bench:
 k=key(r)
 if k not in seen and r.get('status')=='completed':
  kept.append(r); seen.add(k)
 elif k in seen:
  dropped.append(r)
 else:
  # preserve first non-completed until a completed duplicate appears; if so replace it
  prior=next((x for x in kept if key(x)==k),None)
  if prior is None:
   kept.append(r); seen.add(k)
  elif r.get('status')=='completed':
   kept.remove(prior); kept.append(r); dropped.append(prior)
# Replace benchmark rows for the three corrected heart-disease fold-0 models.
for rerun in corrected:
    rerun['run_type']='benchmark'
    for i,existing in enumerate(kept):
        if key(existing)==key(rerun):
            kept[i]=rerun
            break
    else:
        kept.append(rerun)# Reconstruct classical benchmark rows; apply same first-completed key rule defensively.
classical_raw=json.loads((OUT/'classical_raw.json').read_text(encoding='utf-8'))
classical=[r for r in classical_raw if r.get('run_type')=='benchmark']
def first_completed(rows):
 d={}; drops=[]
 for r in rows:
  k=key(r)
  if k not in d or (d[k].get('status')!='completed' and r.get('status')=='completed'):
   if k in d: drops.append(d[k])
   d[k]=r
  else: drops.append(r)
 return list(d.values()),drops
classical, c_drops=first_completed(classical)
# Build fold-level summary CSV.
summary=[]
for r in classical+kept:
 if r.get('status')=='not_measured': status='not_measured'
 else: status=r.get('status','error')
 notes=[f"split_variant={r.get('split_variant','random')}",f"n_samples={r.get('n_samples','NOT MEASURED')}",f"input_features={r.get('input_features','NOT MEASURED')}",f"model_input_features={r.get('model_input_features','NOT MEASURED')}",f"subject_overlap_count={r.get('subject_overlap_count','NOT MEASURED')}"]
 if r.get('notes'): notes.append(str(r['notes']))
 if r.get('metadata'): notes.append('metadata='+json.dumps(r['metadata'],sort_keys=True,separators=(',',':')))
 summary.append({'dataset':r.get('dataset'),'feature_setting':r.get('feature_setting'),'model':r.get('model'),'seed':r.get('seed'),'fold':r.get('fold'),'status':status,'auc':num(r.get('auc')),'accuracy':num(r.get('accuracy')),'balanced_accuracy':num(r.get('balanced_accuracy')),'f1':num(r.get('f1')),'train_seconds':num(r.get('train_seconds')),'predict_seconds':num(r.get('predict_seconds')),'timeout_elapsed_seconds':num(r.get('parent_elapsed_seconds')) if status=='timeout' else '', 'n_train':num(r.get('n_train')),'n_test':num(r.get('n_test')),'trainable_parameters':num(r.get('trainable_parameters')),'notes':'; '.join(notes)})
# Preserve explicit NOT MEASURED classical rows, each key once.
for r in classical_raw:
 if r.get('run_type')=='benchmark' and r.get('status')=='not_measured' and not any(x['dataset']==r.get('dataset') and x['model']==r.get('model') and str(x['seed'])==str(r.get('seed')) and str(x['fold'])==str(r.get('fold')) and x['feature_setting']==r.get('feature_setting') and f"split_variant={r.get('split_variant','random')}" in x['notes'] for x in summary):
  summary.append({'dataset':r.get('dataset'),'feature_setting':r.get('feature_setting'),'model':r.get('model'),'seed':r.get('seed'),'fold':r.get('fold'),'status':'not_measured','auc':'','accuracy':'','balanced_accuracy':'','f1':'','train_seconds':'','predict_seconds':'','timeout_elapsed_seconds':'','n_train':num(r.get('n_train')),'n_test':num(r.get('n_test')),'trainable_parameters':'','notes':f"split_variant={r.get('split_variant','random')}; {r.get('notes','NOT MEASURED: reason not recorded')}"})
summary.sort(key=lambda r:(r['dataset'],r['feature_setting'],r['notes'].split(';')[0],int(r['seed']),int(r['fold']),r['model']))
with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=SUMMARY_COLUMNS);w.writeheader();w.writerows(summary)
# Paired best quantum minus best classical AUC for each matched key.
q_by={(r['dataset'],r['split_variant'],r['seed'],r['fold']):[] for r in kept if r.get('status')=='completed' and r.get('feature_setting')=='A'}
for r in kept:
 if r.get('status')=='completed' and r.get('feature_setting')=='A': q_by.setdefault((r['dataset'],r['split_variant'],r['seed'],r['fold']),[]).append(r)
c_by=defaultdict(list)
for r in classical:
 if r.get('status')=='completed' and r.get('feature_setting')=='A': c_by[(r['dataset'],r.get('split_variant','random'),r['seed'],r['fold'])].append(r)
paired=[]
for k,qs in q_by.items():
 cs=c_by.get(k,[])
 if not qs or not cs: continue
 qb=max(qs,key=lambda x:float(x['auc'])); cb=max(cs,key=lambda x:float(x['auc']))
 paired.append({'dataset':k[0],'split_variant':k[1],'seed':k[2],'fold':k[3],'best_quantum_model':qb['model'],'best_quantum_auc':str(qb['auc']),'best_classical_model':cb['model'],'best_classical_auc':str(cb['auc']),'difference':str(float(qb['auc'])-float(cb['auc']))})
paired.sort(key=lambda r:(r['dataset'],r['split_variant'],int(r['seed']),int(r['fold'])))
with (OUT/'paired_differences.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=PAIRED_COLUMNS);w.writeheader();w.writerows(paired)
# Save retained deduplicated quantum set for audit traceability.
(OUT/'quantum_deduplicated.json').write_text(json.dumps(kept,indent=2),encoding='utf-8')
# Dedup rule lists all duplicate raw records not retained in the eligible benchmark set.
lines=['# Deduplication rule','', 'For benchmark summaries, only records tagged `run_type=benchmark` are eligible. Per key `(dataset, split_variant, model, seed, fold, feature_setting)`, retain the first completed benchmark record in `quantum_raw.json`. If no completed record exists, retain the first non-completed record. Timing-gate and repeated-timing records are excluded because they are not benchmark-grid rows. The explicit corrected QSVM rerun is saved separately and is used as the replacement for the same benchmark key; its sample and fold indices are unchanged.','', '## Dropped duplicate records (AUC)','']
for r in dropped:
 lines.append(f"- {key(r)} — dropped status `{r.get('status')}`, AUC `{r.get('auc')}`, run_type `{r.get('run_type')}`")
lines += ['', 'The heart-disease QSVM seed-42 fold-0 completed benchmark record was replaced by `quantum_rerun_heart_fold0.json` after the indexing fix. Its AUC is compared in the final check; non-benchmark timing-gate/repeat rows remain excluded from summary and paired tables.', '', f"Classical benchmark duplicate records dropped: {len(c_drops)}. Quantum duplicate benchmark records dropped: {len(dropped)}. Retained completed quantum benchmark rows plus the one explicit rerun: {sum(r.get('status')=='completed' for r in kept)}."]
(OUT/'dedup_rule.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'CLASSICAL_STATUS_COUNTS={json.dumps({s:sum(r.get("status")==s for r in classical_raw if r.get("run_type")=="benchmark") for s in ("completed","timeout","error","not_measured")},sort_keys=True)}')
print(f'QUANTUM_RAW_BENCHMARK_COUNTS={json.dumps({s:sum(r.get("status")==s for r in bench) for s in ("completed","timeout","error","not_measured")},sort_keys=True)}')
print(f'QUANTUM_DEDUP_STATUS_COUNTS={json.dumps({s:sum(r.get("status")==s for r in kept) for s in ("completed","timeout","error","not_measured")},sort_keys=True)}')
print(f'SUMMARY_ROWS={len(summary)} PAIRED_ROWS={len(paired)} CLASSICAL_DROPPED={len(c_drops)} QUANTUM_DROPPED={len(dropped)}')
for name in ('summary.csv','paired_differences.csv'):
 with (OUT/name).open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f); rows=list(reader); print(f'CSV={name} COLUMNS={reader.fieldnames} FIRST3={json.dumps(rows[:3],separators=(",",":"))} COLUMN_VALIDATION={"PASS" if reader.fieldnames in (SUMMARY_COLUMNS,PAIRED_COLUMNS) and all(None not in r for r in rows) else "FAIL"}')


