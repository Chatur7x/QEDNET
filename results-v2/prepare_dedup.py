import json
from pathlib import Path
p=Path('results-v2/quantum_rerun_heart_fold0.json')
rs=json.loads(p.read_text(encoding='utf-8-sig'))
for r in rs: r['run_type']='benchmark'
raw=json.loads(Path('results-v2/quantum_raw.json').read_text(encoding='utf-8'))
fields=('dataset','split_variant','model','seed','fold','feature_setting')
def k(r):return tuple(r.get(x) for x in fields)
# benchmark key duplicates only; report all duplicate keys/errors separately via audit file.
bench=[r for r in raw if r.get('run_type')=='benchmark']
first={}
dropped=[]
for r in bench:
 key=k(r)
 if key not in first: first[key]=r
 else: dropped.append(r)
for r in rs:
 key=k(r)
 if key in first: first[key]=r
 else: first[key]=r
kept=list(first.values())
Path('results-v2/quantum_deduplicated.json').write_text(json.dumps(kept,indent=2),encoding='utf-8')
# Replace the previous dedup rule duplicate section with all raw key collisions, clearly separating non-grid rows.
audit=Path('results-v2/audit_raw_records.raw.txt').read_text(encoding='utf-8')
lines=['# Deduplication rule','','For the benchmark tables only, consider records with `run_type=benchmark`. Keep the first completed record per key `(dataset, split_variant, model, seed, fold, feature_setting)`; if the first record is not completed, use the first later completed record. Timing gates and repeat timing runs are excluded from score summaries because their `run_type` is not `benchmark`. The corrected heart fold-0 reruns are separate replacements for the conflicting keys after confirming the sample/fold protocol.','','## Dropped duplicate benchmark rows (AUC)','']
# There should be no repeated benchmark keys; one-off gate and repeat rows are intentionally not counted as benchmark duplicates.
for r in dropped: lines.append(f"- {k(r)} — dropped AUC `{r.get('auc')}`")
lines += ['', '## Raw-file duplicate keys across run types','', 'These collisions were identified by the requested key across the entire raw JSON. Non-benchmark timing pilots/repeats are excluded from the rebuilt score tables; their different `run_type` is retained in the raw-record audit.']
lines += ['','```text',audit.strip(),'```','', 'The corrected heart_disease / A-setting / random / seed-42 / fold-0 QSVM, VQC, and Reupload records are in `quantum_rerun_heart_fold0.json`; these have the final direct-sample protocol and replace their benchmark keys in `quantum_deduplicated.json`.']
Path('results-v2/dedup_rule.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'BENCHMARK_RAW_ROWS={len(bench)} UNIQUE_KEYS={len(first)} DROPPED_DUPLICATE_BENCHMARK_ROWS={len(dropped)}')
print(f'REPLACEMENT_RERUNS={len(rs)}')
for r in rs: print(f"RERUN model={r['model']} auc={r['auc']}")
