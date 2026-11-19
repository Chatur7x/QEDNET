import json
from collections import defaultdict
from pathlib import Path
raw=json.loads(Path('results-v2/quantum_raw.json').read_text(encoding='utf-8'))
fields=('dataset','split_variant','model','seed','fold','feature_setting')
def key(r):return tuple(r.get(k) for k in fields)
g=defaultdict(list)
for i,r in enumerate(raw):g[key(r)].append((i,r))
lines=['# Deduplication rule','','For score summaries, include only the final grid rows tagged `run_type=benchmark`. Retain the first completed record for each requested key `(dataset, split_variant, model, seed, fold, feature_setting)`. If no completed record exists, retain the first non-completed row. Earlier timing pilots and repeated timing runs are not grid records; they remain in the raw audit but are excluded from scores. The corrected fold-0 reruns replace the matching benchmark records after verification on the same sample/fold.','','## Duplicate-key records not used in the final score set','']
for k,entries in g.items():
 if len(entries)<2:continue
 chosen=next(((i,r) for i,r in entries if r.get('run_type')=='benchmark' and r.get('status')=='completed'),None)
 if chosen is None: chosen=next(((i,r) for i,r in entries if r.get('status')=='completed'),entries[0])
 for i,r in entries:
  if i==chosen[0]:continue
  lines.append(f"- key `{k}` -> raw index `{i}`; status `{r.get('status')}`; AUC `{r.get('auc')}`; run_type `{r.get('run_type')}`; excluded from score set")
# benchmark original row replaced by corrected rerun
for r in json.loads(Path('results-v2/quantum_rerun_heart_fold0.json').read_text(encoding='utf-8-sig')):
 orig=next(x for x in raw if x.get('run_type')=='benchmark' and key(x)==key(r))
 lines.append(f"- key `{key(r)}` -> original benchmark record replaced; old AUC `{orig.get('auc')}`; corrected rerun AUC `{r.get('auc')}`")
lines += ['', '## Corrected AUC check','']
for r in json.loads(Path('results-v2/quantum_rerun_heart_fold0.json').read_text(encoding='utf-8-sig')):
 lines.append(f"- {r['model']}: prior benchmark AUC `{next(x['auc'] for x in raw if x.get('run_type')=='benchmark' and key(x)==key(r))}`; rerun AUC `{r['auc']}`; changed `{next(x['auc'] for x in raw if x.get('run_type')=='benchmark' and key(x)==key(r)) != r['auc']}`.")
Path('results-v2/dedup_rule.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'RAW_RECORDS={len(raw)} DUPLICATE_KEYS={sum(len(v)>1 for v in g.values())} DROPPED_DUPLICATE_ROWS={sum(len(v)-1 for v in g.values() if len(v)>1)}')
for line in lines:
 if line.startswith('- key ') or line.startswith('- qsvm') or line.startswith('- vqc') or line.startswith('- reupload'): print(line)
