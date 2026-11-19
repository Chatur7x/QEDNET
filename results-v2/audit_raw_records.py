import json
import collections
from pathlib import Path
p=Path('results-v2/quantum_raw.json')
r=json.loads(p.read_text(encoding='utf-8'))
fields=('dataset','split_variant','model','seed','fold','feature_setting')
def key(x): return tuple(x.get(k) for k in fields)
g=collections.defaultdict(list)
for i,x in enumerate(r): g[key(x)].append((i,x))
errors=[(i,x) for i,x in enumerate(r) if x.get('status')=='error']
dups={k:v for k,v in g.items() if len(v)>1}
lines=[f'RECORD_COUNT={len(r)}',f'ERROR_COUNT={len(errors)}']
for i,x in errors: lines.append(f'ERROR index={i} key={key(x)} status={x.get("status")} auc={x.get("auc")} message={x.get("error", "")}')
lines.append(f'DUPLICATE_KEY_COUNT={len(dups)}')
for k,v in dups.items():
 lines.append(f'DUPLICATE key={k} count={len(v)}')
 for i,x in v: lines.append(f'  index={i} status={x.get("status")} auc={x.get("auc")} run_type={x.get("run_type")} sample_protocol={x.get("sample_protocol")}')
Path('results-v2/audit_raw_records.raw.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
