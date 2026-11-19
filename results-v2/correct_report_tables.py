import csv,re,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
report=(p/'RESULTS.md').read_text(encoding='utf-8')
with (p/'summary.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
def split(r):
 m=re.search(r'(?:^|;\s*)split_variant=([^;]+)',r['notes']);return m.group(1) if m else 'random'
groups={}
for r in rows:
 if r['status']=='completed':groups.setdefault((r['dataset'],split(r),r['feature_setting'],r['model']),[]).append(r)
changes=[]
lines=report.splitlines()
section=''
for i,line in enumerate(lines):
 if line.startswith('### '):section=line[4:]
 if not line.startswith('|') or line.startswith('| Model') or line.startswith('| ---'):continue
 cells=[x.strip() for x in line.strip('|').split('|')]
 if len(cells)<8:continue
 model=cells[0]
 if model not in ('qsvm','vqc','reupload'):continue
 dataset='heart_disease' if 'heart disease' in section.lower() else None
 if not dataset or 'setting A' not in section:continue
 key=(dataset,'random','A',model); vals=groups.get(key,[])
 if not vals:continue
 old=cells[6]
 nums=re.match(r'([0-9.eE+-]+) ± ([0-9.eE+-]+) \(n=(\d+)\)',old)
 if not nums:continue
 times=[float(r['train_seconds']) for r in vals]; mean=statistics.mean(times);sd=statistics.stdev(times)
 new=f'{mean} ± {sd} (n={len(times)})'
 if old!=new:
  changes.append((section,model,'Fit seconds',old,new,'summary.csv'))
  cells[6]=new
  lines[i]='| '+' | '.join(cells)+' |'
correction=['','## Corrections from v2','','Recomputed from completed, deduplicated `summary.csv` rows. No other table cells changed.','']
for sec,model,col,old,new,source in changes:correction.append(f'- {sec}; {model}; {col}: `{old}` -> `{new}`; source: `{source}`.')
lines.extend(correction)
(p/'RESULTS_FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('CREATED=RESULTS_FINAL.md')
print('CHANGED_TABLE_CELLS='+str(len(changes)))
for x in changes:print('|'.join(x))
