import csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
with (P/'paired_tests_recomputed.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
lookup={(r['comparison'],r['dataset'],r['split_variant']):r for r in rows}
text=(P/'RESULTS_FINAL.md').read_text(encoding='utf-8')
found={};section=''
for line in text.splitlines():
 if line.startswith('#### '):section=line[5:].strip().lower()
 if not line.startswith('|'):continue
 c=[x.strip() for x in line.strip('|').split('|')]
 if len(c)==7 and c[0] in ('breast_cancer / random','heart_disease / random','parkinsons / random','parkinsons / grouped','diabetes / random'):
  d,v=c[0].split(' / ');found[('best_of_fold',d,v)]=(float(c[4]),float(c[5]))
 elif len(c)==7 and c[0].startswith('parameter_matched_mlp_') and c[1] in ('qsvm','vqc','reupload'):
  d='parkinsons' if section.startswith('parkinson') else ('breast_cancer' if section.startswith('breast cancer') else ('heart_disease' if section.startswith('heart disease') else 'diabetes'))
  v='grouped' if 'grouped' in section else 'random';found[(c[0]+' - '+c[1],d,v)]=(float(c[5]),float(c[6]))
mismatch=[]
for k,r in lookup.items():
 if k not in found:continue
 for i,col in enumerate(('t_test_p','wilcoxon_p')):
  old=found[k][i];new=float(r[col])
  if abs(old-new)>1e-9*max(abs(new),1e-300):mismatch.append((k,col,old,new))
print(f'PAIRED_TEST_ROWS={len(rows)} PAIRED_REPORT_COMPARISONS={len(found)}')
print('PVALUE_MISMATCH_COUNT='+str(len(mismatch)))
for x in mismatch:print(x)
