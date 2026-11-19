import csv,re,statistics,json
from pathlib import Path
p=Path('results-v2')
with (p/'summary.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
def variant(r):
 m=re.search(r'(?:^|;\s*)split_variant=([^;]+)',r['notes']);return m.group(1) if m else 'random'
for v in ('random','grouped'):
 a=[float(r['auc']) for r in rows if r['dataset']=='parkinsons' and variant(r)==v and r['feature_setting']=='A' and r['model']=='random_forest' and r['status']=='completed']
 print(f'PARKINSON_RF_AUC dataset=parkinsons split={v} setting=A model=random_forest n={len(a)} mean={statistics.mean(a)} sample_sd={statistics.stdev(a)}')
