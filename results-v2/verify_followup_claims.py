import csv,re,statistics,json
from pathlib import Path
p=Path('results-v2')
with (p/'summary.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
def v(r):
 m=re.search(r'(?:^|;\s*)split_variant=([^;]+)',r['notes']);return m.group(1) if m else 'random'
for split in ('random','grouped'):
 values=[float(r['auc']) for r in rows if r['dataset']=='parkinsons' and v(r)==split and r['feature_setting']=='A' and r['model']=='random_forest' and r['status']=='completed']
 print(f'PARKINSON_RF_SETTING_A split={split} n={len(values)} mean={statistics.mean(values)} sample_sd={statistics.stdev(values)}')
source=Path('qednet/models/quantum/qsvm.py').read_text(encoding='utf-8')
fit=slice(53*54//2); pred=slice(27*53)
assert 'if Y is X and j < i:' in source and 'self.kernel_evals_ += 1' in source and 'self.kernel_evals_ = 0' in source
print(f'QSVM_GATE_COUNTS train={53} test={27} fit_evaluations={fit.stop} predict_evaluations={pred.stop} combined={fit.stop+pred.stop}')
print('COUNT_METHOD=fit computes upper triangle including diagonal; prediction computes all test-by-train pairs; counter resets in fit and accumulates in predict')
raw=(p/'recompute_paired_tests_final.raw.txt').read_text(encoding='utf-8')
approx=[line for line in raw.splitlines() if 'WILCOXON_METHOD' in line and 'method=approx' in line]
print(f'WILCOXON_APPROX_ROW_COUNT={len(approx)}')
for line in approx:print(line)
print('WILCOXON_EXACT_ROW_COUNT='+str(sum('WILCOXON_METHOD' in x and 'method=exact' in x for x in raw.splitlines())))
