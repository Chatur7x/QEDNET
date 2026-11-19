import csv,json,statistics
from pathlib import Path
p=Path('results-v2/repeat_qsvm_timing.raw.txt')
rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
times=[r['fit_seconds'] for r in rows if r.get('status')=='completed']
out={'dataset':'heart_disease','model':'qsvm','split_variant':'random','feature_setting':'A','seed':42,'fold':0,'n_samples':80,'n_train':53,'n_test':27,'simulator':'default.qubit','shots':None,'records':rows,'estimator_fit_seconds':times,'min_fit_seconds':min(times),'median_fit_seconds':statistics.median(times),'max_fit_seconds':max(times)}
Path('results-v2/timing_repeats_heart_qsvm.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'estimator_fit_seconds':times,'min_fit_seconds':min(times),'median_fit_seconds':statistics.median(times),'max_fit_seconds':max(times)},separators=(',',':')))
