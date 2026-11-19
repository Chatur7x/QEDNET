import argparse,csv,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NUM=r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
STAT_RE=re.compile(rf'^\s*({NUM})\s*±\s*({NUM})\s*\(n=(\d+)\)\s*$')
def split_notes(s):
 m=re.search(r'(?:^|;\s*)split_variant=([^;]+)',s); return m.group(1) if m else 'random'
def parse(report):
 ds=split_name=setting=None; out=[]
 for line in report.read_text(encoding='utf-8').splitlines():
  if line.startswith('### '):
   h=line.lower(); setting=re.search(r'setting ([ab])$',h).group(1).upper() if re.search(r'setting ([ab])$',h) else None
   if 'parkinson' in h: ds='parkinsons'
   elif 'breast cancer' in h: ds='breast_cancer'
   elif 'heart disease' in h: ds='heart_disease'
   elif 'diabetes' in h: ds='diabetes'
   split_name='grouped' if 'grouped folds' in h else 'random'
  if not line.startswith('|') or line.startswith('| ---') or line.startswith('| Model '): continue
  cells=[x.strip() for x in line.strip().strip('|').split('|')]
  if len(cells)<8 or ds is None or setting is None:continue
  model=cells[0]
  if model not in ('logistic_regression','random_forest','rbf_svm','mlp_default','parameter_matched_mlp_vqc','parameter_matched_mlp_reupload','qsvm','vqc','reupload'):continue
  out.append((ds,split_name,setting,model,cells))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',default='RESULTS.md');a=ap.parse_args(); report=ROOT/a.report
 with (ROOT/'summary.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 groups={}
 for r in rows:
  if r['status']!='completed':continue
  k=(r['dataset'],split_notes(r['notes']),r['feature_setting'],r['model']);groups.setdefault(k,[]).append(r)
 diffs=[]
 names=('AUC mean ± SD (n)','Accuracy mean ± SD (n)','Balanced accuracy mean ± SD (n)','F1 mean ± SD (n)','Fit seconds mean ± SD (n)')
 for d,split,setting,model,cells in parse(report):
  key=(d,split,setting,model); vals=groups.get(key,[])
  if not vals:continue
  for i,col in enumerate(('auc','accuracy','balanced_accuracy','f1','train_seconds'),start=2):
   m=STAT_RE.match(cells[i])
   if not m:continue
   report_mean,report_sd,n=float(m.group(1)),float(m.group(2)),int(m.group(3))
   actual=[float(r[col]) for r in vals]; mean=statistics.mean(actual); sd=statistics.stdev(actual) if len(actual)>1 else None
   if n!=len(actual) or abs(report_mean-mean)>1e-9*max(abs(mean),1e-300) or (sd is not None and abs(report_sd-sd)>1e-9*max(abs(sd),1e-300)):
    if abs(report_mean-mean)>1e-9*max(abs(mean),1e-300): diffs.append((d,split,setting,model,names[i-2]+' mean',m.group(1),str(mean),len(actual)))
    if sd is not None and abs(report_sd-sd)>1e-9*max(abs(sd),1e-300): diffs.append((d,split,setting,model,names[i-2]+' SD',m.group(2),str(sd),len(actual)))
 print('SECTION|MODEL|COLUMN|REPORT_VALUE|RECOMPUTED_VALUE|N')
 for row in diffs:print('|'.join(map(str,row)))
 print('MISMATCH_COUNT='+str(len(diffs)))
 print('DATA_ROWS_PARSED='+str(len(parse(report))))
 print('KNOWN_HEART_SETTING_A_TIMING_MISMATCHES_FOUND='+str(sum(1 for x in diffs if x[0]=='heart_disease' and x[2]=='A' and x[3] in ('qsvm','vqc','reupload') and x[4].startswith('Fit seconds'))))
if __name__=='__main__':main()
