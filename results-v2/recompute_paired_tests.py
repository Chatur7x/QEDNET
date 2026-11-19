import argparse,csv,re,statistics
from pathlib import Path
from scipy import __version__ as scipy_version
from scipy.stats import ttest_1samp,wilcoxon
P=Path(__file__).resolve().parent
FIELDS=['comparison','dataset','split_variant','n','mean_difference','sample_sd','t_test_p','wilcoxon_p']
def keynote(s):
 m=re.search(r'(?:^|;\s*)split_variant=([^;]+)',s);return m.group(1) if m else 'random'
def calc(values):
 n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else float('nan')
 nonzero_abs=[abs(x) for x in values if x!=0]
 ties=len(set(nonzero_abs)) != len(nonzero_abs)
 zeros=any(x==0 for x in values)
 method='exact' if n<=25 and not ties and not zeros else 'approx'
 w=wilcoxon(values,alternative='two-sided',method=method,zero_method='wilcox')
 reason='absolute-difference ties' if ties else ('zero differences' if zeros else 'no ties or zeros')
 return {'n':n,'mean_difference':mean,'sample_sd':sd,'t_test_p':float(ttest_1samp(values,0,alternative='two-sided').pvalue),'wilcoxon_p':float(w.pvalue),'wilcoxon_method':method,'wilcoxon_reason':reason}
def report_pvalues(text):
 found={}
 lines=text.splitlines();section=''
 for line in lines:
  if line.startswith('#### '):section=line[5:].strip().lower()
  if not line.startswith('|'):continue
  c=[x.strip() for x in line.strip('|').split('|')]
  if len(c)==7 and c[0] in ('breast_cancer / random','heart_disease / random','parkinsons / random','parkinsons / grouped','diabetes / random'):
   d,v=c[0].split(' / ');found[('best_of_fold',d,v,'','')]=(c[4],c[5])
  elif len(c)==7 and c[0].startswith('parameter_matched_mlp_') and c[1] in ('qsvm','vqc','reupload'):
   d='parkinsons' if section.startswith("parkinson") else ('breast_cancer' if section.startswith('breast cancer') else ('heart_disease' if section.startswith('heart disease') else 'diabetes'))
   v='grouped' if 'grouped' in section else 'random'
   found[('parameter_matched',d,v,c[0],c[1])]=(c[5],c[6])
 return found
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',default='RESULTS.md');args=ap.parse_args()
 with (P/'paired_differences.csv').open(newline='',encoding='utf-8-sig') as f: paired=list(csv.DictReader(f))
 by={}
 for r in paired:by.setdefault((r['dataset'],r['split_variant']),[]).append(r)
 with (P/'summary.csv').open(newline='',encoding='utf-8-sig') as f: summ=list(csv.DictReader(f))
 groups={}
 for r in summ:
  if r['status']=='completed':groups.setdefault((r['dataset'],keynote(r['notes']),r['feature_setting'],r['model'],r['seed'],r['fold']),r)
 output=[];actual={}
 for (d,v),rs in sorted(by.items()):
  vals=[float(r['difference']) for r in rs];stats=calc(vals); comp='best_of_fold';actual[(comp,d,v,'','')]=stats
  output.append(dict(comparison=comp,dataset=d,split_variant=v,n=stats['n'],mean_difference=stats['mean_difference'],sample_sd=stats['sample_sd'],t_test_p=stats['t_test_p'],wilcoxon_p=stats['wilcoxon_p']))
 qmodels=('qsvm','vqc','reupload'); pm_models=('parameter_matched_mlp_reupload','parameter_matched_mlp_vqc')
 datasets=sorted({(k[0],k[1]) for k in groups if k[2]=='A'})
 for d,v in datasets:
  for pm in pm_models:
   for qm in qmodels:
    vals=[]
    for seed in ('42','1729','2024'):
     for fold in ('0','1','2'):
      a=groups.get((d,v,'A',pm,seed,fold));b=groups.get((d,v,'A',qm,seed,fold))
      if a and b:vals.append(float(a['auc'])-float(b['auc']))
    if not vals:continue
    stats=calc(vals);actual[('parameter_matched',d,v,pm,qm)]=stats
    output.append(dict(comparison=f'{pm} - {qm}',dataset=d,split_variant=v,n=stats['n'],mean_difference=stats['mean_difference'],sample_sd=stats['sample_sd'],t_test_p=stats['t_test_p'],wilcoxon_p=stats['wilcoxon_p']))
 with (P/'paired_tests_recomputed.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(output)
 text=(P/args.report).read_text(encoding='utf-8'); reported=report_pvalues(text); mismatches=[]
 for k,stats in actual.items():
  if k not in reported:continue
  oldt,oldw=map(float,reported[k]);
  for name,old,new in [('t_test_p',oldt,stats['t_test_p']),('wilcoxon_p',oldw,stats['wilcoxon_p'])]:
   if abs(old-new)>1e-9*max(abs(new),1e-300):mismatches.append((k,name,old,new))
 print(f'SCIPY_VERSION={scipy_version}')
 print(f'WILCOXON_RULE=exact when n<=25, no absolute-difference ties, and no zero differences; otherwise approx')
 print(f'COMPARISON_ROWS={len(output)} REPORT_PVALUE_PAIRS_PARSED={len(reported)}')
 for k,st in actual.items(): print(f"WILCOXON_METHOD key={k} method={st['wilcoxon_method']} reason={st['wilcoxon_reason']} n={st['n']}")
 print('PVALUE_MISMATCH|COMPARISON_KEY|COLUMN|REPORT_VALUE|RECOMPUTED_VALUE')
 for k,col,old,new in mismatches:print(f'{k}|{col}|{old}|{new}')
 print(f'PVALUE_MISMATCH_COUNT={len(mismatches)}')
 with (P/'paired_tests_recomputed.csv').open(newline='',encoding='utf-8-sig') as f:
  rd=csv.DictReader(f);rows=list(rd);print(f'CSV_COLUMNS_MATCH={rd.fieldnames==FIELDS} ROW_SHAPES_VALID={all(None not in r for r in rows)} ROW_COUNT={len(rows)}');print('FIRST_3_ROWS='+str(rows[:3]))
if __name__=='__main__':main()
