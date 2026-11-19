import csv,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
with (P/'summary.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
classical_models={'logistic_regression','random_forest','rbf_svm','mlp_default','parameter_matched_mlp_vqc','parameter_matched_mlp_reupload'}
quantum_models={'qsvm','vqc','reupload'}
cc=Counter(r['status'] for r in rows if r['model'] in classical_models)
qc=Counter(r['status'] for r in rows if r['model'] in quantum_models)
raw=json.loads((P/'quantum_raw.json').read_text(encoding='utf-8'))
errors=[r for r in raw if r.get('status')=='error']
vr=subprocess.run([sys.executable,str(P/'verify_report.py'),'--report','RESULTS_FINAL.md'],capture_output=True,text=True,encoding='utf-8')
pr=subprocess.run([sys.executable,str(P/'verify_paired_tests_report.py')],capture_output=True,text=True,encoding='utf-8')
vm=re.search(r'MISMATCH_COUNT=(\d+)',vr.stdout);pm=re.search(r'PVALUE_MISMATCH_COUNT=(\d+)',pr.stdout)
text=(P/'RESULTS_FINAL.md').read_text(encoding='utf-8')
checks={'classical_counts':cc['completed']==468 and cc['not_measured']==72 and cc['timeout']==0 and cc['error']==0,'quantum_counts':qc['completed']==135 and qc['not_measured']==0 and qc['timeout']==0 and qc['error']==0,'raw_quantum_count':len(raw)==146,'raw_error_count':len(errors)==1,'metric_mismatch_count_zero':bool(vm and int(vm.group(1))==0),'paired_mismatch_count_zero':bool(pm and int(pm.group(1))==0),'required_run_fold_section':'run_fold.py error and rerun' in text,'required_corrections_section':'Corrections from v2' in text}
print('CLASSICAL_DEDUP_STATUS_COUNTS='+json.dumps(dict(cc),sort_keys=True))
print('QUANTUM_DEDUP_STATUS_COUNTS='+json.dumps(dict(qc),sort_keys=True))
print(f'RAW_QUANTUM_RECORD_COUNT={len(raw)} RAW_QUANTUM_ERROR_COUNT={len(errors)}')
for r in errors:print(f"RAW_ERROR=dataset={r.get('dataset')} model={r.get('model')} seed={r.get('seed')} fold={r.get('fold')} feature_setting={r.get('feature_setting')} message={r.get('error','').splitlines()[-1]}")
print(f"RESULTS_FINAL_METRIC_MISMATCH_COUNT={vm.group(1) if vm else 'NOT ESTABLISHED'}")
print(f"RESULTS_FINAL_PAIRED_TEST_MISMATCH_COUNT={pm.group(1) if pm else 'NOT ESTABLISHED'}")
print(f"CONTAINS_run_fold.py_error_and_rerun={'run_fold.py error and rerun' in text}")
print(f"CONTAINS_Corrections_from_v2={'Corrections from v2' in text}")
for k,v in checks.items():print(f'CHECK_{k}={"PASS" if v else "FAIL"}')
print('OVERALL='+('PASS' if all(checks.values()) else 'FAIL'))
