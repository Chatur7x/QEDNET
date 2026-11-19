import csv,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
lines=(P/'RESULTS_FINAL.md').read_text(encoding='utf-8').splitlines()
if '## Corrections from v2' in lines:
 lines=lines[:lines.index('## Corrections from v2')]
with (P/'paired_tests_recomputed.csv').open(newline='',encoding='utf-8-sig') as f: stats=list(csv.DictReader(f))
by={(r['comparison'],r['dataset'],r['split_variant']):r for r in stats}
changes=[];out=[];section='';last_primary_row=-1
for line in lines:
 if line.startswith('#### '):section=line[5:].lower()
 if line.startswith('### '):section=line[4:].lower()
 if line.startswith('|') and not line.startswith('| ---'):
  cells=[c.strip() for c in line.strip('|').split('|')]
  if cells and cells[0]=='Dataset / split':
   cells[6]='Observed direction (exploratory)';line='| '+' | '.join(cells)+' |'
  elif len(cells)==7 and cells[0] in ('breast_cancer / random','heart_disease / random','parkinsons / random','parkinsons / grouped','diabetes / random'):
   d,v=cells[0].split(' / ');r=by[('best_of_fold',d,v)]
   for ix,col in ((4,'t_test_p'),(5,'wilcoxon_p')):
    old=float(cells[ix]);new=float(r[col])
    if abs(old-new)>1e-9*max(abs(new),1e-300):changes.append((f'{d} / {v}','best_of_fold',col,cells[ix],r[col]));cells[ix]=r[col]
   # Every paired difference is negative in the saved source, so label direction without inferential language.
   cells[6]=f"quantum AUC lower in this sample; t-test p = {r['t_test_p']}, Wilcoxon p = {r['wilcoxon_p']} (exploratory)"
   line='| '+' | '.join(cells)+' |';last_primary_row=len(out)
  elif len(cells)==7 and cells[0].startswith('parameter_matched_mlp_') and cells[1] in ('qsvm','vqc','reupload'):
   d='parkinsons' if section.startswith('parkinson') else ('breast_cancer' if section.startswith('breast cancer') else ('heart_disease' if section.startswith('heart disease') else 'diabetes'))
   v='grouped' if 'grouped' in section else 'random';r=by[(cells[0]+' - '+cells[1],d,v)]
   for ix,col in ((5,'t_test_p'),(6,'wilcoxon_p')):
    old=float(cells[ix]);new=float(r[col])
    if abs(old-new)>1e-9*max(abs(new),1e-300):changes.append((f'{d} / {v}',cells[0]+' - '+cells[1],col,cells[ix],r[col]));cells[ix]=r[col]
   line='| '+' | '.join(cells)+' |'
 out.append(line)
# Insert required exploratory-independence caveat directly after primary paired table.
insert_at=last_primary_row+1
out.insert(insert_at,'')
out.insert(insert_at+1,'These nine pairs per row reuse the same three folds across three seeds, so they are not independent. The p-values are exploratory, unadjusted for multiple comparisons, and should not be read as confirmation that classical models outperform quantum models in general.')
# Add reproducible run-fold error explanation with limitations plainly marked.
out.extend(['','## run_fold.py error and rerun','','`quantum_raw.json` records the failed heart QSVM request as index 301 against an array of size 297. `heart_row_count_audit.raw.txt` reports 303 source-file rows, 297 after `dropna()`, six dropped rows, and max retained source index 301 (cleaned array positions end at 296). `make_splits.py` and `splits.json` distinguish `sample_source_row_indices` from `sample_local_row_indices`; source row 301 is an audit ID, not a valid position in the 297-row cleaned array. The current `run_fold.py` indexes with `sample_local_row_indices` and checks that positions fit the loaded array.','','The saved `run_fold.py` is untracked: `run_fold_git_history.raw.txt` shows no tracked log or diff and `git ls-files --error-unmatch` fails. The current source is consistent with a code-side index correction; the exact historical before/after code change is NOT ESTABLISHED from git history. The data file and `splits.json` are present in the saved evidence, and the row audit explains the failure, but no tracked snapshot exists to conclusively attribute what changed between failure and rerun.','','The rerun command uses heart disease, random split, seed 42, fold 0, setting A; its record reports n=80, train=53, test=27. `splits.json` fold 0 has the same 80-row sample and 53/27 sizes. However, `quantum_rerun_heart_fold0.json` does not store sample source row IDs or train/test fold index arrays, so exact source-row and fold-index equality between the rerun record and manifest is NOT ESTABLISHED from the saved rerun record. The rerun command and manifest identify the same requested split key, but do not prove the omitted index arrays.','','## Corrections from v2','', 'Table metric changes (source: `summary.csv`):'])
for sec,model,col,old,new in [
 ('heart disease — random folds — setting A','qsvm','Fit seconds', '8.018959766666537 ± 3.9284810318743806 (n=9)','8.501741977777783 ± 3.8932783736114085 (n=9)'),
 ('heart disease — random folds — setting A','reupload','Fit seconds','69.41908447777799 ± 26.84127206128184 (n=9)','76.17163291111117 ± 28.48115382996333 (n=9)'),
 ('heart disease — random folds — setting A','vqc','Fit seconds','44.218291000000235 ± 9.580374429809995 (n=9)','46.85081336666654 ± 10.974763470096034 (n=9)')]:
 out.append(f'- {sec}; {model}; {col}: `{old}` -> `{new}`.')
out += ['', 'Paired-test p-value changes (source: `paired_tests_recomputed.csv`; scipy 1.18.1; values use the reported exact/asymptotic rule):']
for dv,cmp,col,old,new in changes:out.append(f'- {dv}; {cmp}; {col}: {old} -> {new}.')
out += ['', 'The same three folds were reused across three estimator seeds, so these nine pairs are correlated. The paired p-values are exploratory and unadjusted.']
(P/'RESULTS_FINAL.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print(f'REPORT=RESULTS_FINAL.md')
print(f'PVALUE_CELLS_CHANGED={len(changes)}')
for x in changes: print('|'.join(x))


