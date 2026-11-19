import json,subprocess
from pathlib import Path
p=Path('results-v2')
commands=[['git','log','--oneline','--','results-v2/run_fold.py'],['git','diff','--','results-v2/run_fold.py'],['git','status','--short','--branch'],['git','ls-files','--error-unmatch','results-v2/run_fold.py']]
lines=[]
for cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
 lines += ['$ '+' '.join(cmd),f'EXIT={r.returncode}',r.stdout.rstrip(),r.stderr.rstrip()]
 for f in ('results-v2/run_fold.py','results-v2/splits.json','data/raw/heart_cleveland.data'):
  x=Path(f).stat(); lines.append(f'FILE {f} bytes={x.st_size} mtime_ns={x.st_mtime_ns}')
(p/'run_fold_git_history.raw.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

