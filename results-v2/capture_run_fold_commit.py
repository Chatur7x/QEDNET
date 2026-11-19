import subprocess
from pathlib import Path
cmds=[['git','log','-1','--oneline','--','results-v2/run_fold.py'],['git','show','--stat','--oneline','HEAD'],['git','status','--short','--branch']]
lines=[]
for c in cmds:
 r=subprocess.run(c,capture_output=True,text=True);lines.extend(['$ '+' '.join(c),r.stdout.rstrip(),r.stderr.rstrip()])
print('\n'.join(lines))
