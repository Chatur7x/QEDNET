import json
from pathlib import Path
q=json.loads(Path('results-v2/rerun_heart_qsvm.raw.txt').read_text(encoding='utf-8'))
others=[json.loads(Path(f'results-v2/rerun_heart_{m}.raw.txt').read_text(encoding='utf-8')) for m in ('vqc','reupload')]
rows=[q]+others
Path('results-v2/quantum_rerun_heart_fold0.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('RERUNS='+json.dumps([{'model':r['model'],'auc':r['auc'],'status':r['status']} for r in rows],separators=(',',':')))
