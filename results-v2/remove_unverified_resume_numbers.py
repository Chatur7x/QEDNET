from pathlib import Path
import re
p=Path('results-v2/RESULTS_FINAL.md')
s=p.read_text(encoding='utf-8')
s,n=re.subn(r'Do not describe the earlier 0\.92.*?not re-established by these files\.', "Do not describe the earlier full-data Parkinson's result as this 80-row benchmark; its exact source values are not re-established by these files.", s)
print(f'REPLACED_OLD_SAMPLE_CLAIM={n}')
print(f'OLD_CLAIM_REMAINS={bool(re.search(r"0\\.92.*0\\.69",s))}')
p.write_text(s,encoding='utf-8')
