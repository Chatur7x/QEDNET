import pandas as pd
from pathlib import Path
p=Path('data/raw/heart_cleveland.data')
raw=pd.read_csv(p,header=None,na_values='?')
clean=raw.dropna()
print(f'file={p}')
print(f'raw_data_rows={len(raw)}')
print(f'clean_rows_dropna={len(clean)}')
print(f'dropped_rows={len(raw)-len(clean)}')
print(f'max_source_index={int(clean.index.max())}')
print(f'position_valid_max={len(clean)-1}')
