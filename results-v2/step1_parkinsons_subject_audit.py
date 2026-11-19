import re
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data" / "raw" / "parkinsons.data")
ids = df["name"].str.extract(r"phon_R\d+_(S\d+)_\d+")[0]
print(f"rows={len(df)}")
print(f"unmatched_name_rows={int(ids.isna().sum())}")
counts = ids.value_counts().sort_index()
print(f"unique_subjects={len(counts)}")
print(f"min_rows_per_subject={int(counts.min())}")
print(f"max_rows_per_subject={int(counts.max())}")
print("rows_per_subject")
for subject, count in counts.items():
    print(f"{subject}={int(count)}")
