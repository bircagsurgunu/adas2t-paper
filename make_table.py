# make_table.py

import json, pandas as pd
from tqdm import tqdm
from feature_extractor_paper import extract_features

IN  = "runs.jsonl"
OUT = "training_table_25k.csv"

rows = []
for line in tqdm(open(IN), desc="featurising"):
    rec = json.loads(line)
    try:
        feats = extract_features(rec["wav"])
    except Exception as e:
        print(f"  ⚠️ skipping {rec['wav']}: {e}")
        continue

    row = {
        "uid": rec["uid"],
        "dataset": rec["dataset"],
        "model": rec["model"],
        "wer": rec["wer"],
        **{f"f{i}": feats[i] for i in range(len(feats))}
    }
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(rows)} rows)")
