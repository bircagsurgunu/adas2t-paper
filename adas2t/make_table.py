# make_table.py
import json, pandas as pd
from tqdm import tqdm
from feature_extractor import extract_features

IN  = "runs.jsonl"
OUT = "training_table.csv"

rows = []
with open(IN) as f:
    for line in tqdm(f, desc="featurising"):
        rec = json.loads(line)
        feats = extract_features(rec["wav"])
        row = {"uid": rec["uid"],
               "dataset": rec["dataset"],
               "model": rec["model"],
               "wer": rec["wer"],
               **{f"f{i}": feats[i] for i in range(len(feats))}}
        rows.append(row)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"wrote {OUT}")
