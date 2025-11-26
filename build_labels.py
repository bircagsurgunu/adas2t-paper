# build_labels.py — create labels without ties
import pandas as pd

df = pd.read_csv("training_table_25k_dropped_voxpopuli.csv")  # columns: uid, model, wer, f0..fN (features ignored here)

# pivot to uid × model WER table
pivot = df.pivot_table(index="uid", columns="model", values="wer")

# find rows where the min value is unique (no tie)
min_vals = pivot.min(axis=1)
is_min = pivot.eq(min_vals, axis=0)
unique_min_mask = is_min.sum(axis=1) == 1

# keep only non-tied rows
pivot_uniq = pivot[unique_min_mask]

# pick the (unique) best model per clip
labels = pivot_uniq.idxmin(axis=1)  # now guaranteed unique

# write outputs
labels.to_csv("clip_labels_25k_dropped_voxpopuli.csv")        # uid,best_model
pivot.index.difference(pivot_uniq.index).to_series().to_csv("tied_uids.csv", index=False, header=["uid"])
print(f"wrote clip_labels_25k_dropped_voxpopuli.csv (rows: {len(labels)})")
print(f"excluded ties → tied_uids.csv (rows: {len(pivot) - len(labels)})")
