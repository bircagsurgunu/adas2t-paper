# crop_and_renumber.py
import pandas as pd
import sys

IN  = sys.argv[1] if len(sys.argv) > 1 else "training_table.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "training_table_no_whisper.csv"

df = pd.read_csv(IN)

# --- Identify meta vs feature columns ---
meta_cols = [c for c in ["uid", "dataset", "model", "wer"] if c in df.columns]
feat_cols = [c for c in df.columns if c.startswith("f")]

# --- If Whisper columns are still present, drop them by index ranges ---
# Index layout from your extractor (original, before removal):
# blocks and sizes:
sizes = [
    ("mfcc", 78),
    ("prosodic", 6),
    ("activity", 10),
    ("channel", 10),
    ("noise_modulation", 10),
    ("signal", 12),
    ("neural_whisper", 512),
    ("neural_w2v2", 768),
    ("neural_hubert", 768),
    ("embedding_stats_whisper", 6),
    ("embedding_stats_w2v2", 6),
    ("embedding_stats_hubert", 6),
    ("metadata", 1),
]

# Build start/end indices
ranges = {}
start = 0
for name, length in sizes:
    ranges[name] = range(start, start + length)
    start += length

# Columns to drop (if they exist)
to_drop = [f"f{i}" for i in list(ranges["neural_whisper"]) + list(ranges["embedding_stats_whisper"])]

# Drop only the ones that are present
to_drop = [c for c in to_drop if c in df.columns]
if to_drop:
    df = df.drop(columns=to_drop)

# --- Renumber remaining feature columns contiguously ---
feat_cols = [c for c in df.columns if c.startswith("f")]  # refresh after drop
# sort by numeric index just in case
feat_cols_sorted = sorted(feat_cols, key=lambda x: int(x[1:]))

# Build a mapping f(old) -> f(new)
rename_map = {old: f"f{i}" for i, old in enumerate(feat_cols_sorted)}

df = df[meta_cols + feat_cols_sorted].rename(columns=rename_map)

df.to_csv(OUT, index=False)
print(f"Saved {OUT} with {len(meta_cols)} meta columns and {len(feat_cols_sorted)} renumbered features.")
