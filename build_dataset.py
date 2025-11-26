# build_dataset.py
"""
Create runs.jsonl with per‑clip *stable* UID so the pivot in train_xgb.py
gets a full rectangular label matrix (no NaNs).
"""
import os, json, uuid, jiwer, numpy as np
from tqdm import tqdm
from asr_runner import iter_datasets_and_models   # generator we wrote earlier

OUT = "runs.jsonl"

# map wav_path → uid  (tempfiles are unique per clip, reused for all models)
path2uid = {}

def get_uid(wav_path: str) -> str:
    """Return a stable UID for this clip path (create once)."""
    uid = path2uid.get(wav_path)
    if uid is None:
        # use basename so the file moves across dirs OK, or just uuid4()
        uid = os.path.basename(wav_path) or str(uuid.uuid4())
        path2uid[wav_path] = uid
    return uid

with open(OUT, "w") as fout:
    for ds, mdl, ref, hyp, wav in tqdm(iter_datasets_and_models(),
                                       desc="building JSONL"):
        wer_val = jiwer.wer(ref, hyp)
        if np.isnan(wer_val) or np.isinf(wer_val):
            wer_val = 1.0                     # worst‑case

        rec = {
            "uid"    : get_uid(wav),          # ← same for all models
            "dataset": ds,
            "model"  : mdl,
            "wav"    : wav,
            "wer"    : float(wer_val)
        }
        fout.write(json.dumps(rec) + "\n")

print(f"wrote {OUT}  •  unique clips: {len(path2uid)}")
