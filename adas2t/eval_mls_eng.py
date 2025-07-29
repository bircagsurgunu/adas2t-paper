#!/usr/bin/env python
"""
Evaluate three base ASR models + two meta‑learners (XGB, MLP)
on parler‑tts/mls_eng (test split).

Assumes:
  • asr_runner.MODEL_BANK is pre‑loaded with callable transcribers
  • adas2t_xgb.json  (XGBoost regressor)
  • adas2t_mlp.pth   (MLP regressor trained with train_mlp.py)

Usage:
  python eval_mls_eng.py --max_clips 500
"""

import argparse, os, tempfile, json, base64, io
from collections import defaultdict

import numpy as np, torch, soundfile as sf, jiwer, pandas as pd, tqdm
from datasets import load_dataset, Audio
import xgboost as xgb

from asr_runner import MODEL_BANK, MODELS            # MODELS list order matters
from feature_extractor import extract_features, _SR
from train_mlp import WERMLP                         # import the class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── load meta‑learners ───────────────────────────────────────────────────────
xgb_model = xgb.Booster()
xgb_model.load_model("adas2t_xgb.json")

mlp_model = WERMLP().to(DEVICE)
mlp_model.load_state_dict(torch.load("adas2t_mlp.pth", map_location=DEVICE))
mlp_model.eval()

def choose_xgb(wav_path: str) -> int:
    feats = extract_features(wav_path)[None, :]
    preds = xgb_model.predict(xgb.DMatrix(feats))[0]
    return int(np.argmin(preds))

@torch.inference_mode()
def choose_mlp(wav_path: str) -> int:
    feats = extract_features(wav_path)[None, :].astype("float32")
    wer_hat = mlp_model(torch.from_numpy(feats).to(DEVICE)).cpu().numpy()[0]
    return int(wer_hat.argmin())

# ─── evaluation routine ──────────────────────────────────────────────────────
def evaluate(max_clips=500):
    ds = load_dataset("openslr/librispeech_asr", split="test.clean", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=_SR))

    refs_raw        = defaultdict(list)   # model → refs
    hypos_raw       = defaultdict(list)   # model → hypos
    refs_xgb, hyp_xgb = [], []
    refs_mlp, hyp_mlp = [], []

    for i, ex in tqdm.tqdm(enumerate(ds), total=max_clips, desc="clips", unit="clip"):
        if i >= max_clips:
            break

        reference = ex["text"].strip().lower()
        audio_arr = ex["audio"]["array"]

        # temp wav once per clip
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_arr, _SR)
            wav_path = tmp.name

        clip_preds = []
        for mdl in MODELS:
            pred = MODEL_BANK[mdl](audio_arr).strip().lower()
            clip_preds.append(pred)
            refs_raw[mdl].append(reference)
            hypos_raw[mdl].append(pred)

        # meta selections
        idx_xgb = choose_xgb(wav_path)
        idx_mlp = choose_mlp(wav_path)

        refs_xgb.append(reference)
        hyp_xgb.append(clip_preds[idx_xgb])

        refs_mlp.append(reference)
        hyp_mlp.append(clip_preds[idx_mlp])

        os.unlink(wav_path)

    # ─── compute WERs ────────────────────────────────────────────────────────
    wer_scores = {
        mdl: jiwer.wer(refs_raw[mdl], hypos_raw[mdl]) for mdl in MODELS
    }
    wer_scores["ADAS2T‑XGB"] = jiwer.wer(refs_xgb, hyp_xgb)
    wer_scores["ADAS2T‑MLP"] = jiwer.wer(refs_mlp, hyp_mlp)

    # ─── pretty print ────────────────────────────────────────────────────────
    df = (pd.Series(wer_scores)
            .sort_values()
            .rename("WER")
            .to_frame())
    df.index.name = f"parler‑tts/mls_eng  (N={len(refs_xgb)})"
    print("\n" + df.to_string(float_format=lambda x: f"{x:.3%}"))

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_clips", type=int, default=500,
                    help="number of clips to score (default 500)")
    args = ap.parse_args()
    evaluate(args.max_clips)
