#!/usr/bin/env python
"""
Build PCA reducers for the three SSL embeddings used in the paper:

  Whisper encoder 1280 → 64
  Wav2Vec2-Base    768 → 32
  HuBERT-Base     1024 → 32

Output files:
  pca_whisper_1280x64.npy
  pca_w2v2_768x32.npy
  pca_hubert_1024x32.npy
"""

import json, random, numpy as np, librosa, tqdm, torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoProcessor, Wav2Vec2FeatureExtractor

_SR = 16_000
SAMPLE_FILE = "runs.jsonl"
N_CLIPS = 10000                # feel free to raise/lower
MIN_SAMPLES = 0.05*_SR  # skip clips shorter than 50 ms

# ─── load encoders (GPU) ─────────────────────────────────────────────────────
proc_w  = AutoProcessor.from_pretrained("openai/whisper-large-v3")
enc_w   = AutoModel.from_pretrained("openai/whisper-large-v3",
                                    add_cross_attention=False
                                   ).to("cuda").eval()

fe_ssl  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
enc_w2v = AutoModel.from_pretrained("facebook/wav2vec2-base-960h"
                                   ).to("cuda").eval()
enc_hub = AutoModel.from_pretrained("facebook/hubert-base-ls960"
                                   ).to("cuda").eval()

@torch.inference_mode()
def embed_whisper(y):
    feats = proc_w.feature_extractor(y, sampling_rate=_SR,
                                     return_tensors="pt").to("cuda")
    hidden = enc_w.encoder(input_features=feats.input_features).last_hidden_state
    return hidden.mean(1).squeeze().cpu().numpy()

@torch.inference_mode()
def embed_ssl(y, encoder):
    feats = fe_ssl(y, sampling_rate=_SR, return_tensors="pt").to("cuda")
    return encoder(**feats).last_hidden_state.mean(1).squeeze().cpu().numpy()

# ─── gather embeddings ───────────────────────────────────────────────────────
W, V, H = [], [], []
with open(SAMPLE_FILE) as f:
    lines = f.readlines()

pool = random.sample(lines, min(N_CLIPS, len(lines)))

for line in tqdm.tqdm(pool, desc="embedding clips"):
    wav = json.loads(line)["wav"]
    y, _ = librosa.load(wav, sr=_SR, mono=True)
    if len(y) < MIN_SAMPLES:      # ← NEW: skip ultra-short clips
        continue

    W.append(embed_whisper(y))
    V.append(embed_ssl(y, enc_w2v))
    H.append(embed_ssl(y, enc_hub))


# ─── fit PCAs ────────────────────────────────────────────────────────────────
print("Fitting PCA • Whisper")
P_W = PCA(n_components=64, whiten=True, random_state=0).fit(np.vstack(W))
np.save("pca_whisper_1280x64.npy", P_W.components_.T)

print("Fitting PCA • wav2vec2")
P_V = PCA(n_components=32, whiten=True, random_state=0).fit(np.vstack(V))
np.save("pca_w2v2_768x32.npy", P_V.components_.T)

print("Fitting PCA • HuBERT")
P_H = PCA(n_components=32, whiten=True, random_state=0).fit(np.vstack(H))
np.save("pca_hubert_1024x32.npy", P_H.components_.T)

print("✅  PCA files written:")
print("  • pca_whisper_1280x64.npy")
print("  • pca_w2v2_768x32.npy")
print("  • pca_hubert_1024x32.npy")
