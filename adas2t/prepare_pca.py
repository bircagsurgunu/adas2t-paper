import json, random, numpy as np, librosa, torch, tqdm
from feature_extractor import _wav2vec2_embed

IN_JSONL   = "runs.jsonl"       # produced by build_dataset.py
N_SAMPLES  = 3000               # clips to use for PCA (adjust)
SAMPLE_RATE = 16_000            # paper uses 16 kHz throughout

embeds = []
with open(IN_JSONL) as f:
    all_lines = list(f)
    if len(all_lines) < N_SAMPLES:
        print(f"Only {len(all_lines)} clips found; using all of them.")
        lines = all_lines
    else:
        lines = random.sample(all_lines, N_SAMPLES)
    for ln in tqdm.tqdm(lines, desc="embeddings"):
        wav_path = json.loads(ln)["wav"]
        wav, _   = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        embeds.append(_wav2vec2_embed(wav))

X = np.vstack(embeds)           # shape (N, 768)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=41, whiten=True, random_state=0).fit(X)
np.save("pca_w2v2_768x41.npy", pca.components_.T)   # (768, 41)
print("saved pca_w2v2_768x41.npy")
