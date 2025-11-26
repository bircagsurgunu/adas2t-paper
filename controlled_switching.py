#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled switching (CV-aware):
  • Build sequence from clips where one base model wins by WER
  • Use XGB + MLP CV ensembles (fallback to single) to compute P(Parakeet)
  • Plot router probabilities & selections

Outputs:
  • switching_strict_log.csv
  • figure_switching_strict.pdf
"""

import argparse, json, time, tempfile, os, re, glob
from collections import deque

import numpy as np
import pandas as pd
import torch, soundfile as sf, jiwer, xgboost as xgb
from datasets import load_dataset, Audio
import matplotlib, matplotlib.pyplot as plt

# ─────────── Config ───────────
_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER = "openai/whisper-large-v3"
PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"

# Whisper-friendly sources (clean, read speech)
WHISPER_SOURCES = [
    ("openslr/librispeech_asr", None, "test.clean", "text", "Libri Test-Clean"),
    ("facebook/voxpopuli", "en_accented", "test", "raw_text", "VoxPopuli"),
]

# Parakeet-friendly sources (far-field meetings)
PARAKEET_SOURCES = [
    ("edinburghcstr/ami", "sdm", "test", "text", "AMI SDM"),
    ("edinburghcstr/ami", "ihm", "test", "text", "AMI IHM"),
]

# ─────────── Import your code ───────────
from asr_runner import MODEL_BANK, MODELS
from feature_extractor_paper import extract_features, _SR as _SR_FX
assert _SR_FX == _SR, "Feature extractor SR mismatch."
MIN_SEC = 0.2
MIN_SAMPLES = int(MIN_SEC * _SR)

# ─────────── Ensemble loaders & slicing ───────────
def _read_first_json(path_list):
    for p in path_list:
        try:
            with open(p) as f:
                return json.load(f), p
        except FileNotFoundError:
            continue
    return None, None

def _indices_from_feat_names(names):
    idx = []
    for n in names or []:
        if isinstance(n, str) and n.startswith("f"):
            try:
                idx.append(int(n[1:]))
            except ValueError:
                pass
    return idx

def _rebuild_scaler(state):
    from sklearn.preprocessing import StandardScaler
    if state is None: return None
    sc = StandardScaler(**state.get("params", state.get("params", {})))
    sc.mean_ = np.array(state["mean_"], dtype=np.float64)
    sc.scale_ = np.array(state["scale_"], dtype=np.float64)
    sc.var_ = np.array(state["var_"], dtype=np.float64)
    sc.n_features_in_ = int(state["n_features_in_"])
    sc.n_samples_seen_ = int(state["n_samples_seen_"])
    return sc

def _infer_in_dim_from_state_dict(sd) -> int:
    for k, v in sd.items():
        if k.endswith("net.0.weight") and v.ndim == 2:
            return int(v.shape[1])
    for k, v in sd.items():
        if k.endswith("0.weight") and v.ndim == 2:
            return int(v.shape[1])
    raise RuntimeError("Could not infer MLP input dimension from checkpoint.")

# XGB pack
XGB_META, _ = _read_first_json(["adas2t_xgb_meta_cv.json", "adas2t_xgb_meta.json"])
FEAT_COLS_XGB = (XGB_META or {}).get("feature_names")
try:
    LABELS_XGB = np.load("label_order_xgb.npy", allow_pickle=True)
except Exception:
    LABELS_XGB = None

def _load_xgb_pack():
    pack = {"models": [], "best_iters": [], "is_ensemble": False}
    fold_paths = sorted(glob.glob("adas2t_xgb_cls_fold*.json"))
    if fold_paths and XGB_META and "best_iterations" in XGB_META:
        best_iters = list(map(int, XGB_META["best_iterations"]))
        if len(best_iters) != len(fold_paths):
            pad_val = best_iters[-1] if best_iters else 0
            best_iters = (best_iters + [pad_val] * len(fold_paths))[:len(fold_paths)]
        for p in fold_paths:
            b = xgb.Booster(); b.load_model(p); pack["models"].append(b)
        pack["best_iters"] = best_iters
        pack["is_ensemble"] = True
        print(f"[XGB] CV ensemble loaded: {len(pack['models'])} folds.")
        return pack
    # fallback
    b = xgb.Booster(model_file="adas2t_xgb_cls.json")
    best_it = int((XGB_META or {}).get("best_iteration", 0))
    pack["models"] = [b]; pack["best_iters"] = [best_it]; pack["is_ensemble"] = False
    print(f"[XGB] Single model loaded (best_iteration={best_it}).")
    return pack

XGB_PACK = _load_xgb_pack()

# MLP pack
MLP_META, _ = _read_first_json(["adas2t_mlp_meta_cv.json"])
FEAT_COLS_MLP = None
MLP_LABELS = None
from train_mlp_cls import MLP  # uses same architecture

def _load_mlp_pack():
    global FEAT_COLS_MLP, MLP_LABELS
    pack = {"models": [], "scalers": [], "thresholds": [], "is_ensemble": False}

    fold_paths = sorted(glob.glob("adas2t_mlp_cls_fold*.pth"))
    if fold_paths and MLP_META:
        thr_list = MLP_META.get("thresholds", None)
        if thr_list is None or len(thr_list) != len(fold_paths):
            fallback_thr = float(np.mean(thr_list)) if thr_list else 0.5
            thr_list = (thr_list or []) + [fallback_thr] * len(fold_paths)
            thr_list = thr_list[:len(fold_paths)]
        classes_set = None
        for p in fold_paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = ckpt["model"]; in_dim = _infer_in_dim_from_state_dict(sd)
            mdl = MLP(in_dim); mdl.load_state_dict(sd, strict=False); mdl.eval().to(DEVICE)
            sc = _rebuild_scaler(ckpt.get("scaler_state"))
            pack["models"].append(mdl); pack["scalers"].append(sc)
            pack["thresholds"].append(float(ckpt.get("threshold", 0.5)))
            if FEAT_COLS_MLP is None:
                FEAT_COLS_MLP = ckpt.get("feature_names") or (MLP_META or {}).get("feature_names")
            if classes_set is None:
                classes_set = list(ckpt["classes"])
        if classes_set is None:
            raise RuntimeError("MLP CV: missing 'classes' in checkpoints.")
        MLP_LABELS = classes_set
        pack["is_ensemble"] = True
        print(f"[MLP] CV ensemble loaded: {len(pack['models'])} folds.")
        return pack

    # fallback single
    ckpt = torch.load("adas2t_mlp_cls.pth", map_location="cpu", weights_only=False)
    sd = ckpt["model"]; in_dim = _infer_in_dim_from_state_dict(sd)
    mdl = MLP(in_dim); mdl.load_state_dict(sd, strict=False); mdl.eval().to(DEVICE)
    sc = _rebuild_scaler(ckpt.get("scaler_state"))
    pack["models"] = [mdl]; pack["scalers"] = [sc]
    pack["thresholds"] = [float(ckpt.get("threshold", 0.5))]
    FEAT_COLS_MLP = ckpt.get("feature_names")
    MLP_LABELS = list(ckpt["classes"])
    pack["is_ensemble"] = False
    print("[MLP] Single model loaded.")
    return pack

MLP_PACK = _load_mlp_pack()

# Parakeet class indices
if LABELS_XGB is None:
    raise RuntimeError("label_order_xgb.npy not found; required to index Parakeet class for XGB.")
try:
    PARAKEET_IDX_XGB = LABELS_XGB.tolist().index(PARAKEET)
except ValueError:
    raise RuntimeError(f"XGB classes {LABELS_XGB} must include '{PARAKEET}'")

if (MLP_LABELS is None) or (PARAKEET not in MLP_LABELS) or (len(MLP_LABELS) != 2):
    raise RuntimeError(f"MLP classes invalid; expected binary incl. '{PARAKEET}', got: {MLP_LABELS}")

# Feature cache per file
def _cached_feats(path, cache, family: str):
    if cache.get("full") is None:
        t0 = time.perf_counter()
        f_full = extract_features(path).astype("float32")[None, :]
        cache["full"] = f_full
        cache["feat_time"] = time.perf_counter() - t0

    if family == "xgb":
        if cache.get("xgb") is None:
            if FEAT_COLS_XGB:
                keep = _indices_from_feat_names(FEAT_COLS_XGB)
                keep = [i for i in keep if 0 <= i < cache["full"].shape[1]]
                cache["xgb"] = cache["full"][:, keep]
                cache["xgb_names"] = list(FEAT_COLS_XGB)
            else:
                cache["xgb"] = cache["full"]
                cache["xgb_names"] = [f"f{i}" for i in range(cache["full"].shape[1])]
        return cache["xgb"], cache["xgb_names"]

    if family == "mlp":
        if cache.get("mlp") is None:
            if FEAT_COLS_MLP:
                keep = _indices_from_feat_names(FEAT_COLS_MLP)
                keep = [i for i in keep if 0 <= i < cache["full"].shape[1]]
                cache["mlp"] = cache["full"][:, keep]
                cache["mlp_names"] = list(FEAT_COLS_MLP)
            else:
                cache["mlp"] = cache["full"]
                cache["mlp_names"] = [f"f{i}" for i in range(cache["full"].shape[1])]
        return cache["mlp"], cache["mlp_names"]

    raise ValueError(f"Unknown family '{family}'")

# ─────────── Dataset + base WER helpers ───────────
def load_split(path, cfg, split, key):
    if cfg is None:
         ds = load_dataset(path, split=split, trust_remote_code=True, streaming=True)
    else:
        if "facebook/voxpopuli" in path:
            ds = load_dataset(path, cfg, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(path, cfg, split=split, trust_remote_code=True, streaming=True)
    return ds.cast_column("audio", Audio(sampling_rate=_SR)), key

def wer_pair(ref, wav):
    hyp_w = MODEL_BANK[WHISPER](wav).strip().lower()
    try:
        hyp_p = MODEL_BANK[PARAKEET](wav).strip().lower()
    except Exception:
        raise RuntimeError("Parakeet failed on this clip (likely empty/invalid audio).")
    w_w = jiwer.wer(ref, hyp_w)
    w_p = jiwer.wer(ref, hyp_p)
    return w_w, w_p

# ─────────── Router probabilities (P(Parakeet)) ───────────
def p_parakeet_xgb_from_feats(feat_row, cache=None):
    cache = cache or {}
    x, names = _cached_feats(feat_row, cache, "xgb") if isinstance(feat_row, str) \
               else (feat_row[None, :].astype("float32"), [f"f{i}" for i in range(feat_row.shape[0])])

    # If we got path str, x will be batch 1 already; if vector, we built it above
    if isinstance(feat_row, str):
        f, names = _cached_feats(feat_row, cache, "xgb")
        x = f

    expected_dim = XGB_PACK["models"][0].num_features()
    if x.shape[1] != expected_dim:
        raise ValueError(f"XGB expects {expected_dim} features, got {x.shape[1]}")
    dmat = xgb.DMatrix(x, feature_names=names)

    probs = []
    for booster, it in zip(XGB_PACK["models"], XGB_PACK["best_iters"]):
        p = booster.predict(dmat, iteration_range=(0, int(it) + 1))
        probs.append(p)
    prob = np.mean(probs, axis=0) if len(probs) > 1 else probs[0]
    return float(prob[0, PARAKEET_IDX_XGB])

@torch.inference_mode()
def p_parakeet_mlp_from_feats(feat_row, cache=None):
    cache = cache or {}
    if isinstance(feat_row, str):
        f, _ = _cached_feats(feat_row, cache, "mlp")
    else:
        f = feat_row[None, :].astype("float32")
        if FEAT_COLS_MLP:
            # If FEAT_COLS_MLP is defined, feat_row should already match it; otherwise assume raw.
            pass

    ps = []
    # Determine if sigmoids correspond to Parakeet or Whisper
    # Convention: in training, probability from sigmoid is for class index 1
    # If MLP_LABELS[1] == PARAKEET, p1 is already P(Parakeet); else invert.
    invert = (MLP_LABELS[1] != PARAKEET)
    for mdl, sc in zip(MLP_PACK["models"], MLP_PACK["scalers"]):
        x = f
        if sc is not None:
            x = sc.transform(x).astype("float32")
        logits = mdl(torch.from_numpy(x).to(DEVICE))
        p1 = torch.sigmoid(logits).detach().cpu().numpy().ravel()[0]
        ps.append(1.0 - float(p1) if invert else float(p1))
    return float(np.mean(ps)) if len(ps) > 1 else float(ps[0])

def choose_from_p_parakeet(p, thr=0.5):
    return PARAKEET if p >= thr else WHISPER

def set_ieee():
    cand = ["Times New Roman", "Times", "DejaVu Serif"]
    for f in cand:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

# ─────────── Pool building ───────────
def build_pool(sources, who, max_take, margin, max_probe, max_len):
    """
    sources: list of (path, cfg, split, text_key, pretty)
    who: 'whisper' or 'parakeet' — which model must be strictly better by margin
    return: list of dicts with wav, ref, pretty, wer_w, wer_p
    """
    pool = []
    for path, cfg, split, key, pretty in sources:
        ds, tkey = load_split(path, cfg, split, key)
        cnt = 0
        for ex in ds:
            wav = ex["audio"]["array"]
            if wav is None or len(wav) < MIN_SAMPLES or not np.isfinite(wav).all():
                continue
            if len(wav) / _SR > max_len:
                continue
            ref = (ex.get(tkey, "") or "").strip().lower()
            try:
                w_w, w_p = wer_pair(ref, wav)
            except RuntimeError:
                continue
            if who == "whisper" and (w_w + margin < w_p):
                pool.append(dict(wav=wav, ref=ref, pretty=pretty, wer_w=w_w, wer_p=w_p))
            elif who == "parakeet" and (w_p + margin < w_w):
                pool.append(dict(wav=wav, ref=ref, pretty=pretty, wer_w=w_w, wer_p=w_p))
            cnt += 1
            if cnt >= max_probe or len(pool) >= max_take:
                break
        if len(pool) >= max_take:
            break
    return pool

# ─────────── Plotting ───────────
def plot_switch(df, out_pdf, thr=0.5):
    set_ieee()
    fig, ax = plt.subplots(figsize=(10.5, 3.8))

    x = df["idx"].to_numpy()
    ax.plot(x, df["p_parakeet_xgb"].to_numpy(), label="XGB: P(Parakeet)")
    ax.plot(x, df["p_parakeet_mlp"].to_numpy(), label="MLP: P(Parakeet)", linestyle="--")

    # threshold
    ax.axhline(thr, linewidth=1.0)

    # shaded blocks by segment
    seg = df["segment"].to_numpy()
    start = 0
    for i in range(1, len(seg)+1):
        if i == len(seg) or seg[i] != seg[i-1]:
            color = (0.8, 0.9, 1.0) if seg[start] == "Whisper-good" else (0.9, 1.0, 0.9)
            ax.axvspan(start-0.5, i-0.5, alpha=0.25, color=color, linewidth=0)
            start = i

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(x)-0.5)
    ax.set_xlabel("Clip index in sequence")
    ax.set_ylabel("Probability")
    ax.set_title("Router probabilities over controlled switching sequence")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ─────────── Main ───────────
def run(args):
    # 1) Build guaranteed-good pools
    whisper_good = build_pool(
        WHISPER_SOURCES, "whisper",
        max_take=args.need_whisper, margin=args.margin,
        max_probe=args.max_probe, max_len=args.max_len)
    parakeet_good = build_pool(
        PARAKEET_SOURCES, "parakeet",
        max_take=args.need_parakeet, margin=args.margin,
        max_probe=args.max_probe, max_len=args.max_len)

    if len(whisper_good) < args.need_whisper or len(parakeet_good) < args.need_parakeet:
        print(f"Collected Whisper-good: {len(whisper_good)} / {args.need_whisper}, "
              f"Parakeet-good: {len(parakeet_good)} / {args.need_parakeet}")
        raise SystemExit("Not enough qualifying clips. Lower --margin or raise --max_probe.")

    # 2) Build sequence: [P(block_p) → W(block_w)] × repeats
    wq = deque(whisper_good)
    pq = deque(parakeet_good)
    seq = []
    for _ in range(args.repeats):
        for _ in range(args.block_p):   # Parakeet first
            seq.append(("P", pq[0])); pq.rotate(-1)
        for _ in range(args.block_w):   # Whisper second
            seq.append(("W", wq[0])); wq.rotate(-1)

    # 3) Run routers and log
    rows = []
    for idx, (tag, item) in enumerate(seq):
        # temp wav for feature extractor
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, item["wav"], _SR)
            wav_path = tmp.name

        # features once (full; slices handled in *_from_feats)
        t0 = time.perf_counter()
        feats = extract_features(wav_path).astype("float32")
        t_feat = time.perf_counter() - t0

        # XGB → P(Parakeet)
        t0 = time.perf_counter()
        # Use direct vector form to avoid re-extracting in cache
        dmat = xgb.DMatrix  # just to avoid linter removal; we use vector path below
        p_x = p_parakeet_xgb_from_feats(feats)
        t_sel_x = time.perf_counter() - t0
        pick_x = choose_from_p_parakeet(p_x, thr=args.viz_threshold)

        # MLP → P(Parakeet)
        t0 = time.perf_counter()
        p_m = p_parakeet_mlp_from_feats(feats)
        t_sel_m = time.perf_counter() - t0
        pick_m = choose_from_p_parakeet(p_m, thr=args.viz_threshold)

        os.unlink(wav_path)

        rows.append(dict(
            idx=idx,
            segment=("Whisper-good" if tag=="W" else "Parakeet-good"),
            src=item["pretty"],
            ref=item["ref"],
            p_parakeet_xgb=p_x,
            p_parakeet_mlp=p_m,
            pick_xgb=pick_x,
            pick_mlp=pick_m,
            feat_time_s=t_feat,
            sel_time_xgb_s=t_sel_x,
            sel_time_mlp_s=t_sel_m,
            wer_whisper=item["wer_w"],
            wer_parakeet=item["wer_p"],
        ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv}")

    # 4) Plot figure
    plot_switch(df, args.out_fig, thr=args.viz_threshold)
    print(f"✓ wrote {args.out_fig}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_w", type=int, default=5,  help="Whisper-good clips per block")
    ap.add_argument("--block_p", type=int, default=50, help="Parakeet-good clips per block")
    ap.add_argument("--repeats", type=int, default=4, help="How many [P→W] blocks")
    ap.add_argument("--margin", type=float, default=0.03, help="Absolute WER margin to be 'good for sure'")
    ap.add_argument("--max_probe", type=int, default=400, help="Max candidates to scan per source")
    ap.add_argument("--max_len", type=float, default=30.0, help="Max clip length (sec)")
    ap.add_argument("--need_whisper",  type=int, default=100, help="Pool size target for Whisper-good")
    ap.add_argument("--need_parakeet", type=int, default=400, help="Pool size target for Parakeet-good")
    ap.add_argument("--viz_threshold", type=float, default=0.5, help="Decision/visualization threshold")
    ap.add_argument("--out_csv", type=str, default="switching_log.csv")
    ap.add_argument("--out_fig", type=str, default="figure_switching.pdf")
    args = ap.parse_args()
    run(args)