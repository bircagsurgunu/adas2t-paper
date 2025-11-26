#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate base ASR models + meta-learners (XGB, MLP) across datasets.
Normalization matches dataset creation exactly: ONLY .strip().lower().

What's new:
- XGBoost **CV ensemble**: averages fold probabilities using each fold's best_iteration.
- MLP **CV ensemble**: averages fold probabilities; uses each fold's scaler + threshold (decision via mean threshold).
- Per-model feature slicing using saved feature_names from meta/checkpoints.
- Keeps timing accounting (feature extract, selector, total).

Optional flags:
- --no_oracle : hide Oracle in output JSON (paper mode)
- --no_lgbm   : skip LightGBM router (default)
- --with_lgbm : also evaluate LightGBM router
"""

import argparse, os, tempfile, json, time, glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch, soundfile as sf, jiwer
import xgboost as xgb
import lightgbm as lgb
from datasets import load_dataset, Audio

from asr_runner import MODEL_BANK, MODELS
from feature_extractor_paper import extract_features, AudioFeatureExtractor, _SR
from train_mlp_cls import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _read_first_json(path_list):
    for p in path_list:
        try:
            with open(p) as f:
                return json.load(f), p
        except FileNotFoundError:
            continue
    return None, None

def _indices_from_feat_names(names):
    """['f0','f7',...] -> [0,7,...]"""
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
    if state is None:
        return None
    sc = StandardScaler(**state.get("params", {}))
    sc.mean_ = np.array(state["mean_"], dtype=np.float64)
    sc.scale_ = np.array(state["scale_"], dtype=np.float64)
    sc.var_ = np.array(state["var_"], dtype=np.float64)
    sc.n_features_in_ = int(state["n_features_in_"])
    sc.n_samples_seen_ = int(state["n_samples_seen_"])
    return sc

def _infer_in_dim_from_state_dict(sd) -> int:
    # look for first linear weight in the stack (net.0.weight)
    for k, v in sd.items():
        if k.endswith("net.0.weight"):
            return int(v.shape[1])
    # fallback: any 2D weight ending with 0.weight
    for k, v in sd.items():
        if k.endswith("0.weight") and v.ndim == 2:
            return int(v.shape[1])
    raise RuntimeError("Could not infer MLP input dimension from checkpoint.")

# ──────────────────────────────────────────────────────────────────────────────
# XGB loader (supports CV ensemble + single fallback)
# ──────────────────────────────────────────────────────────────────────────────
XGB_META, XGB_META_PATH = _read_first_json(["adas2t_xgb_meta_cv.json", "adas2t_xgb_meta.json"])
FEAT_COLS_XGB = (XGB_META or {}).get("feature_names")

LABELS_XGB = np.load("label_order_xgb.npy", allow_pickle=True)
IDX_MAP_XGB = [MODELS.index(m) for m in LABELS_XGB]

def _load_xgb_pack():
    """
    Returns dict:
      models: list[xgb.Booster]
      best_iters: list[int]
      is_ensemble: bool
    """
    pack = {"models": [], "best_iters": [], "is_ensemble": False}

    fold_paths = sorted(glob.glob("adas2t_xgb_cls_fold*.json"))
    if fold_paths and XGB_META and "best_iterations" in XGB_META:
        best_iters = list(map(int, XGB_META["best_iterations"]))
        if len(best_iters) != len(fold_paths):
            pad_val = best_iters[-1] if best_iters else 0
            best_iters = (best_iters + [pad_val] * len(fold_paths))[:len(fold_paths)]
        for p in fold_paths:
            b = xgb.Booster()
            b.load_model(p)
            pack["models"].append(b)
        pack["best_iters"] = best_iters
        pack["is_ensemble"] = True
        print(f"[XGB] CV ensemble loaded: {len(pack['models'])} folds.")
        return pack

    # single fallback
    try:
        b = xgb.Booster(model_file="adas2t_xgb_cls.json")
        best_it = int((XGB_META or {}).get("best_iteration", 0))
        pack["models"] = [b]
        pack["best_iters"] = [best_it]
        pack["is_ensemble"] = False
        print(f"[XGB] Single model loaded (best_iteration={best_it}).")
        return pack
    except xgb.core.XGBoostError:
        raise FileNotFoundError("No XGB model found (expected CV folds or single model).")

XGB_PACK = _load_xgb_pack()

# ──────────────────────────────────────────────────────────────────────────────
# MLP loader (supports CV ensemble + single fallback)
# ──────────────────────────────────────────────────────────────────────────────
# Prefer CV meta if present; fallback to single .pth
MLP_META, MLP_META_PATH = _read_first_json(["adas2t_mlp_meta_cv.json"])
FEAT_COLS_MLP = None
LABELS_MLP = None
IDX_MAP_MLP = None

def _load_mlp_pack():
    """
    Returns dict:
      models: list[torch.nn.Module]
      scalers: list[StandardScaler or None]
      thresholds: list[float]
      is_ensemble: bool
    Also sets FEAT_COLS_MLP, LABELS_MLP, IDX_MAP_MLP globals.
    """
    global FEAT_COLS_MLP, LABELS_MLP, IDX_MAP_MLP

    pack = {"models": [], "scalers": [], "thresholds": [], "is_ensemble": False}

    # Try CV ensemble
    fold_paths = sorted(glob.glob("adas2t_mlp_cls_fold*.pth"))
    if fold_paths and MLP_META:
        thr_list = MLP_META.get("thresholds", None)
        if thr_list is None or len(thr_list) != len(fold_paths):
            # tolerate mismatch by padding with mean or 0.5
            fallback_thr = float(np.mean(thr_list)) if thr_list else 0.5
            thr_list = (thr_list or []) + [fallback_thr] * len(fold_paths)
            thr_list = thr_list[:len(fold_paths)]

        classes_set = None
        for p in fold_paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = ckpt["model"]
            in_dim = _infer_in_dim_from_state_dict(sd)
            mdl = MLP(in_dim)
            mdl.load_state_dict(sd, strict=False)
            mdl.eval().to(DEVICE)

            sc = _rebuild_scaler(ckpt.get("scaler_state"))
            pack["models"].append(mdl)
            pack["scalers"].append(sc)
            pack["thresholds"].append(float(ckpt.get("threshold", 0.5)))

            # capture classes / feature_names once
            if FEAT_COLS_MLP is None:
                FEAT_COLS_MLP = ckpt.get("feature_names") or MLP_META.get("feature_names")
            if classes_set is None:
                classes_set = list(ckpt["classes"])

        assert classes_set is not None, "MLP CV ckpt missing 'classes'"
        LABELS_MLP = classes_set
        IDX_MAP_MLP = [MODELS.index(m) for m in LABELS_MLP]
        pack["is_ensemble"] = True
        print(f"[MLP] CV ensemble loaded: {len(pack['models'])} folds.")
        return pack

    # Single fallback
    try:
        ckpt = torch.load("adas2t_mlp_cls.pth", map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        in_dim = _infer_in_dim_from_state_dict(sd)
        mdl = MLP(in_dim)
        mdl.load_state_dict(sd, strict=False)
        mdl.eval().to(DEVICE)
        sc = _rebuild_scaler(ckpt.get("scaler_state"))
        pack["models"] = [mdl]
        pack["scalers"] = [sc]
        pack["thresholds"] = [float(ckpt.get("threshold", 0.5))]
        FEAT_COLS_MLP = ckpt.get("feature_names")
        classes_set = list(ckpt["classes"])
        LABELS_MLP = classes_set
        IDX_MAP_MLP = [MODELS.index(m) for m in LABELS_MLP]
        pack["is_ensemble"] = False
        print("[MLP] Single model loaded.")
        return pack
    except FileNotFoundError:
        raise FileNotFoundError("No MLP model found (expected CV folds or single .pth).")

MLP_PACK = _load_mlp_pack()

# ──────────────────────────────────────────────────────────────────────────────
# Feature caching/slicing
# ──────────────────────────────────────────────────────────────────────────────
def _cached_feats(path, cache, family: str):
    """
    Cache once per file:
      - 'full': raw features from extract_features (1 x D)
      - 'xgb'/'mlp': sliced to saved feature_names if provided
    """
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

# ──────────────────────────────────────────────────────────────────────────────
# Choosers
# ──────────────────────────────────────────────────────────────────────────────
def choose_xgb(path: str, cache, *_):
    f, names = _cached_feats(path, cache, "xgb")
    expected_dim = XGB_PACK["models"][0].num_features()
    if f.shape[1] != expected_dim:
        raise ValueError(f"XGB expects {expected_dim} features, got {f.shape[1]}")
    dmat = xgb.DMatrix(f, feature_names=names)

    t0 = time.perf_counter()
    probs = []
    for booster, it in zip(XGB_PACK["models"], XGB_PACK["best_iters"]):
        p = booster.predict(dmat, iteration_range=(0, int(it) + 1))
        probs.append(p)
    prob = np.mean(probs, axis=0) if len(probs) > 1 else probs[0]
    sel_time = time.perf_counter() - t0

    cls = int(prob[0].argmax())
    return IDX_MAP_XGB[cls], sel_time

@torch.inference_mode()
def choose_mlp(path: str, cache, *_):
    f, _ = _cached_feats(path, cache, "mlp")
    t0 = time.perf_counter()

    # Ensemble: each fold has its own scaler
    ps = []
    for mdl, sc in zip(MLP_PACK["models"], MLP_PACK["scalers"]):
        x = f
        if sc is not None:
            x = sc.transform(x).astype("float32")
        logits = mdl(torch.from_numpy(x).to(DEVICE))
        p1 = torch.sigmoid(logits).detach().cpu().numpy().ravel()[0]
        ps.append(p1)
    p_mean = float(np.mean(ps)) if len(ps) > 1 else float(ps[0])

    sel_time = time.perf_counter() - t0
    thr_mean = float(np.mean(MLP_PACK["thresholds"])) if len(MLP_PACK["thresholds"]) > 1 else float(MLP_PACK["thresholds"][0])
    cls_bin = 1 if p_mean >= thr_mean else 0
    return IDX_MAP_MLP[cls_bin], sel_time

def choose_lgb(path: str, cache, *_):
    lgb_pk = _load_lgbm()
    lgb_cls, LABELS_LGB, IDX_MAP_LGB, THR_LGB = lgb_pk
    f, _ = _cached_feats(path, cache, "xgb")  # LGB trained on same feature space as XGB in your setup
    t0 = time.perf_counter()
    p1 = float(lgb_cls.predict(f))
    sel_time = time.perf_counter() - t0
    cls = 1 if p1 >= THR_LGB else 0
    return IDX_MAP_LGB[cls], sel_time

# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────
def _example_is_valid(ex, ref_key):
    ref = (ex.get(ref_key, "") or "").strip().lower()
    if not ref:
        return False, None, None
    wav = ex["audio"]["array"]
    dur = len(wav) / _SR if wav is not None else 0.0
    if dur <= 0.0 or dur > 30.0:
        return False, None, None
    return True, ref, wav

def _collect_valid_examples(ds_iter, ref_key, target_n, tag):
    out = []
    it = iter(ds_iter)
    seen = 0
    while len(out) < target_n:
        try:
            ex = next(it)
        except StopIteration:
            if len(out) < target_n:
                print(f"[WARN] {tag}: exhausted after {seen} items; "
                      f"collected {len(out)}/{target_n} valid examples.")
            break
        seen += 1
        ok, ref, wav = _example_is_valid(ex, ref_key)
        if not ok:
            continue
        out.append({"audio": ex["audio"], "text": ref})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (with timing)
# ──────────────────────────────────────────────────────────────────────────────
def eval_dataset(name, ds_iter, ref_key, max_clips):
    refs_raw, hypos_raw = defaultdict(list), defaultdict(list)
    refs_xgb, hyp_xgb = [], []
    refs_mlp, hyp_mlp = [], []
    refs_lgb, hyp_lgb = [], []

    time_base_sum = {m: 0.0 for m in MODELS}
    time_base_cnt = {m: 0   for m in MODELS}
    time_meta = {
        "ADAS2T-XGB":  {"feat": 0.0, "select": 0.0, "total": 0.0, "count": 0},
        "ADAS2T-MLP":  {"feat": 0.0, "select": 0.0, "total": 0.0, "count": 0},
    }

    it = iter(ds_iter)
    n_collected, n_seen = 0, 0

    while n_collected < max_clips:
        try:
            ex = next(it)
        except StopIteration:
            if n_collected < max_clips:
                print(f"[WARN] {name}: exhausted dataset after {n_seen} items; "
                      f"collected {n_collected}/{max_clips} valid clips.")
            break

        n_seen += 1
        ok, ref, wav = _example_is_valid(ex, ref_key)
        if not ok:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, _SR)
            wav_path = tmp.name

        feats_cache = {"full": None, "feat_time": 0.0}

        # base models (once per clip)
        clip_hypos, clip_times = [], {}
        for mdl in MODELS:
            fn = MODEL_BANK[mdl]
            t0 = time.perf_counter()
            hyp = fn(wav).strip().lower()
            dt = time.perf_counter() - t0
            clip_hypos.append(hyp)
            clip_times[mdl] = dt
            time_base_sum[mdl] += dt
            time_base_cnt[mdl] += 1
            refs_raw[mdl].append(ref)
            hypos_raw[mdl].append(hyp)

        # XGB
        idx, sel_t = choose_xgb(wav_path, feats_cache, None, None)
        sel_model = MODELS[idx]
        total_t = feats_cache["feat_time"] + sel_t + clip_times[sel_model]
        refs_xgb.append(ref); hyp_xgb.append(clip_hypos[idx])
        tm = time_meta["ADAS2T-XGB"]
        tm["feat"] += feats_cache["feat_time"]; tm["select"] += sel_t; tm["total"] += total_t; tm["count"] += 1

        # MLP
        idx, sel_t = choose_mlp(wav_path, feats_cache, None, None)
        sel_model = MODELS[idx]
        total_t = feats_cache["feat_time"] + sel_t + clip_times[sel_model]
        refs_mlp.append(ref); hyp_mlp.append(clip_hypos[idx])
        tm = time_meta["ADAS2T-MLP"]
        tm["feat"] += feats_cache["feat_time"]; tm["select"] += sel_t; tm["total"] += total_t; tm["count"] += 1

        # Optional LGBM — uncomment if needed and files exist
        # lgb_pk = _load_lgbm()
        # idx, sel_t = choose_lgb(wav_path, feats_cache, None, None)
        # sel_model = MODELS[idx]
        # total_t = feats_cache["feat_time"] + sel_t + clip_times[sel_model]
        # refs_lgb.append(ref); hyp_lgb.append(clip_hypos[idx])

        os.unlink(wav_path)
        n_collected += 1

    # WERs
    scores = {m: jiwer.wer(refs_raw[m], hypos_raw[m]) for m in MODELS}
    scores["ADAS2T-XGB"] = jiwer.wer(refs_xgb, hyp_xgb)
    scores["ADAS2T-MLP"] = jiwer.wer(refs_mlp, hyp_mlp)

    df = (pd.Series(scores).sort_values().rename("WER").to_frame())
    df.index.name = f"{name} (N={len(refs_xgb)})"

    # timing averages
    base_avg = {m: (time_base_sum[m] / max(1, time_base_cnt[m])) for m in MODELS}
    meta_avg = {}
    for k, v in time_meta.items():
        c = max(1, v["count"])
        meta_avg[k] = {
            "feature_extraction_sec": v["feat"]   / c,
            "selector_sec":           v["select"] / c,
            "total_sec":              v["total"]  / c,
        }

    timing_summary = {"base_models_avg_sec": base_avg, "meta_avg_sec": meta_avg}
    return df, timing_summary

# ──────────────────────────────────────────────────────────────────────────────
# LGBM (optional helper)
# ──────────────────────────────────────────────────────────────────────────────
def _load_lgbm():
    lgb_cls  = lgb.Booster(model_file="adas2t_lgbm_cls.txt")
    LABELS_LGB = np.load("label_order_lgb.npy", allow_pickle=True)
    IDX_MAP_LGB = [MODELS.index(m) for m in LABELS_LGB]
    try:
        with open("adas2t_lgbm_meta.json") as f:
            LGB_META = json.load(f)
            THR_LGB = float(LGB_META.get("threshold", 0.5))
    except FileNotFoundError:
        THR_LGB = 0.5
    return lgb_cls, LABELS_LGB, IDX_MAP_LGB, THR_LGB

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_clips", type=int, default=500)
    ap.add_argument("--out_json", type=str, default="adas2t_eval_results.json")
    ap.add_argument("--no_oracle", action="store_true")
    ap.add_argument("--with_lgbm", action="store_true")
    ap.add_argument("--no_lgbm", action="store_true")
    args = ap.parse_args()

    DATASETS = [
        dict(
            name="AMI Meeting IHM",
            iter=load_dataset("edinburghcstr/ami", "ihm", split="test",
                              streaming=True, trust_remote_code=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="text"),
        dict(
            name="AMI Meeting SDM",
            iter=load_dataset("edinburghcstr/ami", "sdm", split="test",
                              streaming=True, trust_remote_code=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="text"),
        dict(
            name="LibriSpeech Test-Other",
            iter=load_dataset("openslr/librispeech_asr",
                              split="test.other", streaming=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="text"),
        dict(
            name="LibriSpeech Test-Clean",
            iter=load_dataset("openslr/librispeech_asr",
                              split="test.clean", streaming=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="text"),
        dict(
            name="Fleurs",
            iter=load_dataset("google/fleurs", "en_us", split="test",
                              streaming=True, trust_remote_code=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="transcription"),
        dict(
            name="Mozilla Common Voice",
            iter=load_dataset("mozilla-foundation/common_voice_16_1", "en",
                              split="test", streaming=True, trust_remote_code=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="sentence"),
        dict(
            name="VoxPopuli",
            iter=load_dataset("facebook/voxpopuli", "en_accented", split="test[:1000]")
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="raw_text"),
        dict(
            name="GigaSpeech",
            iter=load_dataset("speechcolab/gigaspeech", "test", split="test",
                              streaming=True, trust_remote_code=True)
                 .cast_column("audio", Audio(sampling_rate=_SR)),
            key="text"),
    ]

    # Deterministic Mixed Dataset with exact per-set quotas
    mixed_data = []
    num_sets = len(DATASETS)
    base_quota = args.max_clips // max(1, num_sets)
    remainder = args.max_clips - base_quota * num_sets
    for i, d in enumerate(DATASETS):
        target = base_quota + (1 if i < remainder else 0)
        if target == 0:
            continue
        mixed_data.extend(_collect_valid_examples(d["iter"], d["key"], target, f"Mixed/{d['name']}"))
    if len(mixed_data) > args.max_clips:
        mixed_data = mixed_data[:args.max_clips]
    DATASETS.append(dict(name="Mixed Dataset", iter=mixed_data, key="text"))

    # Optional LGBM
    use_lgbm = (args.with_lgbm and not args.no_lgbm)
    if use_lgbm:
        try:
            _ = _load_lgbm()
            print("[LGBM] Loaded.")
        except Exception as e:
            print(f"[LGBM] Skipping (load error: {e}).")
            use_lgbm = False

    # Evaluate
    all_results, all_timings = {}, {}
    for d in DATASETS:
        df, timing_summary = eval_dataset(
            d["name"], d["iter"], d["key"], args.max_clips
        )

        dataset_name = df.index.name
        print("=" * 60)
        print(f"Results for: {dataset_name}")
        print(df.to_string(float_format=lambda x: f"{x:.3%}"))

        tm = timing_summary
        print("\nAvg inference time per base model (ms):")
        for m, s in tm["base_models_avg_sec"].items():
            print(f"  {m:30s} {s*1000:.1f}")
        print("Avg meta routing times (ms):")
        for k, v in tm["meta_avg_sec"].items():
            print(f"  {k:13s}  feat {v['feature_extraction_sec']*1000:.1f}  "
                  f"select {v['selector_sec']*1000:.1f}  total {v['total_sec']*1000:.1f}")
        print()

        res_dict = df["WER"].to_dict()
        if isinstance(next(iter(res_dict.keys())), tuple):
            res_dict = {k[-1]: v for k, v in res_dict.items()}
        all_results[dataset_name] = res_dict
        all_timings[dataset_name] = timing_summary

    with open(args.out_json, "w") as f:
        json.dump({"results": all_results, "timings": all_timings}, f, indent=2)

    print(f"\nAll results saved to {args.out_json}")
