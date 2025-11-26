#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lean-MLP (binary) — 5-fold CV
- Excludes ties: only UIDs with a unique best model
- StandardScaler saved as scaler_state (no pickled sklearn object)
- Class-weighted BCE (pos_weight = n_neg / n_pos), NO sampler
- Early stopping on balanced accuracy + ReduceLROnPlateau
- AMP (torch.amp), grad clipping
- Calibrated threshold (Youden J) saved to each fold checkpoint
- NEW: --drop_groups ablation using paper-aligned groups
- Saves: per-fold .pth, OOF CSV, CV metrics JSON, meta JSON
"""

import argparse, random, json, os
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, roc_curve)

from feature_extractor_paper import AudioFeatureExtractor, _SR

# ──────────────────────────────────────────────────────────────────────────────
# Repro
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# Subgroups / Paper groups
# ──────────────────────────────────────────────────────────────────────────────
_SUBGROUP_ORDER = [
    "mfcc",
    "prosodic",
    "activity",
    "channel",
    "noise_modulation",
    "signal",
    # "neural_whisper",
    "neural_w2v2",
    "neural_hubert",
    # "embedding_stats_whisper",
    "embedding_stats_w2v2",
    "embedding_stats_hubert",
    "metadata",
]

_PAPER_GROUPS = {
    "mfcc": ["mfcc"],
    "prosodic_vad": ["prosodic", "activity"],
    "neural_embeddings": ["neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata"],
    "signal_temporal": ["signal", "channel", "noise_modulation"],

    "mfcc_and_prosodic_vad": ["mfcc", "prosodic", "activity"],
    "mfcc_and_neural_embeddings": ["mfcc", "neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata"],
    "mfcc_and_signal_temporal": ["mfcc", "signal", "channel", "noise_modulation"],
    "prosodic_vad_and_neural_embeddings": ["prosodic", "activity", "neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata"],
    "prosodic_vad_and_signal_temporal": ["prosodic", "activity", "signal", "channel", "noise_modulation"],
    "neural_embeddings_and_signal_temporal": ["neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata", "signal", "channel", "noise_modulation"],

    
    "mfcc_and_prosodic_vad_and_neural_embeddings": ["mfcc", "prosodic", "activity", "neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata"],
    "mfcc_and_prosodic_vad_and_signal_temporal": ["mfcc", "prosodic", "activity", "signal", "channel", "noise_modulation"],
    "mfcc_and_neural_embeddings_and_signal_temporal": ["mfcc", "neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata", "signal", "channel", "noise_modulation"],
    "prosodic_vad_and_neural_embeddings_and_signal_temporal": ["prosodic", "activity", "neural_w2v2", "neural_hubert", "embedding_stats_w2v2", "embedding_stats_hubert", "metadata", "signal", "channel", "noise_modulation"],
    # NOTE: embedding_stats_* and metadata are intentionally left out per paper groups
}

def _build_subgroup_slices(sr=_SR, device="cpu"):
    afe = AudioFeatureExtractor(sample_rate=sr, device=device)
    dummy = np.zeros(sr, dtype=np.float32)
    parts = afe.extract_all_features(dummy)
    slices, off = {}, 0
    for name in _SUBGROUP_ORDER:
        n = int(parts[name].shape[0])
        slices[name] = slice(off, off + n)
        off += n
    return slices, off

def _expand_paper_groups(names):
    out = []
    for n in names:
        n = n.strip().lower()
        if not n:
            continue
        out.extend(_PAPER_GROUPS.get(n, []))
    return sorted(set(out))

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    # Data / outputs
    p.add_argument("--table",    type=str,   default="training_table_25k_dropped_voxpopuli.csv")
    p.add_argument("--labels",   type=str,   default="clip_labels_25k_dropped_voxpopuli.csv")
    p.add_argument("--out",      type=str,   default="adas2t_mlp_cls_fold{fold}.pth",
                   help="Per-fold checkpoint path pattern (use {fold})")
    p.add_argument("--meta_out", type=str,   default="adas2t_mlp_meta_cv.json")
    p.add_argument("--oof_out",  type=str,   default="adas2t_mlp_oof.csv")
    p.add_argument("--metrics_out", type=str, default="adas2t_mlp_cv_metrics.json")

    # Training
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=1024)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--patience", type=int,   default=20)
    p.add_argument("--cv_folds", type=int,   default=5)
    p.add_argument("--device",   choices=["cpu","cuda"], default="cuda")
    p.add_argument("--seed",     type=int,   default=SEED)

    # Ablation (paper groups)
    # p.add_argument("--drop_groups", type=str, default="",
    #                help="Comma-separated paper groups to DROP: mfcc, prosodic_vad, neural_embeddings, signal_temporal")
    p.add_argument("--drop_groups", type=str, default="",
                   help="Comma-separated paper groups to DROP: whisper, w2v2, hubert, whisper_w2v2, whisper_hubert, w2v2_hubert, whisper_w2v2_hubert")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): 
        return self.net(x).squeeze(1)

@torch.no_grad()
def eval_balanced_acc(model, loader, device, threshold=0.5):
    model.eval()
    logits, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits.append(model(xb).detach().cpu())
        gts.append(yb)
    logits = torch.cat(logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    yhat = (probs >= threshold).astype(np.int64)
    y_ref = torch.cat(gts).numpy()
    bal = balanced_accuracy_score(y_ref, yhat)
    acc = accuracy_score(y_ref, yhat)
    return bal, acc, probs, y_ref

def find_best_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def scaler_state_dict(sc: StandardScaler, n_samples_seen: int):
    return dict(
        type="StandardScaler",
        params=sc.get_params(),
        mean_=sc.mean_.astype(np.float32),
        scale_=sc.scale_.astype(np.float32),
        var_=sc.var_.astype(np.float32),
        n_features_in_=int(sc.n_features_in_),
        n_samples_seen_=int(n_samples_seen),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main CV training
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    # 1) Load features + labels; exclude ties via intersection
    df = pd.read_csv(args.table)  # uid, model, wer, f*
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_df = df.groupby("uid").first()[feat_cols]  # one row per uid

    lab_series = pd.read_csv(args.labels, index_col=0).iloc[:, 0]  # uid -> best_model
    common_uids = X_df.index.intersection(lab_series.index)
    X_df = X_df.loc[common_uids]
    labels = lab_series.loc[common_uids].values
    uids = X_df.index.to_numpy()

    le = LabelEncoder().fit(labels)
    y_all = le.transform(labels).astype(np.int64)

    # === compute subgroup slices and build drop list ===
    subgroup_slices, feat_dim = _build_subgroup_slices(sr=_SR, device=args.device)
    col_index = {i: f"f{i}" for i in range(len(feat_cols))}

    def _cols_for_subgroup(name):
        sl = subgroup_slices[name]
        return [col_index[i] for i in range(sl.start, sl.stop)]

    drop_groups = [s.strip() for s in args.drop_groups.split(",") if s.strip()]
    dropped_subgroups = _expand_paper_groups(drop_groups)
    drop_cols = []
    for sg in dropped_subgroups:
        drop_cols.extend(_cols_for_subgroup(sg))

    if drop_cols:
        keep_cols = [c for c in feat_cols if c not in set(drop_cols)]
        X_df = X_df[keep_cols]
        feat_cols = keep_cols

    X_all = X_df.values.astype(np.float32)
    n_samples, n_feats = X_all.shape

    print("Rows (after removing ties):", n_samples)
    print("Label order:", le.classes_)
    cls_dist = dict(zip(*np.unique(y_all, return_counts=True)))
    print("Class distribution:", cls_dist)
    if drop_cols:
        print(f"Dropping groups: {drop_groups}")
        print(f"Expanded subgroups: {sorted(dropped_subgroups)}")
        print(f"Removed {len(drop_cols)} feature columns")

    # 2) CV setup
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # Storage for OOF + fold metrics
    oof_prob = np.zeros(n_samples, dtype=np.float32)
    oof_pred = np.zeros(n_samples, dtype=np.int64)
    oof_true = y_all.copy()

    folds_meta = []
    fold_reports = []  # optional text summaries

    # 3) Train per fold
    fold_idx = 0
    for tr_idx, va_idx in skf.split(X_all, y_all):
        fold_idx += 1
        print(f"\n===== Fold {fold_idx}/{args.cv_folds} =====")
        # Fold-local seed (stability with shuffling)
        torch.manual_seed(args.seed + fold_idx)
        np.random.seed(args.seed + fold_idx)
        random.seed(args.seed + fold_idx)

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        uid_va = uids[va_idx]

        # Scale fit only on training
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        tr_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
        va_ds = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va))

        def make_loader(ds, shuffle):
            return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                              drop_last=False, num_workers=2, pin_memory=True)

        tl = make_loader(tr_ds, shuffle=True)
        vl = make_loader(va_ds, shuffle=False)

        # Model / opt
        model = MLP(n_feats).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

        n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
        pos_weight = torch.tensor(n_neg / max(1, n_pos), dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scaler_amp = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

        best_bal, patience_left, best_state = 0.0, args.patience, None
        best_epoch, best_thr = 0, 0.5

        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in tl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, dtype=torch.float32, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logit = model(xb)
                    loss = loss_fn(logit, yb)
                scaler_amp.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(opt)
                scaler_amp.update()

            # Eval on val with threshold=0.5 (for scheduler/early stop)
            bal, acc, probs, y_ref = eval_balanced_acc(model, vl, device, threshold=0.5)
            scheduler.step(bal)
            print(f"[Fold {fold_idx}] Ep{ep:03d}  bal-acc {bal:.3%} | acc {acc:.3%} | lr {opt.param_groups[0]['lr']:.1e}")

            if bal > best_bal + 1e-4:
                best_bal, patience_left = bal, args.patience
                # Recompute per-fold calibrated threshold on current model
                thr = find_best_threshold(y_ref, probs)
                best_epoch, best_thr = ep, thr
                # Save fold checkpoint payload (no pickled sklearn objects)
                best_state = {
                    "model": model.state_dict(),
                    "scaler_state": scaler_state_dict(scaler, n_samples_seen=X_tr_s.shape[0]),
                    "classes": le.classes_,
                    "threshold": float(thr),
                    "epoch": int(ep),
                    "dropped_groups": drop_groups,
                    "dropped_subgroups": sorted(dropped_subgroups),
                    "dropped_feature_indices": [int(c[1:]) for c in drop_cols],
                    "feature_names": feat_cols,
                    "fold": int(fold_idx),
                }
                torch.save(best_state, args.out.format(fold=fold_idx))
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"[Fold {fold_idx}] Early stop at epoch {ep}. Best bal-acc {best_bal:.3%}")
                    break

        # Load best fold checkpoint to compute OOF for this fold
        ckpt_path = args.out.format(fold=fold_idx)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

        # Re-eval at calibrated threshold
        vl_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                               drop_last=False, num_workers=2, pin_memory=False)
        bal, acc, probs, y_ref = eval_balanced_acc(model, vl_loader, device, threshold=ckpt["threshold"])
        y_hat = (probs >= ckpt["threshold"]).astype(np.int64)

        # Fill OOF slots
        oof_prob[va_idx] = probs
        oof_pred[va_idx] = y_hat

        # Fold report
        print(f"\n[Fold {fold_idx}] Best epoch: {ckpt['epoch']}  Thr: {ckpt['threshold']:.4f}")
        print(f"[Fold {fold_idx}] Val bal-acc: {bal:.3%} | Val acc: {acc:.3%}")
        cm = confusion_matrix(y_ref, y_hat)
        print(f"[Fold {fold_idx}] Confusion matrix:\n{cm}")
        print(f"[Fold {fold_idx}] Classification report:\n"
              f"{classification_report(y_ref, y_hat, target_names=le.classes_)}")

        folds_meta.append({
            "fold": fold_idx,
            "best_epoch": int(ckpt["epoch"]),
            "threshold": float(ckpt["threshold"]),
            "val_balanced_accuracy": float(bal),
            "val_accuracy": float(acc),
            "pos_weight": float(pos_weight.item()),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
        })

        # optional keep a terse text report too
        fold_reports.append({
            "fold": fold_idx,
            "cm": cm.tolist(),
        })

    # 4) OOF summary
    oof_acc = accuracy_score(oof_true, oof_pred)
    oof_bal = balanced_accuracy_score(oof_true, oof_pred)
    print(f"\nOOF accuracy          : {oof_acc:.4%}")
    print(f"OOF balanced accuracy : {oof_bal:.4%}")

    # 5) Save OOF CSV (uid, y_true, y_pred, p1)
    oof_df = pd.DataFrame({
        "uid": uids,
        "y_true": oof_true,
        "y_pred": oof_pred,
        "p1": oof_prob,
    })
    oof_df.to_csv(args.oof_out, index=False)
    print(f"Saved OOF → {args.oof_out}")

    # 6) Save CV metrics
    metrics = {
        "folds": folds_meta,
        "oof": {
            "accuracy": float(oof_acc),
            "balanced_accuracy": float(oof_bal),
        }
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved CV metrics → {args.metrics_out}")

    # 7) Save meta (for evaluator to ensemble)
    meta = dict(
        label_order=list(map(str, le.classes_)),
        feature_names=feat_cols,
        dropped_groups=drop_groups,
        dropped_subgroups=sorted(dropped_subgroups),
        dropped_feature_indices=[int(c[1:]) for c in drop_cols],
        cv_folds=args.cv_folds,
        best_epochs=[m["best_epoch"] for m in folds_meta],
        thresholds=[m["threshold"] for m in folds_meta],
        seed=args.seed,
    )
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta → {args.meta_out}")

if __name__ == "__main__":
    # Slight QoL: ensure output dir exists if user put folders in patterns
    args = get_args()
    # Create parent dirs for out patterns if needed (first fold path)
    out_path_fold1 = args.out.format(fold=1)
    os.makedirs(os.path.dirname(out_path_fold1) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.oof_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)

    # Re-run main with parsed args preserved
    # (We reparse inside main for simplicity; safe here.)
    main()
