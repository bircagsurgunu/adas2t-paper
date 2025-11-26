#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost classifier (binary via multi:softprob, num_class inferred)
- Excludes ties: only uses UIDs with a unique best model
- Paper-aligned ablation via --drop_groups
- 5-fold Stratified CV with early stopping
- Class-balanced sample weights computed per-fold
- Saves per-fold models, OOF predictions, metrics, and aggregated feature importances
"""

import argparse, json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# === Feature group utilities (paper-aligned) ===
from feature_extractor_paper import AudioFeatureExtractor, _SR

# Subgroup order must match feature_extractor_paper.extract_all_features() key order
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

# Paper macro groups → extractor subgroups
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
    """Return dict: subgroup -> slice(start, end) in the concatenated feature vector."""
    afe = AudioFeatureExtractor(sample_rate=sr, device=device)
    dummy = np.zeros(sr, dtype=np.float32)  # 1s silence
    parts = afe.extract_all_features(dummy)  # dict name -> np.ndarray
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

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="training_table_25k_dropped_voxpopuli.csv")
    p.add_argument("--labels", default="clip_labels_25k_dropped_voxpopuli.csv")
    p.add_argument("--model_out", default="adas2t_xgb_cls_fold{fold}.json")
    p.add_argument("--labels_out", default="label_order_xgb.npy")
    p.add_argument("--meta_out", default="adas2t_xgb_meta_cv.json")
    p.add_argument("--imp_out", default="adas2t_xgb_feat_importance_gain_cv.csv")
    p.add_argument("--oof_out", default="adas2t_xgb_oof.csv")
    p.add_argument("--metrics_out", default="adas2t_xgb_cv_metrics.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cv_folds", type=int, default=5)
    # p.add_argument("--drop_groups", type=str, default="",
    #                help="Comma-separated paper groups to DROP: mfcc, prosodic_vad, neural_embeddings, signal_temporal")
    p.add_argument("--drop_groups", type=str, default="",
                   help="Comma-separated paper groups to DROP (if using alt mapping): whisper, w2v2, hubert, whisper_w2v2, whisper_hubert, w2v2_hubert, whisper_w2v2_hubert")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()

    # 1) Load features, align to labels (exclude ties)
    df = pd.read_csv(args.table)  # uid, model, wer, f0..fN
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_df = df.groupby("uid").first()[feat_cols]  # one row per uid

    lab_series = pd.read_csv(args.labels, index_col=0).iloc[:, 0]  # uid -> best_model
    common_uids = X_df.index.intersection(lab_series.index)
    X_df = X_df.loc[common_uids]
    labels = lab_series.loc[common_uids].values

    le = LabelEncoder().fit(labels)
    y_all = le.transform(labels).astype(np.int64)
    uid_all = X_df.index.to_numpy()

    # 1a) Compute subgroup slices → column indices
    subgroup_slices, feat_dim = _build_subgroup_slices(sr=_SR, device=args.device)
    col_index = {i: f"f{i}" for i in range(len(feat_cols))}

    def _cols_for_subgroup(name):
        sl = subgroup_slices[name]
        return [col_index[i] for i in range(sl.start, sl.stop)]

    # 1b) Handle ablation (drop paper groups)
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

    print("Rows (after removing ties):", len(X_df))
    print("Label order:", le.classes_)
    if drop_cols:
        print(f"Dropping groups: {drop_groups}")
        print(f"Expanded subgroups: {sorted(dropped_subgroups)}")
        print(f"Removed {len(drop_cols)} feature columns")

    # 2) CV setup
    n_classes = len(le.classes_)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # Storage
    oof_proba = np.zeros((len(X_all), n_classes), dtype=np.float32)
    oof_pred = np.zeros(len(X_all), dtype=np.int64)
    oof_true = y_all.copy()
    oof_uid = uid_all.copy()

    fold_metrics = []
    fold_best_iters = []
    agg_gain = pd.Series(0.0, index=pd.Index(feat_cols, name="feature"))

    # 3) Train per fold
    params = dict(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric=["mlogloss", "merror"],
        tree_method="hist",
        device=args.device,
        max_depth=8,
        min_child_weight=4,
        eta=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_bin=256,
        sampling_method="gradient_based",
        seed=args.seed,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        # Class-balanced weights computed on TRAIN for this fold
        cls_w = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        # Map weights by class id (assumes classes are 0..C-1 after LabelEncoder)
        w_tr = cls_w[y_tr]
        w_va = cls_w[y_va]  # apply same mapping to val for fair eval

        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feat_cols)
        dva = xgb.DMatrix(X_va, label=y_va, weight=w_va, feature_names=feat_cols)

        es = xgb.callback.EarlyStopping(
            rounds=200, save_best=True, maximize=False, data_name="val", metric_name="mlogloss"
        )

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=4000,
            evals=[(dva, "val")],
            callbacks=[es],
            verbose_eval=100
        )

        best_it = int(bst.best_iteration)
        fold_best_iters.append(best_it)

        # Save fold model
        model_path = args.model_out.format(fold=fold)
        bst.save_model(model_path)
        print(f"[Fold {fold}] Best iter: {best_it}  → saved {model_path}")

        # Predict on validation fold
        proba = bst.predict(dva, iteration_range=(0, best_it + 1))
        pred = proba.argmax(axis=1)

        oof_proba[va_idx] = proba
        oof_pred[va_idx] = pred

        acc = accuracy_score(y_va, pred)
        bal = balanced_accuracy_score(y_va, pred)

        # Accumulate feature importances (gain). Fill missing with 0.
        gain = pd.Series(bst.get_score(importance_type="gain"))
        # XGBoost returns feature keys matching the given feature_names
        gain = gain.reindex(feat_cols).fillna(0.0)
        agg_gain = agg_gain.add(gain, fill_value=0.0)

        # Basic per-fold report
        print(f"[Fold {fold}] Accuracy: {acc:.4%} | Balanced Acc: {bal:.4%}")
        print(f"[Fold {fold}] Confusion matrix:\n{confusion_matrix(y_va, pred)}")
        print(f"[Fold {fold}] Classification report:\n"
              f"{classification_report(y_va, pred, target_names=le.classes_)}")

        fold_metrics.append({
            "fold": fold,
            "best_iteration": best_it,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
        })

    # 4) OOF summary
    oof_acc = accuracy_score(oof_true, oof_pred)
    oof_bal = balanced_accuracy_score(oof_true, oof_pred)
    print(f"\nOOF accuracy          : {oof_acc:.4%}")
    print(f"OOF balanced accuracy : {oof_bal:.4%}")

    # Save label order
    np.save(args.labels_out, le.classes_)

    # Save meta
    meta = dict(
        label_order=le.classes_.tolist(),
        feature_names=feat_cols,
        params=params,
        dropped_groups=drop_groups,
        dropped_subgroups=sorted(dropped_subgroups),
        dropped_feature_indices=[int(c[1:]) for c in drop_cols],
        cv_folds=args.cv_folds,
        best_iterations=fold_best_iters,
    )
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    # Save aggregated feature importances (gain)
    agg_gain.name = "gain_sum"
    agg_gain.sort_values(ascending=False).to_csv(args.imp_out)

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "uid": oof_uid,
        "y_true": oof_true,
        "y_pred": oof_pred,
    })
    # add per-class probs
    for c in range(n_classes):
        oof_df[f"p_{le.classes_[c]}"] = oof_proba[:, c]
    oof_df.to_csv(args.oof_out, index=False)

    # Save metrics
    metrics = {
        "folds": fold_metrics,
        "oof": {
            "accuracy": float(oof_acc),
            "balanced_accuracy": float(oof_bal),
        }
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved labels   → {args.labels_out}")
    print(f"Saved meta     → {args.meta_out}")
    print(f"Saved OOF      → {args.oof_out}")
    print(f"Saved metrics  → {args.metrics_out}")
    print(f"Saved feat imp → {args.imp_out}")
