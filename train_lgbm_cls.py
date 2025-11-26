#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM binary classifier
- Excludes ties: only UIDs with a unique best model
- Per-sample class weights, early stopping on logloss, logs AUC
- Optional GPU
- Calibrated threshold (Youden J), saves label order, feature names, meta
"""

import argparse, json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, roc_curve)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="training_table_no_whisper.csv")
    p.add_argument("--labels", default="clip_labels.csv")
    p.add_argument("--model_out", default="adas2t_lgbm_cls.txt")
    p.add_argument("--labels_out", default="label_order_lgb.npy")
    p.add_argument("--meta_out", default="adas2t_lgbm_meta.json")
    p.add_argument("--imp_out", default="adas2t_lgbm_feat_importance_gain.csv")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--num_boost_round", type=int, default=4000)
    p.add_argument("--early_stopping_rounds", type=int, default=200)
    return p.parse_args()

def youden_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def main():
    args = get_args()

    # 1) Load & align (exclude ties via intersection)
    df = pd.read_csv(args.table)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_df = df.groupby("uid").first()[feat_cols]

    lab_series = pd.read_csv(args.labels, index_col=0).iloc[:, 0]
    common_uids = X_df.index.intersection(lab_series.index)
    X_df = X_df.loc[common_uids]
    labels = lab_series.loc[common_uids].values

    le = LabelEncoder().fit(labels)
    y = le.transform(labels).astype(np.int32)
    X = X_df.values.astype(np.float32)

    print("Rows (after removing ties):", len(X_df))
    print("Label order:", le.classes_)
    print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    # 2) Split + class weights
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    cls_w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    w_tr = cls_w[y_tr]
    w_val = cls_w[y_val]

    # 3) Datasets
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols)
    dvl = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtr, feature_name=feat_cols)

    # 4) Params
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.02,
        num_leaves=255,
        max_depth=8,
        min_data_in_leaf=50,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=1.0,
        verbose=-1,
        seed=args.seed,
        num_threads=8,
    )
    if args.device == "gpu":
        params.update(dict(device_type="gpu", gpu_platform_id=0, gpu_device_id=0))

    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result),
    ]

    # 5) Train
    gbm = lgb.train(
        params,
        dtr,
        num_boost_round=args.num_boost_round,
        valid_sets=[dvl],
        valid_names=["val"],
        callbacks=callbacks,
    )
    print(f"Best iteration: {gbm.best_iteration}")

    # 6) Threshold + metrics
    proba_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    thr = youden_threshold(y_val, proba_val)
    y_hat = (proba_val >= thr).astype(np.int32)
    acc = accuracy_score(y_val, y_hat)
    bacc = balanced_accuracy_score(y_val, y_hat)

    print("\n====== Validation metrics ======")
    print(f"Accuracy            : {acc:.2%}")
    print(f"Balanced accuracy   : {bacc:.2%}")
    print(f"Chosen threshold    : {thr:.4f}\n")
    print("Classification report:")
    print(classification_report(y_val, y_hat, target_names=le.classes_.astype(str)))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_hat))

    # 7) Save artifacts
    gbm.save_model(args.model_out, num_iteration=gbm.best_iteration)
    np.save(args.labels_out, le.classes_)
    meta = dict(
        label_order=le.classes_.tolist(),
        feature_names=feat_cols,
        best_iteration=int(gbm.best_iteration),
        threshold=thr,
        params=params,
    )
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    # 8) Feature importances (gain)
    imp = pd.Series(gbm.feature_importance(importance_type="gain"), index=feat_cols, name="gain")
    imp.sort_values(ascending=False).to_csv(args.imp_out)

    print(f"\nSaved model → {args.model_out}")
    print(f"Saved labels → {args.labels_out}")
    print(f"Saved meta   → {args.meta_out}")
    print(f"Saved feature importances → {args.imp_out}")

if __name__ == "__main__":
    main()
