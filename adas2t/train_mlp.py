#!/usr/bin/env python
"""
Train an MLP that maps 130‑D audio features → 3‑D vector of predicted WER
(one slot per base ASR model).

Usage:
  python train_mlp.py --epochs 40 --batch 256 --lr 1e-3
"""

import argparse, json, os
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error

CSV = "training_table.csv"
N_MODELS = 3                 # whisper, voxtral, canary (order matters!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1.  Load and prepare tensors ─────────────────────────────────────────────
df = pd.read_csv(CSV)

# feature_cols = [c for c in df.columns if c.startswith("f")]
# X = df[feature_cols].values.astype(np.float32)

# pivot = df.pivot_table(index="uid", columns="model", values="wer").fillna(1.0)
# y = pivot.values.astype(np.float32)[df["uid"].drop_duplicates().index]

feature_cols = [c for c in df.columns if c.startswith("f")]

# one row per clip (uid)  –  keep order identical to pivot
pivot = df.pivot_table(index="uid", columns="model", values="wer").fillna(1.0)
y = pivot.values.astype(np.float32)           # same row count as X
X = (
    df.groupby("uid").first()[feature_cols]   # collapse duplicates
      .loc[pivot.index]                       # align to pivot row order
      .values
      .astype(np.float32)
)


assert X.shape[0] == y.shape[0], "mismatched rows"

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

# train/val split 80/20
val_ratio = 0.2
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# ─── 2.  MLP definition ───────────────────────────────────────────────────────
class WERMLP(nn.Module):
    def __init__(self, in_dim=130, hidden=256, out_dim=N_MODELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim)      # regression outputs (WER hat)
        )
    def forward(self, x):
        return self.net(x)

# ─── 3.  Training loop ───────────────────────────────────────────────────────
def train(args):
    model = WERMLP().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()                   # MAE on WER

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

        # ---- validation every epoch ----
        model.eval()
        with torch.no_grad():
            preds, targets = [], []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds.append(model(xb).cpu())
                targets.append(yb)
            preds   = torch.cat(preds).numpy()
            targets = torch.cat(targets).numpy()
            val_mae = mean_absolute_error(targets, preds)

            best_val = min(best_val, val_mae)
            print(f"Epoch {epoch:02d}/{args.epochs}  •  val‑MAE = {val_mae:.4f}")

    torch.save(model.state_dict(), "adas2t_mlp.pth")
    print(f"\nBest val‑MAE={best_val:.4f}   ·   model saved → adas2t_mlp.pth")

    # ---- hit‑ratio on validation set ----
    hits = (preds.argmin(axis=1) == targets.argmin(axis=1)).mean()
    print(f"Hit‑ratio on val = {hits:.2%}")

# ─── 4.  CLI args  ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=1e-3)
    args = ap.parse_args()
    train(args)
