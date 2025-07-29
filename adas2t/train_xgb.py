# train_xgb.py
import pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CSV = "training_table.csv"
df  = pd.read_csv(CSV)

feature_cols = [c for c in df.columns if c.startswith("f")]
pivot = df.pivot_table(index="uid", columns="model", values="wer")
labels = pivot.values
X = df.groupby("uid").first()[feature_cols].loc[pivot.index].values

X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=0)
dtr, dte = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_te, label=y_te)

params = dict(
    objective="reg:squarederror",
    eval_metric="mae",
    tree_method="hist",
    device="cuda",
    max_depth=9,
    eta=0.03,
    subsample=0.7,
    colsample_bytree=0.8,
)
bst = xgb.train(params, dtr, num_boost_round=600,
                evals=[(dte,"val")], early_stopping_rounds=40, verbose_eval=20)

bst.save_model("adas2t_xgb.json")

# report
pred = bst.predict(dte)
mae = mean_absolute_error(y_te, pred)
hit = (pred.argmin(axis=1) == y_te.argmin(axis=1)).mean()
print(f"MAE={mae:.4f} · Hit‑ratio={hit:.3%}")
