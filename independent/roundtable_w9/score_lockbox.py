"""Operator scorer for the W9 ensemble roundtable: prints one JSON line of metrics for a competitor's OUT_DIR."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
y = pd.read_csv(ROOT / "for_students/train.csv").set_index("account.id")["label"]
recent = set(pd.read_csv(ROOT / "for_students/subscriptions.csv").query("season == '2013-2014'")["account.id"])
res = {}
try:
    o = pd.read_csv(out / "oof_final.csv").set_index("account.id")["pred"]
    j = y.to_frame().join(o, how="inner")
    res["dev_oof_auroc"] = round(float(roc_auc_score(j["label"], j["pred"])), 5)
except Exception as e:
    res["dev_oof_error"] = str(e)[:200]
try:
    s = pd.read_csv(out / "submission.csv").set_index("ID")["Predicted"]
    j = y.to_frame().join(s.rename("p"), how="inner")
    yv, pv = j["label"].to_numpy(), j["p"].to_numpy()
    rng = np.random.default_rng(0)
    b = [roc_auc_score(yv[i], pv[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000))]
    h = ~j.index.isin(recent)
    res.update({"lockbox_auroc": round(float(roc_auc_score(yv, pv)), 5),
                "lockbox_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                "lockbox_hard_group_auroc": round(float(roc_auc_score(yv[h], pv[h])), 4), "lockbox_n": int(len(yv))})
except Exception as e:
    res["lockbox_error"] = str(e)[:200]
print(json.dumps(res))
