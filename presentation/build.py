"""Build the pre-modelling review deck: presentation/data_review.html

    python3 presentation/build.py     (run src/features.py first)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clean import load_subscriptions, load_tickets  # noqa: E402
from features import FEATURE_GROUPS  # noqa: E402

X = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
T = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
y = X["label"]
R = {"base": round(y.mean() * 100, 1), "n": len(X), "pos": int(y.sum()), "n_feat": T.shape[1]}


def rate(col, bins=None, labels=None, order=None, data=X):
    s = data[col] if bins is None else pd.cut(data[col], bins, labels=labels, right=True)
    g = data.groupby(s, observed=False)["label"].agg(["mean", "size"])
    if order is not None:
        g = g.reindex(order)
    return {"x": [str(i) for i in g.index], "rate": (g["mean"] * 100).round(1).fillna(0).tolist(), "n": g["size"].fillna(0).astype(int).tolist()}


# ---- heuristics
R["h_recency"] = rate("sub_seasons_since_last", [0, 1, 2, 3, 5, 10, 21, 99],
                      ["1 (last season)", "2", "3", "4–5", "6–10", "11–21", "never"])
R["h_nseasons"] = rate("sub_n_seasons", [-1, 0, 1, 2, 3, 5, 10, 15, 21], ["0", "1", "2", "3", "4–5", "6–10", "11–15", "16–21"])
R["h_any"] = rate("any_n_seasons", [-1, 0, 1, 2, 3, 5, 10, 15, 21], ["0", "1", "2", "3", "4–5", "6–10", "11–15", "16–21"])
R["h_don"] = rate("don_lifetime", [-1, 0, 99, 499, 999, 4999, 24999, 1e9], ["$0", "$1–99", "$100–499", "$500–999", "$1k–5k", "$5k–25k", "$25k+"])
R["h_doncount"] = rate("don_count", [-1, 0, 1, 2, 5, 10, 20, 100], ["0", "1", "2", "3–5", "6–10", "11–20", "21+"])
R["h_tix"] = rate("tix_seasons_since_last", [0, 1, 2, 3, 4, 99], ["1 (2013-14)", "2", "3", "4", "never"])
loc_order = ["San Francisco", "Berkeley Saturday", "Berkeley Sunday", "Peninsula", "Contra Costa", "Santa Rosa", "Orange County", "none"]
R["h_loc"] = rate("sub_last_location", order=loc_order)
subs = load_subscriptions()
v = pd.crosstab(subs["location"], subs["season"]).reindex(loc_order[:-1])
R["venues"] = {"x": v.columns.str[:4].tolist(), "y": v.index.tolist(), "z": v.values.tolist()}
recent = X[X["sub_last_season"] == 1]
R["h_multi"] = rate("sub_last_multiple", order=[0, 1], data=recent)
R["h_2014"] = rate("don_first_in_2014", order=[0, 1])
R["h_streak"] = rate("sub_streak", [0, 1, 2, 3, 5, 10, 21], ["1", "2", "3", "4–5", "6–10", "11–21"], data=recent)

# ---- ID cohort pattern (user decision: include in the model)
cohort = np.select([X.index.str[:12] == "001i000000Lh", X.index.str[:12] == "001i000000Nu", X.index > "001i000000Nv"],
                   ["Lh", "Nu", "newer than Nu"], "older/other")
idd = X.assign(cohort=cohort, recent=np.where(X["sub_last_season"] == 1, "subscribed 2013-14", "did not"))
gi = idd.groupby(["recent", "cohort"])["label"].agg(["mean", "size"]).reset_index()
R["id_cohort"] = {r: {"x": d["cohort"].tolist(), "rate": (d["mean"] * 100).round(1).tolist(), "n": d["size"].tolist()}
                  for r, d in gi.groupby("recent")}

# ---- 25-feature selection preview
sel = json.loads((ROOT / "data/processed/selection_preview.json").read_text())
R["sel"] = [{"feature": c, "rf": sel["rf_importance"].get(c), "mi": sel["mi"][c], "missing": round(sel["missing"][c] * 100, 1),
             "stable": f'{sel["fold_stability"].get(c, 0)}/5'} for c in sel["chosen"]]
R["sel_dropped_missing"] = sel["dropped_missing"]
R["sel_dropped_dup"] = [f"{a} ≈ {b}" for a, b in sel["dropped_duplicate"].items()]

# ---- skew / signal table
group_of = {}
for g, p in FEATURE_GROUPS.items():
    for c in T.columns:
        if c.startswith(p):
            group_of[c] = g
rows = []
for c in T.select_dtypes("number").columns:
    s = X[c]
    m = s.notna()
    auc = roc_auc_score(y[m], s[m]) if s[m].nunique() > 1 else np.nan
    lg = np.log1p(s[m].clip(lower=0)) if (s[m] >= 0).all() else None
    rows.append({"feature": c, "group": group_of.get(c, ""), "skew": round(float(skew(s[m])), 2),
                 "skew_log": round(float(skew(lg)), 2) if lg is not None else None,
                 "missing": round(float(s.isna().mean() * 100), 1),
                 "auc": round(float(max(auc, 1 - auc)), 3) if not np.isnan(auc) else None,
                 "dir": "↑" if auc >= 0.5 else "↓",
                 "test_mean_ratio": round(float(T[c].mean() / s.mean()), 2) if s.mean() else None})
tab = pd.DataFrame(rows).sort_values("auc", ascending=False)
R["table"] = tab.to_dict("records")
top = tab.head(20)
R["auc_top"] = {"x": top["feature"].tolist(), "y": top["auc"].tolist()}
sk = tab.dropna(subset=["skew_log"]).assign(a=lambda d: d["skew"].abs()).sort_values("a", ascending=False).head(12)
R["skew_top"] = {"x": sk["feature"].tolist(), "raw": sk["skew"].tolist(), "log": sk["skew_log"].tolist()}
R["don_hist"] = {"raw": X["don_lifetime"].clip(upper=20000).tolist(), "log": np.log1p(X["don_lifetime"]).round(3).tolist()}

# ---- review findings (filled in from the workflow review, if present)
rf = ROOT / "presentation" / "review_findings.json"
R["findings"] = json.loads(rf.read_text()) if rf.exists() else []

# ---- model results (after src/train.py has run)
cvp = ROOT / "results/cv_summary.json"
if cvp.exists():
    cv = json.loads(cvp.read_text())
    R["results"] = cv["summary"]
    R["best_params"] = cv["best_params"]
    R["run_minutes"] = cv["minutes"]
    tu = pd.read_csv(ROOT / "results/tuning.csv")
    R["tuning"] = {m: {"val": d["val_auroc"].round(4).tolist(), "train": d["train_auroc"].round(4).tolist(),
                       "cfg": d["config"].tolist()} for m, d in tu.groupby("model")}

# ---- feature contributions & over-fitting diagnostics (after src/analysis.py)
anp = ROOT / "results/analysis.json"
if anp.exists():
    an = json.loads(anp.read_text())
    pf = pd.DataFrame(an["permutation_feature"])
    piv = pf.pivot_table(index="feature", columns="model", values="mean")
    piv = piv.loc[piv["average_A_B_C"].sort_values(ascending=False).index[:18]]
    R["imp_feat"] = {"x": piv.index.tolist(), **{m: piv[m].round(4).tolist() for m in ["forest", "mlp_ensemble", "average_A_B_C"]}}
    gp = pd.DataFrame(an["permutation_group"]).pivot_table(index="group", columns="model", values="mean")
    gp = gp.loc[gp["average_A_B_C"].sort_values(ascending=False).index]
    R["imp_group"] = {"x": gp.index.tolist(), **{m: gp[m].round(4).tolist() for m in ["forest", "mlp_ensemble", "average_A_B_C"]}}
    tv = pd.DataFrame(an["train_vs_val"]).pivot(index="model", columns="variable", values="mean")
    R["gap"] = {"x": tv.index.tolist(), "train": tv["train_auroc"].round(4).tolist(), "val": tv["val_auroc"].round(4).tolist()}
    lc = pd.DataFrame(an["learning_curve"]).pivot(index="frac", columns="model", values="mean")
    R["lc"] = {"x": (lc.index * 100).round().astype(int).tolist(), **{m: lc[m].round(4).tolist() for m in lc.columns}}

# ---- strategy experiments (after src/experiments.py)
exp = ROOT / "results/experiments.json"
if exp.exists():
    ex = json.loads(exp.read_text())
    R["exp_cmp"] = ex["comparisons"]
    R["exp_summary"] = sorted(ex["summary"], key=lambda r: -r["pooled_oof_auroc"])
    R["exp_minutes"] = ex["minutes"]

css = re.search(r"<style>(.*?)</style>", (ROOT / "eda/report_template.html").read_text(), re.S).group(1)
html = (ROOT / "presentation/template.html").read_text().replace("/*__CSS__*/", css).replace("/*__DATA__*/null", json.dumps(R))
(ROOT / "presentation/data_review.html").write_text(html)
print("wrote presentation/data_review.html")
