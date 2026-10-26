"""Segment modelling for the hard group (user idea, 2026-10-26), on the same 25 folds as earlier experiments.

Pre-registered before running (no tuning on these folds):
  * Segment R = accounts that subscribed in 2013-14; segment H = everyone else (6,596 accounts, 50 positives).
  * R is ranked by the current pruned logistic + forest blend (trained on all training rows) and placed above all of H.
  * H is ranked by a hard-group model trained on H training rows only: rank-average of an L2 logistic regression
    (C = 0.1) and a shallow random forest (depth 4, min leaf 20, 40% features, balanced_subsample).
  * New features (geo2_*, hg_*, cons_*) are appended after per-fold selection/pruning; selection never sees them.

Variants, all compared fold by fold with the pruned L+F blend (headline = AUROC within segment H):
  V1  pruned L+F + new features          (no segmentation)
  V2  segmented, hard model on the base pruned features
  V3  segmented, hard model on base pruned + new features

    python3 src/experiments_v3.py      (~5-10 min) -> results/experiments_v3.json, submissions/v3_*.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Logistic, Forest, FoldFeatures  # noqa: E402
from train import Folds, _rank, _fold_aucs  # noqa: E402
from experiments import PRUNE, compare, log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, SUBMISSIONS = ROOT / "results", ROOT / "submissions"
NEW = ("cons_", "hx_", "geo2_", "hg_")
HARD_LOGISTIC = {"penalty": "l2", "C": 0.1, "class_weight": None, "interactions": False}
HARD_FOREST = {"max_depth": 4, "min_samples_leaf": 20, "max_features": 0.4, "class_weight": "balanced_subsample"}


def blend(models, Xtr, ytr, Xva):
    """Rank-average of the given models' predictions on Xva."""
    return np.mean([_rank(m.fit(Xtr, ytr).predict_proba(Xva)) for m in models], axis=0)


def combine(recent, base_score, hard_score):
    """Segment R above segment H; within each segment, order by that segment's model."""
    return np.where(recent, 1 + _rank(base_score), _rank(hard_score))


def main():
    (RESULTS / "experiments.log").write_text("")
    best = json.loads((RESULTS / "cv_summary.json").read_text())["best_params"]
    train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    test = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
    y = train.pop("label")
    new_cols = [c for c in train.columns if c.startswith(NEW)]
    extra = [c for c in new_cols if not c.startswith("hx_")]       # hx_ interactions already shown not to help
    folds = Folds(train.drop(columns=new_cols), y, n_repeats=5)
    yv = y.to_numpy()
    recent = train["sub_last_season"].to_numpy() == 1
    hard = ~recent
    base_models = lambda: [Logistic(**best["logistic"]), Forest(**best["forest"])]
    hard_models = lambda: [Logistic(**HARD_LOGISTIC), Forest(**HARD_FOREST)]
    t0 = time.time()

    n = len(y)
    oof = {k: np.full((5, n), np.nan) for k in ["base", "V1", "base_for_R", "H_base", "H_new"]}
    for i, (tr, va) in enumerate(folds.splits):
        r = i // 5
        Xtr, ytr, Xva, _ = folds.data(i)
        keep = [c for c in Xtr.columns if not PRUNE(c)]
        Xtr, Xva = Xtr[keep], Xva[keep]
        Ntr, Nva = Xtr.join(train.iloc[tr][extra]), Xva.join(train.iloc[va][extra])
        oof["base"][r, va] = blend(base_models(), Xtr, ytr, Xva)
        oof["V1"][r, va] = blend(base_models(), Ntr, ytr, Nva)
        htr, hva = hard[tr], hard[va]
        p = np.full(len(va), np.nan)
        p[hva] = blend(hard_models(), Xtr[htr], ytr[htr], Xva[hva])
        oof["H_base"][r, va] = p
        p = np.full(len(va), np.nan)
        p[hva] = blend(hard_models(), Ntr[htr], ytr[htr], Nva[hva])
        oof["H_new"][r, va] = p
        log(f"fold {i + 1}/25 done")

    # per-repeat scores (ranks are taken within each repeat so folds are pooled consistently)
    S = {"pruned L+F (base)": np.vstack([_rank(x) for x in oof["base"]]),
         "V1 pruned L+F + new features": np.vstack([_rank(x) for x in oof["V1"]]),
         "V2 segmented (base features)": np.vstack([combine(recent, oof["base"][k], np.nan_to_num(oof["H_base"][k])) for k in range(5)]),
         "V3 segmented + new features": np.vstack([combine(recent, oof["base"][k], np.nan_to_num(oof["H_new"][k])) for k in range(5)])}
    rows = [compare(k, v, "pruned L+F (base)", S["pruned L+F (base)"], folds, yv, hard, "v3") for k, v in S.items() if k != "pruned L+F (base)"]
    summary = [{"variant": k, "pooled_oof_auroc": float(np.mean([roc_auc_score(yv, x) for x in v])),
                "mean_fold_auroc": float(np.mean(_fold_aucs(v, folds, yv))),
                "hard_group_auroc": float(np.mean([roc_auc_score(yv[hard], x[hard]) for x in v]))} for k, v in S.items()]
    for s in summary:
        log(f"  {s['variant']:32s} pooled {s['pooled_oof_auroc']:.4f}  hard group {s['hard_group_auroc']:.4f}")

    # test predictions (refit on all training rows)
    ff = FoldFeatures(use_id=True, seed=0).fit(train.drop(columns=new_cols), y)
    keep = [c for c in ff.columns_ if not PRUNE(c)]
    Xtr, Xte = ff.transform(train.drop(columns=new_cols))[keep], ff.transform(test.drop(columns=new_cols))[keep]
    Ntr, Nte = Xtr.join(train[extra]), Xte.join(test[extra])
    rte = test["sub_last_season"].to_numpy() == 1
    base_te = blend(base_models(), Xtr, y, Xte)
    h_new = np.zeros(len(test))
    h_new[~rte] = blend(hard_models(), Ntr[hard], y[hard], Nte[~rte])
    h_base = np.zeros(len(test))
    h_base[~rte] = blend(hard_models(), Xtr[hard], y[hard], Xte[~rte])
    out = {"v3_V1_LF_plus_new": _rank(blend(base_models(), Ntr, y, Nte)),
           "v3_V2_segmented_base": combine(rte, base_te, h_base) / 2,
           "v3_V3_segmented_new": combine(rte, base_te, h_new) / 2}
    for name, p in out.items():
        pd.DataFrame({"ID": test.index, "Predicted": p}).to_csv(SUBMISSIONS / f"{name}.csv", index=False)

    (RESULTS / "experiments_v3.json").write_text(json.dumps(
        {"comparisons": rows, "summary": summary, "extra_features": extra,
         "hard_models": {"logistic": HARD_LOGISTIC, "forest": HARD_FOREST}, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    np.savez_compressed(RESULTS / "experiments_v3_oof.npz", **S)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
