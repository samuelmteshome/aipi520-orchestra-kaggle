"""Trial the attendance-consistency features (cons_*) and three heuristic interactions (hx_*) in the pruned
logistic regression and pruned random forest, on the same 25 folds as src/experiments.py.

To keep the baseline identical to the earlier run, per-fold feature selection still sees only the original
features; the new columns are appended after pruning (forced in, not selected).

  variants per model:  base | + hx (3 interactions) | + cons (8 consistency) | + both
  blends:              rank-average of logistic + forest for each variant

    python3 src/experiments_v2.py        (~5-10 min) -> results/experiments_v2.json, submissions/v2_*.csv
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
NEW = ("cons_", "hx_")


def main():
    (RESULTS / "experiments.log").write_text("")
    best = json.loads((RESULTS / "cv_summary.json").read_text())["best_params"]
    train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    test = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
    y = train.pop("label")
    new_cols = [c for c in train.columns if c.startswith(NEW)]
    hx = [c for c in new_cols if c.startswith("hx_")]
    cons = [c for c in new_cols if c.startswith("cons_")]
    folds = Folds(train.drop(columns=new_cols), y, n_repeats=5)      # selection never sees the new columns
    yv, hard = y.to_numpy(), train["sub_last_season"].to_numpy() == 0
    t0 = time.time()

    extras = {"base": [], "+hx": hx, "+cons": cons, "+both": hx + cons}
    models = {"logistic": lambda: Logistic(**best["logistic"]), "forest": lambda: Forest(**best["forest"])}
    oofs = {}
    for mname, build in models.items():
        for vname, extra in extras.items():
            key = f"{mname} pruned {vname}"
            log(f"running {key}")
            oof = np.full((5, len(y)), np.nan)
            for i, (tr, va) in enumerate(folds.splits):
                Xtr, ytr, Xva, _ = folds.data(i)
                keep = [c for c in Xtr.columns if not PRUNE(c)]
                Xtr = Xtr[keep].join(train.iloc[tr][extra])
                Xva = Xva[keep].join(train.iloc[va][extra])
                oof[i // 5, va] = build().fit(Xtr, ytr).predict_proba(Xva)
            oofs[key] = oof
    for vname in extras:
        oofs[f"L+F pruned {vname}"] = (_rank(oofs[f"logistic pruned {vname}"]) + _rank(oofs[f"forest pruned {vname}"])) / 2

    old = np.load(RESULTS / "experiments_oof.npz")
    log(f"baseline check: logistic max |Δ| vs earlier run {np.nanmax(np.abs(oofs['logistic pruned base'] - old['logistic pruned (a5)'])):.2e}, "
        f"forest {np.nanmax(np.abs(oofs['forest pruned base'] - old['forest pruned (a5)'])):.2e}")
    log("paired comparisons (Δ = variant - base, same 25 folds)")
    rows = [compare(f"{m} pruned {v}", oofs[f"{m} pruned {v}"], f"{m} pruned base", oofs[f"{m} pruned base"], folds, yv, hard, "v2")
            for m in ["logistic", "forest", "L+F"] for v in ["+hx", "+cons", "+both"]]

    # test predictions for every variant (refit on all training rows)
    ff = FoldFeatures(use_id=True, seed=0).fit(train.drop(columns=new_cols), y)
    keep = [c for c in ff.columns_ if not PRUNE(c)]
    Xtr_b, Xte_b = ff.transform(train.drop(columns=new_cols))[keep], ff.transform(test.drop(columns=new_cols))[keep]
    preds = {}
    for vname, extra in extras.items():
        p = {m: b().fit(Xtr_b.join(train[extra]), y).predict_proba(Xte_b.join(test[extra])) for m, b in models.items()}
        preds[vname] = p
        blend = (_rank(p["logistic"]) + _rank(p["forest"])) / 2
        slug = vname.replace("+", "plus_")
        pd.DataFrame({"ID": test.index, "Predicted": blend}).to_csv(SUBMISSIONS / f"v2_LF_pruned_{slug}.csv", index=False)

    summary = [{"variant": k, "pooled_oof_auroc": float(np.mean([roc_auc_score(yv, x) for x in v])),
                "mean_fold_auroc": float(np.mean(_fold_aucs(v, folds, yv))),
                "hard_group_auroc": float(np.mean([roc_auc_score(yv[hard], x[hard]) for x in v]))} for k, v in oofs.items()]
    (RESULTS / "experiments_v2.json").write_text(json.dumps({"comparisons": rows, "summary": summary, "hx": hx, "cons": cons,
                                                            "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    np.savez_compressed(RESULTS / "experiments_v2_oof.npz", **oofs)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
