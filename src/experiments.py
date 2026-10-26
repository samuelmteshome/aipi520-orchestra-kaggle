"""Generalisation-strategy experiments (actions 3-7), compared with the current models on the same 25 folds.

Same seed as src/train.py, so the folds and each fold's 25 selected features are identical. Every variant is
scored against its baseline fold by fold (paired comparison), plus pooled out-of-fold AUROC per repeat.

  baseline     logistic / forest / MLP ensemble exactly as in train.py (re-run to get per-fold predictions)
  action 3     MLP ensemble: early stopping picks the epoch count, then refit on 100% of the fold's rows
  action 4     ensembles from saved predictions: L+F+M (current), L+F only, forest-heavier (0.5 F, 0.25 L, 0.25 M)
  action 5     pruning: drop ticket + interaction features and 3 of the 4 distance features from each fold's set
  action 6     stronger regularisation: forest depth 3; forest leaf 10; logistic C = 10; logistic C = 1
  action 7     MLP 32-16 architecture with 10x weight decay (1e-2) vs its tuned 1e-3

    python3 src/experiments.py           (~60 min)  -> results/experiments.json + submissions/exp_*.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Logistic, Forest, MLP, MLPEnsemble, FoldFeatures  # noqa: E402
from train import Folds, mlp_params, _rank, _fold_aucs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, SUBMISSIONS = ROOT / "results", ROOT / "submissions"
SEEDS = (0, 1, 2)
PRUNE = lambda c: c.startswith(("tix_", "ix_")) or c in {"geo_km_sf", "geo_km_berkeley", "geo_km_peninsula"}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "experiments.log", "a") as f:
        f.write(line + "\n")


def members(best, **overrides):
    return [{**mlp_params({**c, "hidden": tuple(c["hidden"])}), **overrides} for c in best["mlp_members"]]


def run(folds, build, prune=False):
    """Out-of-fold predictions, shape (5 repeats, n accounts)."""
    oof = np.full((5, len(folds.y)), np.nan)
    for i, (_, va) in enumerate(folds.splits):
        Xtr, ytr, Xva, _ = folds.data(i)
        if prune:
            keep = [c for c in Xtr.columns if not PRUNE(c)]
            Xtr, Xva = Xtr[keep], Xva[keep]
        oof[i // 5, va] = build().fit(Xtr, ytr).predict_proba(Xva)
    return oof


def compare(name, oof, base_name, base_oof, folds, y, hard, action):
    fa, fb = np.array(_fold_aucs(oof, folds, y)), np.array(_fold_aucs(base_oof, folds, y))
    pooled = lambda o, m=slice(None): float(np.mean([roc_auc_score(y[m], r[m]) for r in o]))
    r = {"variant": name, "baseline": base_name, "action": action,
         "mean_fold_auroc": float(fa.mean()), "pooled_oof_auroc": pooled(oof), "hard_group_auroc": pooled(oof, hard),
         "delta_mean_fold": float((fa - fb).mean()), "delta_sd": float((fa - fb).std()),
         "folds_better": int((fa > fb).sum()), "delta_pooled": pooled(oof) - pooled(base_oof),
         "delta_hard": pooled(oof, hard) - pooled(base_oof, hard)}
    log(f"  {name:34s} vs {base_name:14s} Δfold {r['delta_mean_fold']:+.4f} (±{r['delta_sd']:.4f}, better in "
        f"{r['folds_better']}/25) Δpooled {r['delta_pooled']:+.4f} Δhard {r['delta_hard']:+.4f}")
    return r


def main():
    (RESULTS / "experiments.log").write_text("")
    best = json.loads((RESULTS / "cv_summary.json").read_text())["best_params"]
    train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    test = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
    y = train.pop("label")
    folds = Folds(train, y, n_repeats=5)
    yv, hard = y.to_numpy(), train["sub_last_season"].to_numpy() == 0
    t0 = time.time()

    V = {   # variant -> (builder, prune?)
        "logistic": (lambda: Logistic(**best["logistic"]), False),
        "forest": (lambda: Forest(**best["forest"]), False),
        "mlp_ensemble": (lambda: MLPEnsemble(members(best), SEEDS), False),
        "mlp_ensemble refit 100% (a3)": (lambda: MLPEnsemble(members(best), SEEDS, refit_full=True), False),
        "logistic pruned (a5)": (lambda: Logistic(**best["logistic"]), True),
        "forest pruned (a5)": (lambda: Forest(**best["forest"]), True),
        "mlp_ensemble pruned (a5)": (lambda: MLPEnsemble(members(best), SEEDS), True),
        "forest depth 3 (a6)": (lambda: Forest(**{**best["forest"], "max_depth": 3}), False),
        "forest leaf 10 (a6)": (lambda: Forest(**{**best["forest"], "min_samples_leaf": 10}), False),
        "logistic C=10 (a6)": (lambda: Logistic(**{**best["logistic"], "C": 10.0}), False),
        "logistic C=1 (a6)": (lambda: Logistic(**{**best["logistic"], "C": 1.0}), False),
        "mlp 32-16 (tuned wd 1e-3)": (lambda: MLPEnsemble(members(best)[:1], SEEDS), False),
        "mlp 32-16 wd 1e-2 (a7)": (lambda: MLPEnsemble([{**members(best)[0], "weight_decay": 1e-2}], SEEDS), False),
    }
    oofs = {}
    for name, (build, prune) in V.items():
        log(f"running {name}")
        oofs[name] = run(folds, build, prune)
    log(f"baseline check: logistic pooled {np.mean([roc_auc_score(yv, r) for r in oofs['logistic']]):.4f} "
        f"(train.py run: see cv_summary.json)")

    # action 4: ensembles from the stored predictions (rank-averaged per repeat)
    rk = {k: _rank(v) for k, v in oofs.items()}
    oofs["average L+F+M (current)"] = (rk["logistic"] + rk["forest"] + rk["mlp_ensemble"]) / 3
    oofs["average L+F (a4)"] = (rk["logistic"] + rk["forest"]) / 2
    oofs["forest-heavier 0.5F (a4)"] = 0.5 * rk["forest"] + 0.25 * rk["logistic"] + 0.25 * rk["mlp_ensemble"]

    cmp = [("mlp_ensemble refit 100% (a3)", "mlp_ensemble", 3),
           ("average L+F (a4)", "average L+F+M (current)", 4), ("forest-heavier 0.5F (a4)", "average L+F+M (current)", 4),
           ("logistic pruned (a5)", "logistic", 5), ("forest pruned (a5)", "forest", 5), ("mlp_ensemble pruned (a5)", "mlp_ensemble", 5),
           ("forest depth 3 (a6)", "forest", 6), ("forest leaf 10 (a6)", "forest", 6),
           ("logistic C=10 (a6)", "logistic", 6), ("logistic C=1 (a6)", "logistic", 6),
           ("mlp 32-16 wd 1e-2 (a7)", "mlp 32-16 (tuned wd 1e-3)", 7)]
    log("paired comparisons (Δ = variant - baseline, same 25 folds)")
    rows = [compare(v, oofs[v], b, oofs[b], folds, yv, hard, a) for v, b, a in cmp]

    # test-set predictions for every variant whose model can be refit directly
    log("refitting variants on all training rows for test predictions")
    ff = FoldFeatures(use_id=True, seed=0).fit(train, y)
    Xtr, Xte = ff.transform(train), ff.transform(test)
    keep = [c for c in Xtr.columns if not PRUNE(c)]
    preds = {}
    for name, (build, prune) in V.items():
        cols = keep if prune else list(Xtr.columns)
        preds[name] = build().fit(Xtr[cols], y).predict_proba(Xte[cols])
    r = {k: _rank(v) for k, v in preds.items()}
    preds["average L+F (a4)"] = (r["logistic"] + r["forest"]) / 2
    preds["forest-heavier 0.5F (a4)"] = 0.5 * r["forest"] + 0.25 * r["logistic"] + 0.25 * r["mlp_ensemble"]
    for name, p in preds.items():
        slug = "exp_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
        pd.DataFrame({"ID": test.index, "Predicted": p}).to_csv(SUBMISSIONS / f"{slug}.csv", index=False)

    summary = [{"variant": k, "pooled_oof_auroc": float(np.mean([roc_auc_score(yv, x) for x in v])),
                "mean_fold_auroc": float(np.mean(_fold_aucs(v, folds, yv))),
                "hard_group_auroc": float(np.mean([roc_auc_score(yv[hard], x[hard]) for x in v]))} for k, v in oofs.items()]
    (RESULTS / "experiments.json").write_text(json.dumps(
        {"comparisons": rows, "summary": summary, "pruned_features": "tix_*, ix_*, geo_km_sf, geo_km_berkeley, geo_km_peninsula",
         "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    np.savez_compressed(RESULTS / "experiments_oof.npz", **{k: v for k, v in oofs.items()})
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
