"""Cross-validate, tune and fit every model; write results and Kaggle submission files.

Design ("tune once", approved 2026-10-26):
  1. Folds: repeated stratified 5-fold x 5 repeats, stratified on label x subscribed-in-2013-14.
     Every model uses the same 25 folds, so comparisons are paired.
  2. Inside each fold's training rows: choose 25 features, fit preprocessing, fit the model.
  3. Tune each model by random search on repeat 0 only (5 folds); pick the best mean fold AUROC.
  4. Score the frozen best settings on all 25 folds: mean fold AUROC (lecture metric, W2_H1 s25),
     pooled out-of-fold AUROC per repeat, AUROC on the hard group (did not subscribe in 2013-14),
     and average precision (PR curve, W2_H2 s24).
  5. Ablation: the same frozen models without the two ID-cohort features.
  6. Refit on all training rows and predict test.csv.

    python3 src/train.py            full run (~30-40 min on an M2 CPU)
    python3 src/train.py --quick    smoke test (~1 min)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, RepeatedStratifiedKFold
from scipy.stats import loguniform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (ZeroRule, Heuristic, Logistic, Forest, MLP, MLPEnsemble, FoldFeatures)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"
SEED = 42

SEARCH = {
    "logistic": {"penalty": ["l1", "l2"], "C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"],
                 "interactions": [True, False]},
    "forest": {"max_depth": [4, 6, 10, None], "min_samples_leaf": [2, 5, 10, 20, 40],
               "max_features": ["sqrt", 0.4, 0.7], "class_weight": [None, "balanced"]},
    "mlp": {"hidden": [(32, 16), (64, 32)], "lr": [3e-4, 1e-3, 3e-3], "weight_decay": [1e-4, 1e-3, 1e-2],
            "regulariser": ["dropout0", "dropout0.1", "dropout0.2", "batchnorm"]},
}
N_CONFIGS = {"logistic": 24, "forest": 24, "mlp": 12}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "train.log", "a") as f:
        f.write(line + "\n")


def mlp_params(cfg):
    """Turn the 'regulariser' choice into MLP arguments (batch norm replaces dropout, W8 s77)."""
    cfg = dict(cfg)
    r = cfg.pop("regulariser")
    return {**cfg, "dropout": 0.0 if r == "batchnorm" else float(r.replace("dropout", "")),
            "batch_norm": r == "batchnorm"}


def make(name, cfg, quick=False):
    if name == "logistic":
        return Logistic(**cfg)
    if name == "forest":
        return Forest(**cfg, n_estimators=100 if quick else 500)
    if name == "mlp":
        return MLP(**mlp_params(cfg), max_epochs=5 if quick else 300)
    raise ValueError(name)


class Folds:
    """The 25 shared folds, with the 25 selected features cached per fold (and per ID setting)."""

    def __init__(self, X, y, n_repeats):
        strata = y.astype(int) * 2 + X["sub_last_season"].astype(int)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
        self.splits = list(cv.split(X, strata))
        self.X, self.y = X, y
        self._features = {}

    def features(self, i, use_id=True):
        key = (i, use_id)
        if key not in self._features:
            tr = self.splits[i][0]
            self._features[key] = FoldFeatures(use_id=use_id, seed=i).fit(self.X.iloc[tr], self.y.iloc[tr])
        return self._features[key]

    def data(self, i, use_id=True):
        tr, va = self.splits[i]
        ff = self.features(i, use_id)
        return (ff.transform(self.X.iloc[tr]), self.y.iloc[tr], ff.transform(self.X.iloc[va]), self.y.iloc[va])


def run_cv(folds, build, fold_ids, use_id=True, raw=False, train_score=False):
    """Fit `build()` on each fold; return out-of-fold predictions (per repeat), fold AUROCs, train AUROCs."""
    n = len(folds.y)
    oof = np.full((len(folds.splits) // 5, n), np.nan)
    aucs, train_aucs, models = [], [], []
    for i in fold_ids:
        tr, va = folds.splits[i]
        if raw:   # baselines use the raw features (no selection)
            Xtr, ytr, Xva, yva = folds.X.iloc[tr], folds.y.iloc[tr], folds.X.iloc[va], folds.y.iloc[va]
        else:
            Xtr, ytr, Xva, yva = folds.data(i, use_id)
        m = build().fit(Xtr, ytr)
        p = m.predict_proba(Xva)
        oof[i // 5, va] = p
        aucs.append(roc_auc_score(yva, p))
        if train_score:
            train_aucs.append(roc_auc_score(ytr, m.predict_proba(Xtr)))
        models.append(m)
    return oof, aucs, train_aucs, models


def summarise(name, oof, aucs, y, hard):
    pooled = [roc_auc_score(y, r) for r in oof if not np.isnan(r).any()]
    hard_auc = [roc_auc_score(y[hard], r[hard]) for r in oof if not np.isnan(r).any()]
    ap = [average_precision_score(y, r) for r in oof if not np.isnan(r).any()]
    s = {"model": name, "mean_fold_auroc": float(np.mean(aucs)), "sd_fold_auroc": float(np.std(aucs)),
         "pooled_oof_auroc": float(np.mean(pooled)), "pooled_oof_sd": float(np.std(pooled)),
         "hard_group_auroc": float(np.mean(hard_auc)), "avg_precision": float(np.mean(ap)), "n_folds": len(aucs)}
    log(f"  {name:22s} mean fold AUROC {s['mean_fold_auroc']:.4f} ± {s['sd_fold_auroc']:.4f} | "
        f"pooled OOF {s['pooled_oof_auroc']:.4f} | hard group {s['hard_group_auroc']:.4f} | AP {s['avg_precision']:.3f}")
    return s


def main(quick=False):
    RESULTS.mkdir(exist_ok=True)
    SUBMISSIONS.mkdir(exist_ok=True)
    (RESULTS / "train.log").write_text("")
    train = pd.read_parquet(DATA / "features_train.parquet")
    test = pd.read_parquet(DATA / "features_test.parquet")
    y = train.pop("label")
    folds = Folds(train, y, n_repeats=1 if quick else 5)
    all_folds = list(range(len(folds.splits)))
    tune_folds = all_folds[:5]                         # repeat 0
    yv, hard = y.to_numpy(), train["sub_last_season"].to_numpy() == 0
    n_cfg = {k: 2 for k in N_CONFIGS} if quick else N_CONFIGS
    t0 = time.time()

    # ---- 1. tune on repeat 0 (validation curves come from the train vs validation AUROC here)
    tuning, best = [], {}
    for name in ["logistic", "forest", "mlp"]:
        log(f"tuning {name}: {n_cfg[name]} configs x {len(tune_folds)} folds")
        for j, cfg in enumerate(ParameterSampler(SEARCH[name], n_cfg[name], random_state=SEED)):
            _, aucs, tr_aucs, _ = run_cv(folds, lambda: make(name, cfg, quick), tune_folds, train_score=True)
            row = {"model": name, "config": json.dumps(cfg, default=str), "val_auroc": np.mean(aucs),
                   "val_sd": np.std(aucs), "train_auroc": np.mean(tr_aucs)}
            tuning.append(row)
            log(f"  {name} #{j:02d} val {row['val_auroc']:.4f} train {row['train_auroc']:.4f}  {row['config']}")
        rows = [r for r in tuning if r["model"] == name]
        best[name] = json.loads(max(rows, key=lambda r: r["val_auroc"])["config"])
        if name == "mlp":   # best config per architecture -> the ensemble members
            best["mlp_members"] = []
            for h in SEARCH["mlp"]["hidden"]:
                hr = [r for r in rows if json.loads(r["config"])["hidden"] == list(h)]
                if hr:
                    best["mlp_members"].append(json.loads(max(hr, key=lambda r: r["val_auroc"])["config"]))
        log(f"best {name}: {best.get('mlp_members', best[name]) if name == 'mlp' else best[name]}")
    pd.DataFrame(tuning).to_csv(RESULTS / "tuning.csv", index=False)
    for m in best.get("mlp_members", []):
        m["hidden"] = tuple(m["hidden"])
    seeds = (0,) if quick else (0, 1, 2)

    # ---- 2. score frozen models on all folds
    log(f"scoring frozen models on {len(all_folds)} folds")
    summary, oofs = [], {}
    builders = {
        "zero_rule": (lambda: ZeroRule(), True),
        "heuristic_sub_n_last5": (lambda: Heuristic(), True),
        "logistic": (lambda: make("logistic", best["logistic"], quick), False),
        "forest": (lambda: make("forest", best["forest"], quick), False),
    }
    for name, (build, raw) in builders.items():
        oof, aucs, _, _ = run_cv(folds, build, all_folds, raw=raw)
        oofs[name] = oof
        summary.append(summarise(name, oof, aucs, yv, hard))

    # MLP members: fit once per (architecture, seed); ensembles are averages of the stored predictions
    members = {}
    for k, cfg in enumerate(best["mlp_members"]):
        for s in seeds:
            build = lambda: MLP(**mlp_params(cfg), seed=s, max_epochs=5 if quick else 300)
            members[(k, s)] = run_cv(folds, build, all_folds)[0]
    arch = {k: np.mean([members[(k, s)] for s in seeds], axis=0) for k in range(len(best["mlp_members"]))}
    for k, o in arch.items():
        oofs[f"mlp_arch{k}"] = o
        summary.append(summarise(f"mlp_arch{k} ({'-'.join(map(str, best['mlp_members'][k]['hidden']))})", o,
                                 _fold_aucs(o, folds, yv), yv, hard))
    oofs["mlp_ensemble"] = np.mean(list(arch.values()), axis=0)
    summary.append(summarise("mlp_ensemble", oofs["mlp_ensemble"], _fold_aucs(oofs["mlp_ensemble"], folds, yv), yv, hard))

    # Equal-weight average of the three families (probabilities are rank-averaged so scales match)
    ranks = [_rank(oofs[k]) for k in ["logistic", "forest", "mlp_ensemble"]]
    oofs["average_A_B_C"] = np.mean(ranks, axis=0)
    summary.append(summarise("average_A_B_C", oofs["average_A_B_C"], _fold_aucs(oofs["average_A_B_C"], folds, yv), yv, hard))

    # ---- 3. ablation without the ID-cohort features
    log("ablation: without ID-cohort features")
    for name in ["logistic", "forest"]:
        oof, aucs, _, _ = run_cv(folds, builders[name][0], all_folds, use_id=False)
        summary.append(summarise(f"{name} (no ID)", oof, aucs, yv, hard))
    abl = [run_cv(folds, lambda: MLP(**mlp_params(c), seed=0, max_epochs=5 if quick else 300), all_folds, use_id=False)[0]
           for c in best["mlp_members"]]
    o = np.mean(abl, axis=0)
    summary.append(summarise("mlp_ensemble (no ID, 1 seed)", o, _fold_aucs(o, folds, yv), yv, hard))

    # ---- 4. refit on all training rows, predict test
    log("refitting on all training rows")
    ff = FoldFeatures(use_id=True, seed=0).fit(train, y)
    Xtr, Xte = ff.transform(train), ff.transform(test)
    preds = {
        "logistic": make("logistic", best["logistic"], quick).fit(Xtr, y).predict_proba(Xte),
        "forest": make("forest", best["forest"], quick).fit(Xtr, y).predict_proba(Xte),
        "mlp_ensemble": MLPEnsemble([mlp_params({**c, "hidden": c["hidden"]}) | {"max_epochs": 5 if quick else 300}
                                     for c in best["mlp_members"]], seeds=seeds).fit(Xtr, y).predict_proba(Xte),
    }
    preds["average_A_B_C"] = np.mean([_rank(p) for p in preds.values()], axis=0)
    for name, p in preds.items():
        pd.DataFrame({"ID": test.index, "Predicted": p}).to_csv(SUBMISSIONS / f"{name}.csv", index=False)

    # ---- save
    oof_df = pd.DataFrame({k: np.nanmean(v, axis=0) for k, v in oofs.items()}, index=train.index)
    oof_df["label"] = y.values
    oof_df.to_parquet(RESULTS / "oof_predictions.parquet")
    picks = pd.Series([c for (i, u), f in folds._features.items() if u for c in f.columns_]).value_counts()
    (RESULTS / "cv_summary.json").write_text(json.dumps({
        "summary": summary, "best_params": {k: v for k, v in best.items()},
        "final_features": ff.columns_, "feature_pick_counts": picks.to_dict(),
        "n_folds": len(all_folds), "minutes": round((time.time() - t0) / 60, 1), "quick": quick,
    }, indent=1, default=str))
    log(f"done in {(time.time() - t0) / 60:.1f} min; submissions in {SUBMISSIONS.relative_to(ROOT)}/")


def _rank(p):
    """Rank-transform each repeat (or a 1-D vector) to [0, 1] so different models can be averaged."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return pd.Series(p).rank(pct=True).to_numpy()
    return np.vstack([pd.Series(r).rank(pct=True).to_numpy() for r in p])


def _fold_aucs(oof, folds, y):
    return [roc_auc_score(y[va], oof[i // 5, va]) for i, (_, va) in enumerate(folds.splits)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    main(**vars(ap.parse_args()))
