"""Feature contributions and over-fitting diagnostics for the tuned models.

Uses the 5 folds of repeat 0 (the same folds the models were tuned on) and the best settings saved
by src/train.py in results/cv_summary.json.

1. Permutation importance (model-agnostic): on each validation fold, shuffle one feature and measure
   how much AUROC drops. Done for the random forest, the MLP ensemble, logistic regression and the
   A+B+C average. Because many history features are near-copies of each other, shuffling one alone
   understates it, so whole feature groups (sub_, any_, don_, ...) are also shuffled together.
2. Model-specific views: random-forest impurity importance (W1 s33) and logistic coefficients on
   standardised inputs (direction of effect).
3. Over-fitting checks (W2_H1 s37): train vs validation AUROC per fold, and a learning curve
   (validation AUROC when training on 20-100% of each fold's training rows).

    python3 src/analysis.py        (~10-15 min) -> results/analysis.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Logistic, Forest, MLP, MLPEnsemble  # noqa: E402
from train import Folds, mlp_params, _rank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N_SHUFFLES = 5
GROUPS = {"subscription history": "sub_", "any purchase": "any_", "donations": "don_", "tickets": "tix_",
          "geography": "geo_", "ID cohort": "id_", "interactions": "ix_", "account flags": "has_"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_models(best, Xtr, ytr, seeds=(0, 1, 2)):
    members = [mlp_params({**c, "hidden": tuple(c["hidden"])}) for c in best["mlp_members"]]
    return {
        "logistic": Logistic(**best["logistic"]).fit(Xtr, ytr),
        "forest": Forest(**best["forest"]).fit(Xtr, ytr),
        "mlp_ensemble": MLPEnsemble(members, seeds=seeds).fit(Xtr, ytr),
    }


def scores(models, X, y):
    """AUROC of each model and of the rank-average of the three."""
    p = {k: m.predict_proba(X) for k, m in models.items()}
    out = {k: roc_auc_score(y, v) for k, v in p.items()}
    out["average_A_B_C"] = roc_auc_score(y, np.mean([_rank(v) for v in p.values()], axis=0))
    return out


def permute(X, cols, rng):
    X = X.copy()
    idx = rng.permutation(len(X))
    for c in cols:
        X[c] = X[c].to_numpy()[idx] if not isinstance(X[c].dtype, pd.CategoricalDtype) \
            else pd.Categorical(X[c].to_numpy()[idx], categories=X[c].cat.categories)
    return X


def main():
    cv = json.loads((RESULTS / "cv_summary.json").read_text())
    best = cv["best_params"]
    train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    y = train.pop("label")
    folds = Folds(train, y, n_repeats=1)   # identical to repeat 0 of the 5x5 run (same seed)
    rng = np.random.default_rng(0)
    feat_rows, group_rows, gap_rows, lc_rows, rf_imp, lr_coef = [], [], [], [], [], []

    for i in range(5):
        Xtr, ytr, Xva, yva = folds.data(i)
        log(f"fold {i}: fitting")
        models = fit_models(best, Xtr, ytr)
        base_va, base_tr = scores(models, Xva, yva), scores(models, Xtr, ytr)
        for m in base_va:
            gap_rows.append({"fold": i, "model": m, "train_auroc": base_tr[m], "val_auroc": base_va[m]})

        # 1a. one feature at a time
        for c in Xva.columns:
            drops = [{m: base_va[m] - s for m, s in scores(models, permute(Xva, [c], rng), yva).items()}
                     for _ in range(N_SHUFFLES)]
            for m in base_va:
                feat_rows.append({"fold": i, "feature": c, "model": m, "drop": float(np.mean([d[m] for d in drops]))})
        # 1b. whole groups
        for g, prefix in GROUPS.items():
            cols = [c for c in Xva.columns if c.startswith(prefix)]
            if not cols:
                continue
            drops = [{m: base_va[m] - s for m, s in scores(models, permute(Xva, cols, rng), yva).items()}
                     for _ in range(N_SHUFFLES)]
            for m in base_va:
                group_rows.append({"fold": i, "group": g, "n_features": len(cols), "model": m,
                                   "drop": float(np.mean([d[m] for d in drops]))})

        # 2. model-specific views
        rf = models["forest"].model_
        names = rf.named_steps["prep"].get_feature_names_out()
        rf_imp += [{"fold": i, "feature": n.split("__", 1)[1], "importance": v}
                   for n, v in zip(names, rf.named_steps["rf"].feature_importances_)]
        lr = models["logistic"].model_
        names = lr.named_steps["prep"].named_steps["columns"].get_feature_names_out()
        lr_coef += [{"fold": i, "feature": n.split("__", 1)[1], "coef": v}
                    for n, v in zip(names, lr.named_steps["lr"].coef_[0])]

        # 3. learning curve: train on a stratified fraction of the fold's training rows
        for frac in [0.2, 0.4, 0.6, 0.8]:
            sub = (pd.Series(np.arange(len(ytr)), index=ytr.index)
                   .groupby(ytr.values, group_keys=False).apply(lambda s: s.sample(frac=frac, random_state=i)))
            m_sub = fit_models(best, Xtr.iloc[sub.values], ytr.iloc[sub.values], seeds=(0,))
            for m, s in scores(m_sub, Xva, yva).items():
                lc_rows.append({"fold": i, "frac": frac, "model": m, "val_auroc": s})
        for m, s in base_va.items():
            lc_rows.append({"fold": i, "frac": 1.0, "model": m, "val_auroc": s})

    agg = lambda rows, keys, val: (pd.DataFrame(rows).groupby(keys)[val].agg(["mean", "std", "count"])
                                   .reset_index().round(5).to_dict("records"))
    out = {
        "permutation_feature": agg(feat_rows, ["model", "feature"], "drop"),
        "permutation_group": agg(group_rows, ["model", "group"], "drop"),
        "train_vs_val": agg(pd.DataFrame(gap_rows).melt(["fold", "model"]).to_dict("records"), ["model", "variable"], "value"),
        "learning_curve": agg(lc_rows, ["model", "frac"], "val_auroc"),
        "rf_impurity": agg(rf_imp, ["feature"], "importance"),
        "logistic_coef": agg(lr_coef, ["feature"], "coef"),
        "notes": "repeat-0 folds; permutation drop = AUROC(original) - AUROC(shuffled), mean of 5 shuffles; "
                 "learning-curve fractions < 1.0 use 1 MLP seed per architecture",
    }
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=1))
    log("wrote results/analysis.json")


if __name__ == "__main__":
    main()
