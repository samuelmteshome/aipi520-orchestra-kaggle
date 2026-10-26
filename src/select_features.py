"""Feature selection to K features, following the AIPI 520 W1 strategy (slide 34):

1. drop features with > 50% missing values
2. filter: rank by mutual information with the label, then drop near-duplicates
   (|Spearman| >= 0.95 with a higher-ranked feature), keeping ~35 candidates
3. embedded: random-forest impurity importance picks the final K

`select(X, y)` is meant to be called inside each CV training fold so selection never sees
validation rows (W2_H1 slide 19). Running this file does one selection on the full training
set as a preview only; CV scores must come from per-fold selection.

    python3 src/select_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

K = 25
N_CANDIDATES = 35
MAX_MISSING = 0.50
MAX_CORR = 0.95
ALWAYS_KEEP = ["id_cohort_Lh", "id_newer_than_Nu"]  # user decision: ID-cohort pattern stays in the model


def _numeric(X):
    """Categoricals -> integer codes so MI / correlation / forest can rank them."""
    out = X.copy()
    for c in out.select_dtypes(["category", "object"]).columns:
        out[c] = out[c].astype("category").cat.codes
    return out


def select(X, y, k=K, seed=0, report=False, keep=ALWAYS_KEEP):
    Xn = _numeric(X)
    # 1. null threshold
    miss = Xn.isna().mean()
    kept = miss[miss <= MAX_MISSING].index.tolist()
    Xf = Xn[kept].fillna(Xn[kept].median())

    # 2. filter: mutual information, then de-duplicate by rank correlation
    mi = pd.Series(mutual_info_classif(Xf, y, random_state=seed,
                                       discrete_features=[Xf[c].nunique() <= 10 for c in kept]), index=kept)
    corr = Xf.corr(method="spearman").abs()
    cands, dropped_dup = [], {}
    for c in mi.sort_values(ascending=False).index:
        twin = next((d for d in cands if corr.loc[c, d] >= MAX_CORR), None)
        if twin is None:
            cands.append(c)
        else:
            dropped_dup[c] = twin
        if len(cands) == N_CANDIDATES:
            break

    # 3. embedded: random-forest importance
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(Xf[cands], y)
    imp = pd.Series(rf.feature_importances_, index=cands).sort_values(ascending=False)
    chosen = [c for c in keep if c in X.columns]
    chosen += [c for c in imp.index if c not in chosen][: k - len(chosen)]

    if report:
        return chosen, {"missing": miss.round(3).to_dict(), "mi": mi.round(4).to_dict(),
                        "dropped_missing": sorted(set(X.columns) - set(kept)), "dropped_duplicate": dropped_dup,
                        "candidates": cands, "rf_importance": imp.round(4).to_dict()}
    return chosen


if __name__ == "__main__":
    tr = pd.read_parquet(PROCESSED / "features_train.parquet")
    y = tr.pop("label")
    chosen, rep = select(tr, y, report=True)
    rep["chosen"] = chosen
    (PROCESSED / "selection_preview.json").write_text(json.dumps(rep, indent=1))
    print(f"dropped >50% missing ({len(rep['dropped_missing'])}): {', '.join(rep['dropped_missing'])}")
    print(f"dropped as near-duplicates ({len(rep['dropped_duplicate'])}): {rep['dropped_duplicate']}")
    print(f"\nchosen {len(chosen)}:")
    for c in chosen:
        print(f"  {c:28s} rf={rep['rf_importance'].get(c, 0):.4f}  mi={rep['mi'][c]:.4f}  missing={rep['missing'][c]:.1%}")
