"""Final model: reference blend (pruned logistic + random forest) plus a nested-CV-selected additive spline logistic model.

Code generated during the blind modelling tournament (independent/tournament.py, "Model_Package_3") and kept unchanged except
for this header and default paths. It rebuilds features from the raw Kaggle files, runs 5-outer x 3-inner nested CV, refits on
all training rows and writes a submission.

Run (from the repo root, with the Kaggle files in for_students/):
    python3 src/final_model.py
Outputs in results/final_model/: submission.csv, oof_final.csv, results.json, diagnostics.json.
Environment overrides: DATA_DIR (raw files), OUT_DIR (outputs), BASELINE_SRC (module folder, default src/).
"""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
)
from threadpoolctl import threadpool_limits

START = time.monotonic()
SEED = 1729
CPUS = 8
CS = (0.1, 1.0, 10.0, 74.0)
WEIGHTS = (0.25, 0.5)
TOLERANCE = 0.0005
_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DATA_DIR", str(_ROOT / "for_students"))
os.environ.setdefault("OUT_DIR", str(_ROOT / "results" / "final_model"))
os.environ.setdefault("BASELINE_SRC", str(_ROOT / "src"))
OUT = Path(os.environ["OUT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ["DATA_DIR"])
sys.path.insert(0, os.environ["BASELINE_SRC"])

from clean import load_account, load_subscriptions, load_tickets, load_zipcodes
import clean as _clean
_clean.RAW = DATA          # read the raw files from DATA_DIR

np.random.seed(SEED)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

ACTIVE = {
    "San Francisco", "Berkeley Saturday", "Berkeley Sunday", "Peninsula"
}
IDS = ["id_cohort_Lh", "id_newer_than_Nu"]
PRUNE = {"geo_km_sf", "geo_km_berkeley", "geo_km_peninsula"}
CANDIDATE_COLUMNS = [
    "sub_ever", "sub_seasons_since_last", "sub_last_season",
    "sub_n_last3", "sub_n_last5", "sub_n_seasons", "sub_streak",
    "cons_sub_active_share",
    "sub_last_package", "sub_last_seats", "sub_last_venue_active",
    "any_seasons_since_last", "any_n_seasons",
    "don_2013", "don_lifetime", "don_count", "don_years_since_first",
    "don_first_in_2014", "geo_km_nearest_venue",
    "id_cohort_Lh", "id_newer_than_Nu",
]
SPLINE_COLUMNS = {
    "sub_seasons_since_last", "any_seasons_since_last",
    "don_years_since_first", "don_2013", "don_lifetime",
}
LOG_COLUMNS = {
    "sub_n_last3", "sub_n_last5", "sub_n_seasons", "sub_streak",
    "sub_last_seats", "any_n_seasons", "don_count",
    "geo_km_nearest_venue",
}
BROAD_REMOVE = {
    "id_cohort_Lh", "id_newer_than_Nu", "don_lifetime", "don_count",
    "don_years_since_first", "don_first_in_2014",
}

def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None

def sole_or_multiple(values):
    vals = pd.Series(values).dropna().astype(str).unique()
    if len(vals) == 0:
        return np.nan
    return vals[0] if len(vals) == 1 else "multiple"

def haversine(lat, lon, lat0, lon0):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lat0, lon0 = np.radians([float(lat0), float(lon0)])
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def construct_features(all_ids):
    account = load_account()
    subs = load_subscriptions()
    tickets = load_tickets()
    zips = load_zipcodes()

    assert account["account.id"].is_unique
    assert zips["zip5"].is_unique
    audit = {
        "subscription_rows_after_cutoff_removed": int((subs["year"] > 2013).sum()),
        "ticket_rows_after_cutoff_removed": int((tickets["year"] > 2013).sum()),
    }
    subs = subs.loc[subs["year"] <= 2013].copy()
    tickets = tickets.loc[tickets["year"] <= 2013].copy()
    audit["multirow_account_seasons"] = int(
        (subs.groupby(["account.id", "year"]).size() > 1).sum()
    )
    index = pd.Index(all_ids, name="account.id")
    X = pd.DataFrame(index=index)

    records = []
    for account_id, g in subs.groupby("account.id", sort=False):
        years = sorted(set(int(v) for v in g["year"]))
        latest = g.loc[g["year"] == years[-1]]
        previous = g.loc[g["year"] == years[-2]] if len(years) > 1 else None
        ys = set(years)
        streak = 0
        while 2013 - streak in ys:
            streak += 1
        last_seats = float(latest["no.seats"].max())
        last_tier = float(latest["subscription_tier"].max())
        price = pd.to_numeric(latest["price.level"], errors="coerce")
        records.append({
            "account.id": account_id,
            "sub_n_seasons": len(years),
            "sub_n_last3": sum(v >= 2011 for v in years),
            "sub_n_last5": sum(v >= 2009 for v in years),
            "sub_n_locations": g["location"].nunique(),
            "sub_n_multiple": int(g["multiple.subs"].sum()),
            "sub_mean_seats": float(g["no.seats"].mean()),
            "sub_max_tier": float(g["subscription_tier"].max()),
            "sub_mean_price_level": pd.to_numeric(
                g["price.level"], errors="coerce"
            ).mean(),
            "sub_share_active_venue": float(g["location"].isin(ACTIVE).mean()),
            "sub_streak": streak,
            "sub_seasons_since_last": 2014 - years[-1],
            "sub_span": years[-1] - years[0] + 1,
            "sub_gap_seasons": years[-1] - years[0] + 1 - len(years),
            "sub_last_season": int(2013 in ys),
            "sub_prev_season": int(2012 in ys),
            "sub_returned_after_gap": int(
                2013 in ys and 2012 not in ys and len(ys) > 1
            ),
            "sub_last_package": sole_or_multiple(latest["package_group"]),
            "sub_last_seats": last_seats,
            "sub_last_tier": last_tier,
            "sub_last_price_level": price.median() if price.notna().any() else np.nan,
            "sub_last_location": sole_or_multiple(latest["location"]),
            "sub_last_venue_active": int(latest["location"].isin(ACTIVE).any()),
            "sub_last_multiple": int(latest["multiple.subs"].any()),
            "sub_seat_change": (
                last_seats - float(previous["no.seats"].max())
                if previous is not None else np.nan
            ),
            "sub_tier_change": (
                last_tier - float(previous["subscription_tier"].max())
                if previous is not None else np.nan
            ),
            "cons_sub_active_share": len(years) / (2014 - years[0]),
        })
    X = X.join(pd.DataFrame(records).set_index("account.id"))

    ticket_records = []
    for account_id, g in tickets.groupby("account.id", sort=False):
        ticket_records.append({
            "account.id": account_id,
            "tix_n_purchases": len(g),
            "tix_n_seasons": g["year"].nunique(),
            "tix_n_concerts": len(
                g[["year", "set"]].dropna().drop_duplicates()
            ),
            "tix_total_seats": g["no.seats"].sum(),
            "tix_mean_price_level": g["price.level.num"].mean(),
            "tix_any_multiple": int(g["multiple.tickets"].any()),
            "tix_has_marketing_source": int(g["marketing.source"].notna().any()),
            "tix_family_concert": int(g["location"].eq("Family concert").any()),
            "tix_seasons_since_last": 2014 - int(g["year"].max()),
            "tix_last_season": int(g["year"].eq(2013).any()),
        })
    X = X.join(pd.DataFrame(ticket_records).set_index("account.id"))

    a = account.set_index("account.id").reindex(index)
    fd = pd.to_datetime(a["first.donated"], errors="coerce")
    X["don_lifetime"] = a["amount.donated.lifetime"]
    X["don_count"] = a["no.donations.lifetime"]
    X["don_2013"] = a["amount.donated.2013"]
    X["don_any"] = X["don_lifetime"].gt(0).astype(float)
    X["don_2013_any"] = X["don_2013"].gt(0).astype(float)
    X["don_avg_gift"] = (
        X["don_lifetime"] / X["don_count"]
    ).where(X["don_count"] > 0, 0)
    X["don_years_since_first"] = (
        (pd.Timestamp("2014-01-01") - fd).dt.days / 365.25
    ).where(fd.dt.year < 2014)
    X["don_first_in_2014"] = fd.dt.year.eq(2014).astype(float)
    X["has_relationship"] = a["relationship"].notna().astype(float)
    X["has_billing_zip"] = a["billing.zip5"].notna().astype(float)
    X["has_shipping_zip"] = a["shipping.zip5"].notna().astype(float)

    z = zips.set_index("zip5")
    home = z.reindex(a["billing.zip5"].to_numpy()).copy()
    home.index = index
    X["geo_in_CA"] = home["State"].eq("CA").astype(float)
    for name, zipcode in (
        ("sf", "94102"), ("berkeley", "94704"), ("peninsula", "94301")
    ):
        X["geo_km_" + name] = haversine(
            home["Lat"], home["Long"], z.loc[zipcode, "Lat"], z.loc[zipcode, "Long"]
        )
    X["geo_km_nearest_venue"] = X[
        ["geo_km_sf", "geo_km_berkeley", "geo_km_peninsula"]
    ].min(axis=1)
    X["geo_zip_wages_per_return"] = home["wages_per_return"]
    X["geo_zip_population"] = home["EstimatedPopulation"]

    id_series = pd.Series(index.to_numpy(), index=index, dtype="string")
    prefix = "001i000000Lh"
    assert len(prefix) == 12
    # Compare values rather than Series.equals, which also compares dtypes.
    sliced = id_series.str[:len(prefix)].eq(prefix).to_numpy(
        dtype=bool, na_value=False
    )
    starts = id_series.str.startswith(prefix, na=False).to_numpy(
        dtype=bool, na_value=False
    )
    if not np.array_equal(sliced, starts):
        raise AssertionError("Prefix implementations disagree on actual ID values.")
    X["id_cohort_Lh"] = starts.astype(float)
    X["id_newer_than_Nu"] = (
        (id_series > "001i000000Nv").to_numpy(dtype=bool, na_value=False)
    ).astype(float)
    audit["prefix_value_equivalence_verified"] = True
    audit["prefix_length"] = len(prefix)

    zero_prefixes = (
        "sub_n", "sub_streak", "sub_last_season", "sub_prev_season",
        "sub_returned", "sub_share", "sub_last_venue", "sub_last_multiple",
        "tix_n", "tix_total", "tix_any", "tix_has", "tix_family",
        "tix_last_season", "cons_sub_active_share",
    )
    for col in X:
        if col.startswith(zero_prefixes):
            X[col] = X[col].fillna(0)
    for col in ("sub_seasons_since_last", "tix_seasons_since_last"):
        X[col] = X[col].fillna(99)

    history = pd.concat([
        subs[["account.id", "year"]], tickets[["account.id", "year"]]
    ], ignore_index=True).groupby("account.id")["year"]
    X["any_n_seasons"] = history.nunique().reindex(index).fillna(0)
    X["any_seasons_since_last"] = (
        2014 - history.max()
    ).reindex(index).fillna(99)
    X["sub_ever"] = X["sub_n_seasons"].gt(0).astype(float)
    X["tix_ever"] = X["tix_n_purchases"].gt(0).astype(float)

    for col in ("sub_last_package", "sub_last_location"):
        X[col] = X[col].astype(object)
        X.loc[X["sub_ever"].eq(0), col] = "none"
        X[col] = X[col].fillna("unknown").astype("category")

    X["ix_recent_x_don_count"] = X["sub_last_season"] * X["don_count"]
    X["ix_recent_x_tix_seats"] = X["sub_last_season"] * X["tix_total_seats"]

    for col in X:
        if col not in ("sub_last_package", "sub_last_location"):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    assert X.index.is_unique
    assert len(X) == len(index)
    audit["feature_columns_before_selection"] = list(X.columns)
    audit["cohort_prevalence_all_scored_accounts"] = {
        c: float(X[c].mean()) for c in IDS
    }
    return X, audit

def reference_pool(X):
    prefixes = ("sub_", "tix_", "any_", "don_", "has_", "geo_", "id_", "ix_")
    return X[[c for c in X if c.startswith(prefixes)]]

def select_reference_features(X, y):
    Xn = X.copy()
    for c in Xn:
        if not pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = Xn[c].astype("category").cat.codes.astype(float)
    keep = [c for c in Xn if Xn[c].isna().mean() <= 0.50]
    Z = Xn[keep].copy()
    Z = Z.fillna(Z.median()).fillna(0)
    mi = pd.Series(
        mutual_info_classif(
            Z, y, random_state=SEED,
            discrete_features=[Z[c].nunique() <= 10 for c in Z]
        ),
        index=keep,
    )
    corr = Z.corr(method="spearman").abs()
    candidates = []
    for c in mi.sort_values(ascending=False, kind="stable").index:
        if not any(corr.loc[c, d] >= 0.95 for d in candidates):
            candidates.append(c)
        if len(candidates) == 35:
            break
    selector = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, n_jobs=CPUS, random_state=SEED
    )
    selector.fit(Z[candidates], y)
    ranked = pd.Series(
        selector.feature_importances_, index=candidates
    ).sort_values(ascending=False, kind="stable")
    chosen = [c for c in IDS if c in X]
    chosen += [c for c in ranked.index if c not in chosen][:25 - len(chosen)]
    return [
        c for c in chosen
        if not c.startswith(("tix_", "ix_")) and c not in PRUNE
    ]

def never_flags(X):
    out = X.copy()
    for c in list(out.columns):
        if c.endswith("seasons_since_last"):
            out[c + "_never"] = out[c].eq(99).astype(float)
            out[c] = out[c].where(out[c] != 99)
    return out

def make_reference_scaled(X):
    Z = never_flags(X)
    cats = [c for c in Z if not pd.api.types.is_numeric_dtype(Z[c])]
    nums = [c for c in Z if c not in cats]
    skew = [
        c for c in nums
        if (Z[c].dropna() >= 0).all() and Z[c].skew() > 2
    ]
    other = [c for c in nums if c not in skew]

    def numeric_steps(log):
        steps = []
        if log:
            steps.append(("log", FunctionTransformer(np.log1p)))
        steps += [
            ("impute", SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )),
            ("scale", StandardScaler()),
        ]
        return Pipeline(steps)

    ct = ColumnTransformer([
        ("skew", numeric_steps(True), skew),
        ("num", numeric_steps(False), other),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ], sparse_threshold=0)
    return Pipeline([
        ("never", FunctionTransformer(never_flags)),
        ("columns", ct),
    ])

def make_reference_tree(X):
    cats = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X if c not in cats]
    return ColumnTransformer([
        ("num", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ), nums),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ), cats),
    ], sparse_threshold=0)

def cdf_map(sorted_training_scores, scores):
    left = np.searchsorted(sorted_training_scores, scores, side="left")
    right = np.searchsorted(sorted_training_scores, scores, side="right")
    return (left + right) / (2.0 * len(sorted_training_scores))

class Reference:
    def fit(self, X, y):
        X = reference_pool(X)
        self.columns = select_reference_features(X, y)
        Z = X[self.columns]
        self.linear = Pipeline([
            ("prep", make_reference_scaled(Z)),
            ("model", LogisticRegression(
                C=74.0, l1_ratio=0.0, solver="saga",
                max_iter=20000, tol=1e-3, random_state=SEED
            )),
        ])
        self.forest = Pipeline([
            ("prep", make_reference_tree(Z)),
            ("model", RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_leaf=2,
                max_features=0.4, class_weight="balanced",
                n_jobs=CPUS, random_state=SEED
            )),
        ])
        self.linear.fit(Z, y)
        self.forest.fit(Z, y)
        self.linear_cdf = np.sort(self.linear.predict_proba(Z)[:, 1])
        self.forest_cdf = np.sort(self.forest.predict_proba(Z)[:, 1])
        return self

    def predict(self, X):
        Z = X[self.columns]
        a = cdf_map(self.linear_cdf, self.linear.predict_proba(Z)[:, 1])
        b = cdf_map(self.forest_cdf, self.forest.predict_proba(Z)[:, 1])
        return 0.5 * (a + b)

class AdditivePreprocessor:
    def __init__(self, columns):
        self.columns = list(columns)

    def values(self, X, c):
        arr = pd.to_numeric(X[c], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ).copy()
        arr[~np.isfinite(arr)] = np.nan
        if c.endswith("seasons_since_last"):
            arr[arr == 99] = np.nan
        if c in {"don_2013", "don_lifetime"}:
            if np.any(arr[np.isfinite(arr)] < 0):
                raise ValueError("Negative donation value violates frozen log transform.")
            arr = np.log1p(arr)
        return arr

    def fit(self, X):
        self.spec = {}
        for c in self.columns:
            if c == "sub_last_package":
                continue
            v = self.values(X, c)
            finite = v[np.isfinite(v)]
            if c in SPLINE_COLUMNS:
                if len(finite):
                    lower = (
                        0.0 if c in {"don_2013", "don_lifetime"}
                        else float(finite.min())
                    )
                    upper = float(finite.max())
                    pool = (
                        finite[finite > 0]
                        if c in {"don_2013", "don_lifetime"} else finite
                    )
                    knots = (
                        np.unique(np.quantile(pool, [0.25, 0.5, 0.75]))
                        if len(pool) else np.array([], dtype=float)
                    )
                    knots = knots[(knots > lower) & (knots < upper)]
                else:
                    lower = upper = 0.0
                    knots = np.array([], dtype=float)
                self.spec[c] = ("spline", lower, upper, knots)
            else:
                if c in LOG_COLUMNS:
                    finite = np.log1p(finite)
                median = float(np.median(finite)) if len(finite) else 0.0
                self.spec[c] = ("linear", median)
        self.scaler = StandardScaler().fit(self.numeric(X))
        self.encoder = None
        if "sub_last_package" in self.columns:
            self.encoder = OneHotEncoder(
                categories=[["Full", "Quartet", "Trio", "CYO",
                             "unknown", "multiple", "none"]],
                handle_unknown="ignore", sparse_output=False,
            ).fit(self.packages(X))
        self.n_columns = self.transform(X).shape[1]
        return self

    def packages(self, X):
        return X["sub_last_package"].astype(object).fillna(
            "unknown"
        ).astype(str).to_numpy().reshape(-1, 1)

    def numeric(self, X):
        blocks = []
        for c in self.columns:
            if c == "sub_last_package":
                continue
            v = self.values(X, c)
            missing = ~np.isfinite(v)
            spec = self.spec[c]
            if spec[0] == "spline":
                _, lower, upper, knots = spec
                v = np.clip(np.where(missing, lower, v), lower, upper)
                blocks.append(v)
                blocks.extend(np.maximum(v - knot, 0.0) for knot in knots)
            else:
                if c in LOG_COLUMNS:
                    v = np.log1p(v)
                blocks.append(np.where(missing, spec[1], v))
            blocks.append(missing.astype(float))
        mat = np.column_stack(blocks)
        if not np.isfinite(mat).all():
            raise ValueError("Nonfinite additive design matrix")
        return mat

    def transform(self, X):
        mat = self.scaler.transform(self.numeric(X))
        if self.encoder is not None:
            mat = np.column_stack([mat, self.encoder.transform(self.packages(X))])
        return mat

    def describe(self):
        return {
            "expanded_columns": int(self.n_columns),
            "spline_specs": {
                c: {
                    "lower": float(spec[1]),
                    "upper": float(spec[2]),
                    "knots": spec[3].tolist(),
                }
                for c, spec in self.spec.items() if spec[0] == "spline"
            },
        }

def fit_additive_scores(Xtr, ytr, Xval, cs=CS, columns=CANDIDATE_COLUMNS):
    prep = AdditivePreprocessor(columns).fit(Xtr)
    A, B = prep.transform(Xtr), prep.transform(Xval)
    output = {}
    info = prep.describe()
    info["iterations"] = {}
    for C in cs:
        model = LogisticRegression(
            C=C, l1_ratio=0.0, solver="lbfgs",
            class_weight=None, max_iter=10000, tol=1e-6,
            random_state=SEED,
        )
        model.fit(A, ytr)
        training = model.predict_proba(A)[:, 1]
        ref = np.sort(training)
        output[float(C)] = {
            "validation": cdf_map(ref, model.predict_proba(B)[:, 1]),
            "training": cdf_map(ref, training),
        }
        info["iterations"][str(C)] = int(model.n_iter_[0])
    return output, info

def make_splits(X, y, n_splits, seed):
    strata = np.asarray(y, dtype=int) * 2 + X["sub_last_season"].to_numpy(dtype=int)
    stratification = "label_x_previous_subscription"
    if pd.Series(strata).value_counts().min() < n_splits:
        strata = np.asarray(y, dtype=int)
        stratification = "label"
    if pd.Series(strata).value_counts().min() < n_splits:
        raise ValueError("Insufficient class counts for frozen validation protocol")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), strata)), stratification

CONFIGS = [(0.0, None)] + [(w, C) for w in WEIGHTS for C in CS]

def config_name(w, C):
    return "reference" if w == 0 else f"additive_weight={w:g},C={C:g}"

def inner_select(X, y):
    splits, stratification = make_splits(X, y, 3, SEED + 1)
    scores = {config_name(w, C): [] for w, C in CONFIGS}
    details = []
    for fold, (tr, va) in enumerate(splits):
        ref = Reference().fit(X.iloc[tr], y[tr])
        rp = ref.predict(X.iloc[va])
        additive, add_info = fit_additive_scores(X.iloc[tr], y[tr], X.iloc[va])
        scores["reference"].append(auc(y[va], rp))
        for w, C in CONFIGS[1:]:
            pred = (1 - w) * rp + w * additive[C]["validation"]
            scores[config_name(w, C)].append(auc(y[va], pred))
        details.append({
            "fold": fold,
            "selected_reference_features": ref.columns,
            "additive_design": add_info,
            "training_positive_count": int(y[tr].sum()),
            "validation_positive_count": int(y[va].sum()),
        })
        del ref

    means = {name: float(np.mean(values)) for name, values in scores.items()}
    best_nonzero = max(means[config_name(w, C)] for w, C in CONFIGS[1:])
    chosen = (0.0, None)
    if best_nonzero - means["reference"] >= TOLERANCE:
        eligible = [
            (w, C) for w, C in CONFIGS[1:]
            if means[config_name(w, C)] >= best_nonzero - TOLERANCE
        ]
        chosen = min(eligible, key=lambda v: (v[0], v[1]))
    return chosen, {
        "stratification": stratification,
        "mean_inner_auroc": means,
        "fold_inner_auroc": scores,
        "choice": config_name(*chosen),
        "fold_details": details,
    }

def bootstrap_difference(y, p, q, seed=SEED, repetitions=500):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(np.asarray(y) == 1)
    neg = np.flatnonzero(np.asarray(y) == 0)
    differences = []
    for _ in range(repetitions):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        differences.append(
            roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], q[idx])
        )
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(roc_auc_score(y, p) - roc_auc_score(y, q)),
        "approximate_95_percent_interval": [float(lo), float(hi)],
        "replicates": repetitions,
        "method": "paired account bootstrap, stratified by label",
        "limitation": (
            "Conditional on fitted OOF predictions; does not capture all training dependence."
        ),
    }

def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def write_json(name, value):
    with (OUT / name).open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, allow_nan=False)

def main():
    train = pd.read_csv(DATA / "train.csv", dtype={"account.id": str})
    test = pd.read_csv(DATA / "test.csv", dtype={"ID": str})
    assert train["account.id"].is_unique and test["ID"].is_unique
    assert train["account.id"].notna().all() and test["ID"].notna().all()
    assert set(train["account.id"]).isdisjoint(set(test["ID"]))
    assert set(train["label"].unique()) == {0, 1}
    y = train["label"].to_numpy(dtype=int)

    all_ids = train["account.id"].tolist() + test["ID"].tolist()
    features, audit = construct_features(all_ids)
    X = features.loc[train["account.id"]].copy()
    XT = features.loc[test["ID"]].copy()
    assert X.index.tolist() == train["account.id"].tolist()
    assert XT.index.tolist() == test["ID"].tolist()
    print(f"Features constructed: {len(X)} training, {len(XT)} test accounts.", flush=True)

    outer, stratification = make_splits(X, y, 5, SEED)
    oof = np.full(len(y), np.nan)
    reference_oof = np.full(len(y), np.nan)
    additive_c1_oof = np.full(len(y), np.nan)
    flag_ablation_oof = np.full(len(y), np.nan)
    broad_ablation_oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    folds = []

    for fold, (tr, va) in enumerate(outer):
        chosen, inner_info = inner_select(X.iloc[tr], y[tr])
        w, C = chosen
        reference = Reference().fit(X.iloc[tr], y[tr])
        rp = reference.predict(X.iloc[va])
        rt = reference.predict(X.iloc[tr])
        reference_oof[va] = rp

        requested_cs = sorted(set([1.0] + ([C] if C is not None else [])))
        add, add_info = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=requested_cs
        )
        additive_c1_oof[va] = add[1.0]["validation"]
        if w == 0:
            prediction, training_prediction = rp, rt
        else:
            prediction = (1 - w) * rp + w * add[C]["validation"]
            training_prediction = (1 - w) * rt + w * add[C]["training"]
        oof[va] = prediction
        fold_ids[va] = fold

        no_flag = [c for c in CANDIDATE_COLUMNS if c != "don_first_in_2014"]
        diagnostic, _ = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=[1.0], columns=no_flag
        )
        flag_ablation_oof[va] = diagnostic[1.0]["validation"]
        broad_columns = [c for c in CANDIDATE_COLUMNS if c not in BROAD_REMOVE]
        diagnostic, _ = fit_additive_scores(
            X.iloc[tr], y[tr], X.iloc[va], cs=[1.0], columns=broad_columns
        )
        broad_ablation_oof[va] = diagnostic[1.0]["validation"]

        hard = X.iloc[va]["sub_last_season"].eq(0).to_numpy()
        folds.append({
            "fold": fold,
            "inner_selection": inner_info,
            "selected_reference_features": reference.columns,
            "additive_design": add_info,
            "final_validation_auroc": auc(y[va], prediction),
            "reference_validation_auroc": auc(y[va], rp),
            "final_training_auroc": auc(y[tr], training_prediction),
            "reference_training_auroc": auc(y[tr], rt),
            "validation_rows": len(va),
            "validation_positive_count": int(y[va].sum()),
            "hard_group_rows": int(hard.sum()),
            "hard_group_positive_count": int(y[va][hard].sum()),
            "hard_group_auroc": auc(y[va][hard], prediction[hard]),
        })
        print(
            f"Outer fold {fold + 1}/5 complete; "
            f"elapsed {(time.monotonic() - START) / 60:.1f} minutes",
            flush=True,
        )
        del reference

    assert np.isfinite(oof).all()
    assert (fold_ids >= 0).all()

    # The same frozen inner selection rule is applied to all supplied training rows.
    final_choice, final_inner = inner_select(X, y)
    w, C = final_choice
    reference = Reference().fit(X, y)
    test_reference = reference.predict(XT)
    if w == 0:
        test_prediction = test_reference
        final_add_info = None
    else:
        final_add, final_add_info = fit_additive_scores(X, y, XT, cs=[C])
        test_prediction = (
            (1 - w) * test_reference + w * final_add[C]["validation"]
        )
    assert len(test_prediction) == len(test)
    assert np.isfinite(test_prediction).all()
    assert np.all((test_prediction >= 0) & (test_prediction <= 1))

    pd.DataFrame({
        "account.id": train["account.id"].to_numpy(),
        "pred": oof,
    }).to_csv(OUT / "oof_final.csv", index=False)
    pd.DataFrame({
        "ID": test["ID"].to_numpy(),
        "Predicted": test_prediction,
    }).to_csv(OUT / "submission.csv", index=False)

    hard = X["sub_last_season"].eq(0).to_numpy()
    model_name = "nested_reference_additive_blend"
    write_json("results.json", {
        "models": {
            "corrected_reference": {
                "cv_auroc": auc(y, reference_oof),
                "notes": (
                    "Outer OOF evaluation of frozen corrected logistic/forest "
                    "reference with fold-local selection and training-score CDFs."
                ),
            },
            model_name: {
                "cv_auroc": auc(y, oof),
                "notes": (
                    "Outer OOF evaluation of the complete inner-selection rule. "
                    f"Full-training inner choice: {config_name(*final_choice)}. "
                    "Outer results and sensitivity diagnostics do not alter selection."
                ),
            },
        },
        "best": model_name,
        "validation": (
            "Five fixed outer folds and three inner folds; joint label/history "
            "stratification where feasible; no outer-score model selection."
        ),
    })

    diagnostics = {
        "seed": SEED,
        "elapsed_seconds": time.monotonic() - START,
        "training_rows": len(train),
        "test_rows": len(test),
        "training_positives": int(y.sum()),
        "outer_stratification": stratification,
        "audit": audit,
        "frozen_additive_source_features": CANDIDATE_COLUMNS,
        "inverse_penalty_grid": CS,
        "additive_weight_grid": [0.0, *WEIGHTS],
        "selection_tolerance": TOLERANCE,
        "outer_folds": folds,
        "outer_mean_final_auroc": float(np.mean([
            f["final_validation_auroc"] for f in folds
        ])),
        "outer_mean_reference_auroc": float(np.mean([
            f["reference_validation_auroc"] for f in folds
        ])),
        "paired_oof_difference": bootstrap_difference(y, oof, reference_oof),
        "hard_group": {
            "rows": int(hard.sum()),
            "positives": int(y[hard].sum()),
            "final_auroc": auc(y[hard], oof[hard]),
            "reference_auroc": auc(y[hard], reference_oof[hard]),
        },
        "diagnostic_only_ablations": {
            "fixed_C": 1.0,
            "full_additive_auroc": auc(y, additive_c1_oof),
            "without_first_gift_flag_auroc": auc(y, flag_ablation_oof),
            "without_cohorts_and_uncertain_donations_auroc": auc(
                y, broad_ablation_oof
            ),
            "flag_removal_difference": float(
                auc(y, flag_ablation_oof) - auc(y, additive_c1_oof)
            ),
            "broad_removal_difference": float(
                auc(y, broad_ablation_oof) - auc(y, additive_c1_oof)
            ),
            "used_for_selection": False,
        },
        "full_training_inner_selection": final_inner,
        "final_reference_features": reference.columns,
        "final_additive_design": final_add_info,
        "limitations": [
            "Training-score CDFs may differ from unseen score distributions.",
            "Outputs are bounded ranking scores, not calibrated probabilities.",
            "Bootstrap intervals omit some training and fold dependence.",
            "Donation snapshots and record cohorts have temporal transportability risks.",
            "Historical development choices may contain inherited selection optimism.",
            "Maximum within-season seats is a deterministic summary, not a verified total.",
            "No hidden labels or leaderboard outcomes are read.",
        ],
    }
    write_json("diagnostics.json", diagnostics)
    print("Required outputs written.", flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
