"""Build one row of features per account.

Features are grouped so whole groups can be switched on or off when modelling
(see FEATURE_GROUPS). History features are computed "as of" a cutoff season,
so the same code can build a time-shifted dataset (e.g. predict 2013-14 from
data up to 2012-13) for extra validation.

    python3 src/features.py        -> data/processed/features_{train,test}.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import PROCESSED, TARGET_YEAR, load_all  # noqa: E402

NEVER = 99  # "seasons since" value for accounts with no such event


# Approximate venue towns, used for distance features (zip-code centroids, not exact halls).
VENUE_ZIPS = {"sf": "94102", "berkeley": "94704", "peninsula": "94301"}


# ---------------------------------------------------------------- subscriptions
def streak_ending(years, last_year):
    k = 0
    while (last_year - k) in years:
        k += 1
    return k


def subscription_features(subs, asof):
    """History up to and including season start year `asof` (2013 for the real task)."""
    s = subs[subs["year"] <= asof].sort_values("year")
    # Venue series still running in season `asof` (in 2013-14: SF, Berkeley Sat/Sun, Peninsula;
    # Santa Rosa ended after 1999-00, Orange County ran 1996-2002, Contra Costa ended after 2010-11).
    active = set(s.loc[s["year"] == asof, "location"].dropna())
    g = s.groupby("account.id")
    f = pd.DataFrame({
        "sub_n_seasons": g["year"].nunique(),
        "sub_first_year": g["year"].min(),
        "sub_last_year": g["year"].max(),
        "sub_n_last3": g["year"].apply(lambda y: y.ge(asof - 2).sum()),
        "sub_n_last5": g["year"].apply(lambda y: y.ge(asof - 4).sum()),
        "sub_n_locations": g["location"].nunique(),
        "sub_n_multiple": g["multiple.subs"].sum(),
        "sub_mean_seats": g["no.seats"].mean(),
        "sub_max_tier": g["subscription_tier"].max(),
        "sub_mean_price_level": g["price.level"].mean(),
        "sub_share_active_venue": g["location"].apply(lambda l: l.isin(active).mean()),
    })
    years = g["year"].apply(set)
    f["sub_streak"] = years.map(lambda y: streak_ending(y, asof))
    f["sub_seasons_since_last"] = asof + 1 - f["sub_last_year"]           # 1 = subscribed last season
    f["sub_span"] = f["sub_last_year"] - f["sub_first_year"] + 1
    f["sub_gap_seasons"] = f["sub_span"] - f["sub_n_seasons"]             # missed seasons inside the span
    f["sub_last_season"] = f["sub_seasons_since_last"].eq(1).astype(int)
    f["sub_prev_season"] = years.map(lambda y: int(asof - 1 in y))
    f["sub_returned_after_gap"] = years.map(lambda y: int(asof in y and asof - 1 not in y and len(y) > 1))

    last = s.groupby("account.id").tail(1).set_index("account.id")
    f["sub_last_package"] = last["package_group"]
    f["sub_last_seats"] = last["no.seats"]
    f["sub_last_tier"] = last["subscription_tier"]
    f["sub_last_price_level"] = last["price.level"]
    f["sub_last_location"] = last["location"]
    f["sub_last_venue_active"] = last["location"].isin(active).astype(int)
    f["sub_last_multiple"] = last["multiple.subs"].astype(int)

    # Package / seat change between the last two subscribed seasons
    prev = s.groupby("account.id").nth(-2).set_index("account.id")
    f["sub_seat_change"] = f["sub_last_seats"] - prev["no.seats"]
    f["sub_tier_change"] = f["sub_last_tier"] - prev["subscription_tier"]
    return f.drop(columns=["sub_first_year", "sub_last_year"])


# ---------------------------------------------------------------- tickets
def ticket_features(tix, asof):
    t = tix[tix["year"] <= asof]
    g = t.groupby("account.id")
    f = pd.DataFrame({
        "tix_n_purchases": g.size(),
        "tix_n_seasons": g["year"].nunique(),
        "tix_n_concerts": g.apply(lambda d: d[["year", "set"]].dropna().drop_duplicates().shape[0], include_groups=False),
        "tix_total_seats": g["no.seats"].sum(),
        "tix_mean_price_level": g["price.level.num"].mean(),
        "tix_any_multiple": g["multiple.tickets"].max().astype(int),
        "tix_has_marketing_source": g["marketing.source"].apply(lambda m: int(m.notna().any())),
        "tix_family_concert": g["location"].apply(lambda l: int(l.eq("Family concert").any())),
    })
    f["tix_seasons_since_last"] = asof + 1 - g["year"].max()
    f["tix_last_season"] = f["tix_seasons_since_last"].eq(1).astype(int)
    return f


# ---------------------------------------------------------------- account / donations
def account_features(acc):
    a = acc.set_index("account.id")
    fd = a["first.donated"]
    f = pd.DataFrame(index=a.index)
    f["don_lifetime"] = a["amount.donated.lifetime"]
    f["don_count"] = a["no.donations.lifetime"]
    f["don_2013"] = a["amount.donated.2013"]
    f["don_any"] = a["amount.donated.lifetime"].gt(0).astype(int)
    f["don_2013_any"] = a["amount.donated.2013"].gt(0).astype(int)
    f["don_avg_gift"] = (a["amount.donated.lifetime"] / a["no.donations.lifetime"]).where(a["no.donations.lifetime"] > 0, 0)
    # Years of giving before 2014; NaN for 2014 first gifts so the 2014 signal lives only in the flag below.
    f["don_years_since_first"] = ((pd.Timestamp("2014-01-01") - fd).dt.days / 365.25).where(fd.dt.year < 2014)
    # Kept on purpose (user decision): first gift in 2014 is close to the 2014-15 sales window.
    f["don_first_in_2014"] = fd.dt.year.eq(2014).astype(int)
    # Strict variants for the ablation: accounts whose first gift was in 2014 have only 2014 money in
    # their lifetime totals, so zero it out. (Repeat donors' 2014 gifts can't be separated in the data.)
    pre = fd.dt.year.ne(2014)
    f["pre14_don_lifetime"] = f["don_lifetime"].where(pre, 0)
    f["pre14_don_count"] = f["don_count"].where(pre, 0)
    f["has_relationship"] = a["relationship"].notna().astype(int)
    f["has_billing_zip"] = a["billing.zip5"].notna().astype(int)
    f["has_shipping_zip"] = a["shipping.zip5"].notna().astype(int)
    return f


def id_features(ids):
    """Record-cohort features from the Salesforce-style account ID (optional group)."""
    pfx = ids.str[:12]
    f = pd.DataFrame(index=ids)
    f["id_cohort_Lh"] = pfx.eq("001i000000Lh").astype(int).values
    f["id_newer_than_Nu"] = (ids > "001i000000Nv").astype(int).values
    return f


# ---------------------------------------------------------------- geography
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def geo_features(acc, zips):
    z = zips.set_index("zip5")
    a = acc.set_index("account.id")
    j = z.reindex(a["billing.zip5"].values)
    j.index = a.index
    f = pd.DataFrame(index=a.index)
    f["geo_in_CA"] = j["State"].eq("CA").astype(int)
    for name, vz in VENUE_ZIPS.items():
        f[f"geo_km_{name}"] = haversine_km(j["Lat"], j["Long"], z.loc[vz, "Lat"], z.loc[vz, "Long"])
    f["geo_km_nearest_venue"] = f[[f"geo_km_{n}" for n in VENUE_ZIPS]].min(axis=1)
    f["geo_zip_wages_per_return"] = j["wages_per_return"]
    f["geo_zip_population"] = j["EstimatedPopulation"]
    return f


# ---------------------------------------------------------------- attendance consistency (user idea, 2026-10-26)
def longest_run(years):
    best = run = 0
    prev = None
    for yr in sorted(years):
        run = run + 1 if prev is not None and yr == prev + 1 else 1
        best, prev = max(best, run), yr
    return best


def consistency_features(subs, tix, acc, asof):
    """Reliable "goes to everything" patrons vs spontaneous ones.

    Subscriptions: share of seasons attended since first subscribing, longest unbroken run, number of lapses,
    share of seasons on a Full (every-concert) package, loyalty to one venue. Tickets: concerts per ticket
    season and share of the ticket seasons (2010-14) with a purchase. Donations: account.csv has no per-season
    giving, so regularity is approximated by gifts per year since the first gift (capped at 3).
    """
    s = subs[subs["year"] <= asof]
    g = s.groupby("account.id")
    years = g["year"].apply(set)
    f = pd.DataFrame(index=years.index)
    first, n = g["year"].min(), g["year"].nunique()
    f["cons_sub_active_share"] = n / (asof - first + 1)                       # 1.0 = every season since joining
    f["cons_sub_longest_run"] = years.map(longest_run)
    f["cons_sub_lapses"] = years.map(lambda y: sum(1 for yr in y if yr - 1 not in y) - 1)   # breaks in the run
    f["cons_sub_full_share"] = g["package_group"].apply(lambda p: p.eq("Full").mean())
    f["cons_sub_venue_loyalty"] = g["location"].apply(lambda l: l.value_counts(normalize=True).iloc[0] if l.notna().any() else 0)

    t = tix[tix["year"] <= asof]
    tg = t.groupby("account.id")
    tf = pd.DataFrame({
        "cons_tix_concerts_per_season": tg.apply(lambda d: d[["year", "set"]].dropna().drop_duplicates().shape[0], include_groups=False)
                                        / tg["year"].nunique(),
        "cons_tix_season_share": tg["year"].nunique() / tix.loc[tix["year"] <= asof, "year"].nunique(),
    })

    a = acc.set_index("account.id")
    first_gift = a["first.donated"].dt.year
    yrs = (TARGET_YEAR - first_gift).clip(lower=1)
    df = pd.DataFrame({"cons_don_per_year": (a["no.donations.lifetime"] / yrs).clip(upper=3).fillna(0)})
    return f.join(tf, how="outer").join(df, how="outer")


def heuristic_interactions(X):
    """Three interactions picked from the 2x2 screen (2026-10-26)."""
    last = X["sub_last_season"].eq(1)
    return pd.DataFrame({
        "hx_activevenue_x_donor": (X["sub_last_venue_active"].eq(1) & X["don_count"].gt(0)).astype(int),
        "hx_last_x_singletix": (last & X["tix_n_purchases"].gt(0)).astype(int),
        "hx_last_x_downsized": (last & X["sub_last_package"].astype(str).isin(["Quartet", "Trio", "CYO"])).astype(int),
    }, index=X.index)


# ---------------------------------------------------------------- assemble
FEATURE_GROUPS = {
    "subscriptions": "sub_", "tickets": "tix_", "any_purchase": "any_", "donations": "don_",
    "donations_pre2014": "pre14_", "account_flags": ("has_",), "geography": "geo_", "id_cohort": "id_",
    "consistency": "cons_", "heuristic_interactions": "hx_",
}
CATEGORICAL = ["sub_last_package", "sub_last_location"]


def build(d, ids, asof=TARGET_YEAR - 1):
    """Feature table for `ids`, using history up to season start year `asof`."""
    base = pd.DataFrame(index=pd.Index(ids, name="account.id"))
    parts = [
        subscription_features(d["subscriptions"], asof),
        ticket_features(d["tickets"], asof),
        account_features(d["account"]),
        geo_features(d["account"], d["zipcodes"]),
        id_features(pd.Series(ids)),
    ]
    if asof < TARGET_YEAR - 1:
        # Time-shifted builds: donation fields and ID cohorts are one 2014 snapshot and cannot be shifted.
        parts = parts[:2] + [parts[3].drop(columns=["geo_zip_wages_per_return", "geo_zip_population"], errors="ignore")]
    X = base.join(parts, how="left")
    # Accounts with no subscription / ticket history: counts are 0, "since" values are NEVER.
    for c in X.columns:
        if c.startswith(("sub_n", "sub_streak", "sub_last_season", "sub_prev_season", "sub_returned",
                         "sub_share", "sub_last_venue", "sub_last_multiple", "tix_n", "tix_total",
                         "tix_any", "tix_has", "tix_family", "tix_last_season")):
            X[c] = X[c].fillna(0)
    X["sub_seasons_since_last"] = X["sub_seasons_since_last"].fillna(NEVER)
    X["tix_seasons_since_last"] = X["tix_seasons_since_last"].fillna(NEVER)
    # Distinct seasons with any purchase (subscription or single ticket), and recency of any purchase.
    hist = pd.concat([d["subscriptions"][["account.id", "year"]], d["tickets"][["account.id", "year"]]])
    hist = hist[hist["year"] <= asof].groupby("account.id")["year"]
    X["any_n_seasons"] = hist.nunique().reindex(X.index).fillna(0)
    X["any_seasons_since_last"] = (asof + 1 - hist.max()).reindex(X.index).fillna(NEVER)
    X["sub_ever"] = X["sub_n_seasons"].gt(0).astype(int)
    X["tix_ever"] = X["tix_n_purchases"].gt(0).astype(int)
    for c in CATEGORICAL:
        X[c] = X[c].fillna("none").astype("category")
    cons = consistency_features(d["subscriptions"], d["tickets"], d["account"], asof).reindex(X.index).fillna(0)
    if asof < TARGET_YEAR - 1:
        cons = cons.drop(columns=["cons_don_per_year"])
    X = X.join(cons)
    X = X.join(heuristic_interactions(X))
    return X


def groups(columns):
    out = {}
    for g, prefix in FEATURE_GROUPS.items():
        out[g] = [c for c in columns if c.startswith(prefix)]
    return out


if __name__ == "__main__":
    d = load_all()
    Xtr = build(d, d["train"]["account.id"]).join(d["train"].set_index("account.id")["label"])
    Xte = build(d, d["test"]["account.id"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    Xtr.to_parquet(PROCESSED / "features_train.parquet")
    Xte.to_parquet(PROCESSED / "features_test.parquet")
    (PROCESSED / "feature_groups.json").write_text(json.dumps(groups(Xte.columns), indent=2))
    print(f"train {Xtr.shape}  test {Xte.shape}")
    for g, cols in groups(Xte.columns).items():
        print(f"  {g:14s} {len(cols):2d}  {', '.join(cols)}")
