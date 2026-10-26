"""Exploratory data analysis for the orchestra subscription Kaggle competition.

Run from anywhere:  python3 eda/eda.py
Prints a text EDA to stdout and writes eda/data_summary.html (interactive charts).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "for_students"
OUT = Path(__file__).resolve().parent / "data_summary.html"


# ---------------------------------------------------------------- load
def read(name, enc="utf-8-sig"):
    return pd.read_csv(DATA / f"{name}.csv", encoding=enc, low_memory=False)


train = read("train")
test = read("test")
sample = read("sample_submission")
account = read("account", "latin-1")  # not valid UTF-8
subs = read("subscriptions")
tix = read("tickets_all")
concerts = read("concerts")
c1415 = read("concerts_2014-15")
zips = read("zipcodes")

files = {
    "train.csv": train, "test.csv": test, "sample_submission.csv": sample,
    "account.csv": account, "subscriptions.csv": subs, "tickets_all.csv": tix,
    "concerts.csv": concerts, "concerts_2014-15.csv": c1415, "zipcodes.csv": zips,
}

# ---------------------------------------------------------------- helpers
def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def rate_by(df, col, order=None):
    g = df.groupby(col, dropna=False)["label"].agg(["mean", "size"]).reset_index()
    g.columns = ["group", "rate", "n"]
    g["group"] = g["group"].astype(str)
    if order is not None:
        g = g.set_index("group").loc[order].reset_index()
    return g


def counts(s, top=None, dropna=False, ordinal=False):
    vc = s.value_counts(dropna=dropna)
    if ordinal:  # natural category order; text labels and "(missing)" last
        key = lambda v: (0, float(v)) if re.fullmatch(r"[\d.]+", str(v)) else (1, str(v))
        vc = vc.loc[sorted(vc.index, key=key)]
    if top:
        vc = vc.head(top)
    return {"x": [str(i) if pd.notna(i) else "(missing)" for i in vc.index], "y": vc.tolist()}


def nulls(df):
    p = (df.isna().mean() * 100).round(1)
    return {"x": p.index.tolist(), "y": p.tolist()}


def profile(df):
    return [
        {"col": c, "dtype": str(df[c].dtype), "null": round(df[c].isna().mean() * 100, 1),
         "nuniq": int(df[c].nunique()),
         "example": str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else ""}
        for c in df.columns
    ]


R = {}  # everything the HTML report needs

# ---------------------------------------------------------------- inventory
section("File inventory")
inv = []
for n, d in files.items():
    inv.append({"file": n, "rows": len(d), "cols": d.shape[1],
                "size_kb": round((DATA / n).stat().st_size / 1024)})
    print(f"{n:24s} {d.shape[0]:>7,} rows x {d.shape[1]:>2} cols")
R["inventory"] = inv

# ---------------------------------------------------------------- train / test
section("train.csv / test.csv")
pos = int(train.label.sum())
print(f"train: {len(train):,} accounts, {pos} positives ({pos / len(train):.2%})")
print(f"test : {len(test):,} accounts; overlap with train = "
      f"{len(set(train['account.id']) & set(test.ID))}")
R["kpi"] = {"train": len(train), "test": len(test), "pos": pos,
            "rate": round(pos / len(train) * 100, 1), "accounts": len(account)}

ids_tr, ids_te = set(train["account.id"]), set(test.ID)
cov = []
for n, d in [("account.csv", account), ("subscriptions.csv", subs), ("tickets_all.csv", tix)]:
    s = set(d["account.id"])
    cov.append({"file": n, "train": round(len(ids_tr & s) / len(ids_tr) * 100, 1),
                "test": round(len(ids_te & s) / len(ids_te) * 100, 1)})
    print(f"  in {n:18s}: train {cov[-1]['train']}%  test {cov[-1]['test']}%")
R["coverage"] = cov

# ---------------------------------------------------------------- account
section("account.csv")
R["account_profile"] = profile(account)
R["account_nulls"] = nulls(account)
don = account["amount.donated.lifetime"]
print(f"donors (lifetime > 0): {(don > 0).mean():.1%}")
print(account[["amount.donated.2013", "amount.donated.lifetime",
               "no.donations.lifetime"]].describe().round(1).to_string())
bins = [0, 1, 100, 500, 1000, 5000, 25000, np.inf]
labels = ["$0", "$1–99", "$100–499", "$500–999", "$1k–5k", "$5k–25k", "$25k+"]
R["don_hist"] = {"x": labels,
                 "y": pd.cut(don, bins, right=False, labels=labels).value_counts().loc[labels].tolist()}
R["ndon_hist"] = {"x": ["0", "1", "2", "3–5", "6–10", "11–20", "21+"],
                  "y": pd.cut(account["no.donations.lifetime"], [-1, 0, 1, 2, 5, 10, 20, 1000],
                              labels=["0", "1", "2", "3–5", "6–10", "11–20", "21+"]).value_counts()
                  .reindex(["0", "1", "2", "3–5", "6–10", "11–20", "21+"]).tolist()}
fd = pd.to_datetime(account["first.donated"], errors="coerce")
R["first_don_year"] = {"x": sorted(fd.dt.year.dropna().astype(int).unique().tolist()),
                       "y": fd.dt.year.dropna().astype(int).value_counts().sort_index().tolist()}
R["relationship"] = counts(account.relationship.dropna())
R["cities"] = counts(account["billing.city"], top=15, dropna=True)
print(f"first.donated range: {fd.min().date()} → {fd.max().date()}  "
      f"({(fd >= '2014-01-01').sum()} accounts first donated in 2014)")

# ---------------------------------------------------------------- subscriptions
section("subscriptions.csv")
R["subs_profile"] = profile(subs)
R["subs_nulls"] = nulls(subs)
print(f"{subs['account.id'].nunique():,} unique accounts, seasons "
      f"{subs.season.min()} → {subs.season.max()}; one row per account-season")
R["subs_season"] = {"x": sorted(subs.season.unique()), "y": subs.season.value_counts().sort_index().tolist()}
R["subs_season_acct"] = subs.groupby("season")["account.id"].nunique().reindex(R["subs_season"]["x"]).tolist()
R["subs_package"] = counts(subs.package)
R["subs_location"] = counts(subs.location)
R["subs_section"] = counts(subs.section)
R["subs_price"] = counts(subs["price.level"].map(lambda v: "(missing)" if pd.isna(v) else str(int(v))), ordinal=True)
R["subs_seats"] = counts(subs["no.seats"].clip(upper=6).map(lambda v: "6+" if v >= 6 else str(v)), ordinal=True)
R["subs_tier"] = counts(subs.subscription_tier.astype(str), top=10)
# package mix over time (top packages)
mix = pd.crosstab(subs.season, subs.package.fillna("(missing)"))
keep = ["Full", "Quartet", "Quartet CC", "Trio", "Trio A", "Trio B"]
mix["Other"] = mix.drop(columns=keep).sum(axis=1)
R["subs_mix"] = {"x": mix.index.tolist(), "series": {k: mix[k].tolist() for k in keep + ["Other"]}}
for c in ["package", "location", "section", "price.level"]:
    print(f"  {c}: {subs[c].value_counts(dropna=False).head(6).to_dict()}")

# ---------------------------------------------------------------- tickets
section("tickets_all.csv")
R["tix_profile"] = profile(tix)
R["tix_nulls"] = nulls(tix)
print(f"{tix['account.id'].nunique():,} unique accounts, seasons {sorted(tix.season.unique())}")
print("  price.level is mixed-type:", tix["price.level"].value_counts(dropna=False).to_dict())
R["tix_season"] = {"x": sorted(tix.season.unique()), "y": tix.season.value_counts().sort_index().tolist()}
R["tix_location"] = counts(tix.location)
R["tix_price"] = counts(tix["price.level"], ordinal=True)
R["tix_marketing"] = counts(tix["marketing.source"].dropna())
R["tix_seats"] = counts(tix["no.seats"].map(lambda v: "6+" if v >= 6 else str(v)), ordinal=True)
R["tix_per_acct"] = counts(tix.groupby("account.id").size().map(lambda v: "5+" if v >= 5 else str(v)).sort_values(), ordinal=True)

# ---------------------------------------------------------------- concerts
section("concerts.csv / concerts_2014-15.csv")
print(f"{concerts['concert.name'].nunique()} concerts, {len(concerts)} performances, "
      f"seasons {sorted(concerts.season.unique())}")
ct = pd.crosstab(concerts.season, concerts.location.fillna("(missing)"))
R["conc_season_loc"] = {"x": ct.index.tolist(), "series": {c: ct[c].tolist() for c in ct.columns}}


def composers(text):
    """Composer surnames: ALL-CAPS runs in the free-text programme (e.g. 'MOZART:', 'HANDELArias')."""
    out = set()
    for m in re.finditer(r"[A-Z][A-Z\.]{3,}", str(text)):
        w = m.group()
        if re.match(r"\s[A-Z]{2}", str(text)[m.end():m.end() + 3]) or w == "INTERMISSION":
            continue  # first name of 'ANTONIO VIVALDI', or a heading
        if m.end() < len(str(text)) and str(text)[m.end()].islower():
            w = w[:-1]  # 'HANDELArias' -> 'HANDEL'
        w = w.split(".")[-1]
        if len(w) >= 4 and not set(w) <= set("IVXLC"):
            out.add(w.title())
    return out


comp_hist = pd.Series([c for w in concerts.drop_duplicates("concert.name").what for c in composers(w)])
R["composers_hist"] = counts(comp_hist, top=12)
c1415_out = c1415.copy()
c1415_out["composers"] = c1415_out.what.map(lambda w: ", ".join(sorted(composers(w))))
R["c1415"] = c1415_out[["set", "concert.name", "who", "composers"]].to_dict("records")
R["concerts_unique"] = (concerts.drop_duplicates("concert.name")[["season", "set", "concert.name"]]
                        .to_dict("records"))
print("2014-15 sets:", c1415["concert.name"].tolist())

# ---------------------------------------------------------------- zipcodes
section("zipcodes.csv")
R["zip_profile"] = profile(zips)
zips["zip5"] = zips.Zipcode.astype(str).str.zfill(5)
acc_zip = account["billing.zip.code"].dropna().astype(str).str.split("-").str[0].str.zfill(5)
match = acc_zip.isin(set(zips.zip5)).mean()
print(f"billing zips present: {account['billing.zip.code'].notna().mean():.1%} of accounts; "
      f"{match:.1%} of those match zipcodes.csv")
R["zip_match"] = round(match * 100, 1)
R["zip_type"] = counts(zips.ZipCodeType)
R["zip_states"] = counts(acc_zip.map(zips.set_index("zip5").State), top=8)

# ---------------------------------------------------------------- label vs. features (train only)
section("Label rate by candidate signal (train only)")
d = train.merge(account, on="account.id", how="left")
sub_ids = lambda season: set(subs.loc[subs.season == season, "account.id"])
nseas = subs.groupby("account.id").season.nunique()
last = subs.groupby("account.id").season.max()
d["n_seasons"] = d["account.id"].map(nseas).fillna(0)
d["last_season"] = d["account.id"].map(last).fillna("never")
flags = {
    "Subscribed in 2013-14": d["account.id"].isin(sub_ids("2013-2014")),
    "Donated in 2013": d["amount.donated.2013"] > 0,
    "Ever donated": d["amount.donated.lifetime"] > 0,
    "Ever subscribed": d.n_seasons > 0,
    "Bought single tickets (2010-14)": d["account.id"].isin(set(tix["account.id"])),
    "Has billing zip": d["billing.zip.code"].notna(),
    "Has a 'relationship' tag": d.relationship.notna(),
}
sig = []
for name, f in flags.items():
    y, n = d.loc[f, "label"], d.loc[~f, "label"]
    sig.append({"signal": name, "yes": round(y.mean() * 100, 1), "no": round(n.mean() * 100, 1),
                "n_yes": int(f.sum()), "n_no": int((~f).sum())})
    print(f"  {name:34s} yes {y.mean():6.1%} (n={f.sum():>5})   no {n.mean():6.1%} (n={(~f).sum():>5})")
R["signals"] = sig

nb = pd.cut(d.n_seasons, [-1, 0, 1, 3, 6, 10, 15, 25], labels=["0", "1", "2–3", "4–6", "7–10", "11–15", "16–21"])
g = d.groupby(nb, observed=False).label.agg(["mean", "size"])
R["rate_nseasons"] = {"x": g.index.astype(str).tolist(), "rate": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}
print("\n  by # seasons subscribed:\n", g.round(3).to_string())

lb = d.last_season.where(d.last_season.isin(["2009-2010", "2010-2011", "2011-2012", "2012-2013", "2013-2014", "never"]),
                         "≤2008-09")
order = ["never", "≤2008-09", "2009-2010", "2010-2011", "2011-2012", "2012-2013", "2013-2014"]
g = d.groupby(lb).label.agg(["mean", "size"]).reindex(order)
R["rate_last"] = {"x": order, "rate": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}
print("\n  by most recent season subscribed:\n", g.round(3).to_string())

db = pd.cut(d["amount.donated.lifetime"], bins, right=False, labels=labels)
g = d.groupby(db, observed=False).label.agg(["mean", "size"])
R["rate_don"] = {"x": labels, "rate": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}

d["region"] = d["billing.city"].where(d["billing.city"].isin(account["billing.city"].value_counts().head(10).index), "Other / missing")
g = d.groupby("region").label.agg(["mean", "size"]).sort_values("size", ascending=False)
R["rate_city"] = {"x": g.index.tolist(), "rate": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}

# patron geography (train+test accounts, by billing zip → lat/long)
zl = zips.drop_duplicates("zip5").set_index("zip5")[["Lat", "Long"]]
acc = account[["account.id", "billing.zip.code"]].dropna().copy()
acc["zip5"] = acc["billing.zip.code"].astype(str).str.split("-").str[0].str.zfill(5)
geo = train.merge(acc, on="account.id").join(zl, on="zip5").dropna(subset=["Lat"])
geo = geo[(geo.Lat.between(36.5, 39)) & (geo.Long.between(-123.5, -121))]
gz = geo.groupby("zip5").agg(lat=("Lat", "first"), lon=("Long", "first"), n=("label", "size"), rate=("label", "mean"))
R["geo"] = {"lat": gz.lat.tolist(), "lon": gz.lon.tolist(), "n": gz.n.tolist(),
            "rate": (gz.rate * 100).round(1).tolist(), "zip": gz.index.tolist()}

# ---------------------------------------------------------------- write report
html = (Path(__file__).resolve().parent / "report_template.html").read_text()
OUT.write_text(html.replace("/*__DATA__*/null", json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
print(f"\nWrote {OUT.relative_to(ROOT)}")
