"""Load and clean the raw competition files.

Every loader returns a tidy DataFrame with consistent types:
- text decoded to proper UTF-8 (account.csv is Latin-1; concerts files have a BOM / mojibake)
- seasons parsed to an integer start year (2013-2014 -> 2013)
- zip codes normalised to 5-character strings ("94501-3315" -> "94501", "7631" -> "07631")
- price levels converted to numbers, with text labels kept in a separate column

Run directly to write cleaned copies to data/processed/:  python3 src/clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "for_students"
PROCESSED = ROOT / "data" / "processed"

TARGET_YEAR = 2014  # the 2014-2015 season we are predicting


# ---------------------------------------------------------------- helpers
def season_year(s):
    """'2013-2014' -> 2013 (start year of the season)."""
    return s.str[:4].astype("Int64")


def zip5(s):
    """Normalise zip codes to 5 characters; returns NaN when no 5-digit zip can be formed."""
    raw = s.astype("string").str.strip()
    z = raw.str.split("-").str[0].str.replace(r"\D", "", regex=True)
    z = z.where(z.str.len() != 9, z.str[:5])                          # ZIP+4 without a hyphen
    z = z.where(~raw.str.contains("[A-Za-z]", na=False))               # Canadian / UK postcodes are not US zips
    z = z.where(z.str.len().between(3, 5)).str.zfill(5)
    return z


def fix_mojibake(s):
    """Repair Mac-Roman characters that were decoded as Latin-1 (e.g. 'RossiniÕs' -> 'Rossini's')."""
    def fix(t):
        if not isinstance(t, str):
            return t
        try:
            return t.encode("latin-1").decode("mac_roman")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return t
    return s.map(fix)


# ---------------------------------------------------------------- loaders
def load_train():
    return pd.read_csv(RAW / "train.csv")


def load_test():
    return pd.read_csv(RAW / "test.csv").rename(columns={"ID": "account.id"})


def load_account():
    a = pd.read_csv(RAW / "account.csv", encoding="latin-1", dtype={"shipping.zip.code": str, "billing.zip.code": str})
    a["billing.zip5"] = zip5(a["billing.zip.code"])
    a["shipping.zip5"] = zip5(a["shipping.zip.code"])
    a["first.donated"] = pd.to_datetime(a["first.donated"], errors="coerce")
    for c in ["billing.city", "shipping.city"]:
        a[c] = a[c].str.strip().str.title()
    a["relationship"] = a["relationship"].str.strip()
    return a


def load_subscriptions():
    s = pd.read_csv(RAW / "subscriptions.csv")
    s["year"] = season_year(s["season"])
    s["package_group"] = s["package"].str.extract(r"^(Full|Quartet|Trio|CYO)", expand=False).fillna("unknown")
    s["multiple.subs"] = s["multiple.subs"].eq("yes")
    return s


def load_tickets():
    t = pd.read_csv(RAW / "tickets_all.csv", dtype={"price.level": str})
    t["year"] = season_year(t["season"])
    pl = t["price.level"].str.strip()
    t["price.level.num"] = pd.to_numeric(pl, errors="coerce")          # '4.0' -> 4, 'Adult' -> NaN
    t["price.level.label"] = pl.where(t["price.level.num"].isna())     # keeps 'Adult' / 'Youth' / 'GA'
    t["set"] = t["set"].astype("Int64")
    t["multiple.tickets"] = t["multiple.tickets"].eq("yes")
    return t


def load_concerts():
    c = pd.read_csv(RAW / "concerts.csv", encoding="utf-8-sig")
    c["year"] = season_year(c["season"])
    for col in ["who", "what"]:
        c[col] = c[col].str.replace("\r", " | ", regex=False)
    return c


def load_concerts_2014():
    c = pd.read_csv(RAW / "concerts_2014-15.csv")
    for col in ["concert.name", "who", "what"]:
        c[col] = fix_mojibake(c[col])
    return c


def load_zipcodes():
    z = pd.read_csv(RAW / "zipcodes.csv")
    z["zip5"] = z["Zipcode"].astype(str).str.zfill(5)
    z["wages_per_return"] = z["TotalWages"] / z["TaxReturnsFiled"]
    return z.drop_duplicates("zip5")


LOADERS = {
    "train": load_train, "test": load_test, "account": load_account,
    "subscriptions": load_subscriptions, "tickets": load_tickets,
    "concerts": load_concerts, "concerts_2014": load_concerts_2014, "zipcodes": load_zipcodes,
}


def load_all():
    return {name: fn() for name, fn in LOADERS.items()}


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in load_all().items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name:14s} {df.shape}")
