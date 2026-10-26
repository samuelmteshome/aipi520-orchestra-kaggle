"""Build independent/llm_comparison.html: Claude vs GPT-6 Astra vs Gemini model pipelines, on common footing.

Common footing: every pipeline's final out-of-fold (OOF) predictions are re-scored here with the same code:
AUROC with a 95% bootstrap interval, AUROC on the hard group (accounts that did not subscribe in 2013-14),
average precision, and a paired bootstrap of the AUROC difference vs Claude's final model (same accounts).

    python3 independent/compare.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "independent" / "runs"
sys.path.insert(0, str(ROOT / "src"))
from train import _rank  # noqa: E402
from experiments import PRUNE  # noqa: E402

y = pd.read_csv(ROOT / "for_students/train.csv").set_index("account.id")["label"]
feat = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
hard = (feat.loc[y.index, "sub_last_season"] == 0).to_numpy()
rng = np.random.default_rng(0)
BOOT = 1000
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(BOOT)]

CATEGORIES = [   # (label, regex over feature names)
    ("Subscription history", r"^sub_"),
    ("Single-ticket history", r"^(tix_|tkt_)"),
    ("Any-purchase recency / frequency", r"^any_|purchase_recency|no_purchase|both_subscription"),
    ("Donations in 2013", r"don(ation|ated)?_2013"),
    ("Lifetime donations", r"don_count|don_lifetime|don_avg|pre14_|lifetime"),
    ("2014 first-gift flag", r"first_in_2014|first_donat|first\.donated"),
    ("Account-ID cohort", r"^id_|account_id|id_prefix|cohort"),
    ("Geography / distance", r"^geo_|zip|distance|_km|lat|long"),
    ("Interaction terms", r"^ix_|_and_|_x_|per_subscription"),
    ("Address-completeness flags", r"^has_|shipping|billing|account_record"),
    ("Concert programme text", r"concert|composer|programme|program"),
]


def oof_vector(path):
    o = pd.read_csv(path).set_index("account.id")["pred"]
    return o.reindex(y.index).to_numpy()


def metrics(p):
    yv = y.to_numpy()
    auc = roc_auc_score(yv, p)
    boots = [roc_auc_score(yv[i], p[i]) for i in boot_idx]
    return {"auroc": auc, "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "hard_auroc": roc_auc_score(yv[hard], p[hard]), "avg_precision": average_precision_score(yv, p)}


def paired_diff(p, ref):
    yv = y.to_numpy()
    d = [roc_auc_score(yv[i], p[i]) - roc_auc_score(yv[i], ref[i]) for i in boot_idx]
    return {"diff": roc_auc_score(yv, p) - roc_auc_score(yv, ref), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_better": float(np.mean(np.array(d) > 0))}


def stage_progress(run):
    """Best independently recomputed AUROC per stage from the driver log (last successful attempt)."""
    out = {}
    log = (run / "log.txt").read_text() if (run / "log.txt").exists() else ""
    for stage, rc, rec in re.findall(r"(\d_\w+): exit (\S+), [\d.]+ min, recomputed AUROC (\{.*?\})", log):
        vals = json.loads(rec.replace("'", '"')) if rec != "{}" else {}
        if rc == "0" and vals:
            out[stage[0]] = max(vals.values())
    return out


def categorise(features):
    return {lab: sorted(f for f in features if re.search(rx, f)) for lab, rx in CATEGORIES}


def process_stats(run):
    log = (run / "log.txt").read_text() if (run / "log.txt").exists() else ""
    tokens = sum(int(t) for t in re.findall(r"'total_tokens': (\d+)", log))
    return {"llm_calls": len(re.findall(r"reply from", log)),
            "script_runs": len(re.findall(r": exit ", log)),
            "failed_runs": len(re.findall(r": exit (?!0,)", log)),
            "rate_limit_errors": len(re.findall(r"gateway error", log)),
            "tokens": tokens,
            "served": json.loads((run / "served_models.json").read_text()) if (run / "served_models.json").exists() else []}


# ---------------------------------------------------------------- Claude (this project's pipeline)
ex = np.load(ROOT / "results/experiments_oof.npz")
# Repeat 1 only: a single out-of-fold pass like the other pipelines (averaging 5 repeats would itself be an ensemble;
# repeat 0 was used for tuning, so it would flatter Claude).
claude_oof = (_rank(ex["logistic pruned (a5)"][1]) + _rank(ex["forest pruned (a5)"][1])) / 2
cv = json.loads((ROOT / "results/cv_summary.json").read_text())
claude_feats = [c for c in cv["final_features"] if not PRUNE(c)]
P = {
    "Claude": {
        "label": "Claude (Opus 5.5, in-session)", "requested": "native", "final_model": "Rank-average of pruned logistic regression + pruned random forest",
        "oof": claude_oof, "features": claude_feats,
        "validation": "Repeated stratified 5×5 CV; tuned once on repeat 0 (mildly optimistic); per-fold feature selection; pruning chosen after seeing results.",
        "stages": {"2": 0.9825, "3": 0.9830, "4": 0.9835}, "public": 0.97679,
        "submission": ROOT / "submissions/exp_average_l_f_pruned.csv",
        "excluded": "Nothing excluded on leakage grounds (ID cohort, lifetime donations and the 2014 gift flag kept by your decision).",
        "process": {"llm_calls": None, "script_runs": None, "failed_runs": None, "rate_limit_errors": None, "tokens": None, "served": ["claude-opus-5-5"]},
    }
}

# ---------------------------------------------------------------- GPT-6 Astra
g = RUNS / "gpt_astra"
proto = json.loads((g / "4_combine_final/protocol.json").read_text())
P["GPT"] = {
    "label": "GPT-6 Astra (high)", "requested": "cx/gpt-6-astra-high", "final_model": "Single regularised histogram gradient boosting (7 leaves, min leaf 40, L2 25, 200 iterations)",
    "oof": oof_vector(g / "4_combine_final/oof_final_compact_hgb.csv"), "features": proto["features"],
    "validation": "Nested CV: 5 outer × 3 inner folds for tuning and pruning; final model re-validated on new outer folds.",
    "stages": stage_progress(g), "public": None, "submission": g / "4_combine_final/submission.csv",
    "excluded": "Deliberately excluded: account-ID cohort, lifetime donation totals, donation dates (incl. 2014 gifts), programme text, fine-grained zip/city.",
    "process": process_stats(g),
}

# ---------------------------------------------------------------- Gemini (3.8 Flash substituted for Argon)
gm = RUNS / "gemini_flash"
final_dir = gm / "4_combine_final"
gem = {"label": "Gemini 3.8 Flash (substituted for Gemini 4 Argon)", "requested": "gemini/gemini-3.8-flash",
       "stages": stage_progress(gm), "process": process_stats(gm), "public": None}
if (final_dir / "results.json").exists():
    r = json.loads((final_dir / "results.json").read_text())
    best = r.get("best")
    gem.update({"final_model": f"{best}: {r['models'].get(best, {}).get('notes', '')}", "validation": r.get("validation", ""),
                "oof": oof_vector(final_dir / f"oof_{best}.csv") if (final_dir / f"oof_{best}.csv").exists() else None,
                "submission": final_dir / "submission.csv"})
    fl = [f for f in final_dir.glob("*feature*") if f.suffix in (".json", ".csv")]
    feats = []
    for f in fl:
        try:
            d = json.loads(f.read_text()) if f.suffix == ".json" else pd.read_csv(f).iloc[:, 0].tolist()
            feats = d if isinstance(d, list) else d.get("features", feats)
        except Exception:
            pass
    gem["features"] = [str(x) for x in feats]
    gem["excluded"] = "See Gemini's final summary."
    gem["complete"] = True
else:
    gem.update({"final_model": "Not finished (Google rate limit)", "validation": "", "oof": None, "features": [],
                "submission": None, "excluded": "", "complete": False})
P["Gemini"] = gem

# ---------------------------------------------------------------- common-footing scoring
for k, v in P.items():
    v["metrics"] = metrics(v["oof"]) if v.get("oof") is not None else None
    v["vs_claude"] = paired_diff(v["oof"], claude_oof) if (k != "Claude" and v.get("oof") is not None) else None
    v["categories"] = categorise(v.get("features", []))
    v["n_features"] = len(v.get("features", []))
    fs = RUNS / k.lower() / "final_summary.md"

subs = {k: pd.read_csv(v["submission"]).set_index("ID")["Predicted"] for k, v in P.items()
        if v.get("submission") is not None and Path(v["submission"]).exists()}
names = list(subs)
corr = [[round(float(subs[a].corr(subs[b], method="spearman")), 3) for b in names] for a in names]
top = {k: set(s.sort_values(ascending=False).index[:150]) for k, s in subs.items()}
overlap = [[len(top[a] & top[b]) for b in names] for a in names]

summaries = {"GPT": (g / "final_summary.md").read_text() if (g / "final_summary.md").exists() else "",
             "Gemini": (gm / "final_summary.md").read_text() if (gm / "final_summary.md").exists() else ""}

out = {k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "submission")} for k, v in P.items()}
lp = ROOT / "independent/lessons.json"
data = {"P": out, "corr": {"names": names, "m": corr, "top150": overlap}, "summaries": summaries,
        "lessons": json.loads(lp.read_text()) if lp.exists() else [],
        "cats": [c for c, _ in CATEGORIES]}
html = (ROOT / "independent/compare_template.html").read_text()
css = re.search(r"<style>(.*?)</style>", (ROOT / "eda/report_template.html").read_text(), re.S).group(1)
(ROOT / "independent/llm_comparison.html").write_text(
    html.replace("/*__CSS__*/", css).replace("/*__DATA__*/null", json.dumps(data, default=float)))
print("wrote independent/llm_comparison.html")
for k, v in P.items():
    m = v["metrics"]
    print(k, "n/a" if m is None else f"AUROC {m['auroc']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}] hard {m['hard_auroc']:.4f} AP {m['avg_precision']:.3f}",
          "| vs Claude", v["vs_claude"])
