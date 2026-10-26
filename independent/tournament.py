"""Blind peer-review modelling tournament between three external LLM personas, run through the local OmniRoute gateway.

Phases (strictly sequential; every step is checkpointed, so the script can be re-run to resume):
  1  isolated proposals (explain the current best model + propose improvements), locked
  2  round-1 presentations and dual critiques (order A, B, C), shared with all teams
  3  synthesis 1, then round-2 presentations and dual critiques on the upgraded plans
  4  synthesis 2 -> final architecture + documentation; each team writes ONE script that the operator runs
     (a) on the development data, where a 20% LOCKBOX of training accounts is hidden and scored once, and
     (b) on the full data for a Kaggle submission; results are packaged anonymously in final_analysis/
  5  three fresh zero-context evaluators score the anonymous packages and vote; votes are tallied, identities revealed

The operator (this script) relays messages, runs code and reports raw outcomes. It never adds modelling advice.
Teams are labelled A/B/C inside prompts so brand names cannot bias critiques.

    python3 independent/tournament.py
"""
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "independent"))
import driver as D  # noqa: E402  (gateway chat, code extraction and safety checks)

RAW = ROOT / "for_students"
RUN = ROOT / "independent" / "runs" / "tournament"
OUT = ROOT / "final_analysis"
STATE = RUN / "state.json"
TEAMS = {"A": ("Gemini", "antigravity/gemini-3.8-flash-tiered"),
         "B": ("GPT", "cx/gpt-6-astra-high"),
         "C": ("Claude", "antigravity/claude-opus-4-6-thinking")}
ORDER = ["A", "B", "C"]
LOCKBOX_FRAC, LOCKBOX_SEED = 0.20, 2026
SCRIPT_TIMEOUT = 25 * 60
MAX_FIX = 2
IDENTITY = re.compile(r"\b(gpt[-\w.]*|openai|chatgpt|gemini[-\w.]*|google|anthropic|claude[-\w.]*|astra|opus|sonnet|"
                      r"flash|deepmind|codex|team\s+[abc])\b", re.I)


# ---------------------------------------------------------------- state / logging
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"threads": {t: [] for t in TEAMS}, "pending": {t: "" for t in TEAMS}, "done": {}}


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(RUN / "log.txt", "a") as f:
        f.write(line + "\n")


def ask(st, team, content, key):
    """One turn in a team's own conversation; shared records queued for that team are prepended."""
    if key in st["done"]:
        return st["done"][key]
    msg = (st["pending"][team] + "\n\n" + content).strip() if st["pending"][team] else content
    st["threads"][team].append({"role": "user", "content": msg})
    log(f"{key}: asking team {team}")
    reply, served = D.chat(TEAMS[team][1], [{"role": "system", "content": SYSTEM.format(team=team)}] + st["threads"][team], log)
    st["threads"][team].append({"role": "assistant", "content": reply})
    st["pending"][team] = ""
    st["done"][key] = reply
    st.setdefault("served", {}).setdefault(team, [])
    if served and served not in st["served"][team]:
        st["served"][team].append(served)
    save(st)
    return reply


def share(st, text, teams=TEAMS):
    for t in teams:
        st["pending"][t] += ("\n\n" if st["pending"][t] else "") + text
    save(st)


# ---------------------------------------------------------------- data: lockbox + per-run data dirs
def prepare_data():
    if (RUN / "lockbox_ids.csv").exists():
        return pd.read_csv(RUN / "lockbox_ids.csv")["account.id"].tolist()
    tr = pd.read_csv(RAW / "train.csv")
    recent = set(pd.read_csv(RAW / "subscriptions.csv").query("season == '2013-2014'")["account.id"])
    strata = tr["label"].astype(str) + tr["account.id"].isin(recent).astype(int).astype(str)
    dev, lock = train_test_split(tr, test_size=LOCKBOX_FRAC, stratify=strata, random_state=LOCKBOX_SEED)
    lock[["account.id"]].to_csv(RUN / "lockbox_ids.csv", index=False)
    for name, train_df, test_ids in [("data_dev", dev, lock["account.id"]), ("data_full", tr, None)]:
        d = RUN / name
        shutil.copytree(RAW, d, dirs_exist_ok=True)
        if test_ids is not None:   # the lockbox plays the role of test.csv; its labels never enter this directory
            train_df.to_csv(d / "train.csv", index=False)
            pd.DataFrame({"ID": test_ids}).to_csv(d / "test.csv", index=False)
            pd.DataFrame({"ID": test_ids, "Predicted": 0.5}).to_csv(d / "sample_submission.csv", index=False)
        src = RUN / f"src_{name[5:]}"
        shutil.copytree(ROOT / "src", src, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "experiments*", "train.py", "analysis.py"))
        c = (src / "clean.py").read_text()
        c = c.replace('RAW = ROOT / "for_students"', 'import os\nRAW = Path(os.environ["DATA_DIR"])')
        c = c.replace('PROCESSED = ROOT / "data" / "processed"', 'PROCESSED = Path(os.environ["OUT_DIR"]) / "cache"')
        (src / "clean.py").write_text(c)
    log(f"lockbox: {len(lock)} accounts ({int(lock['label'].sum())} positives) hidden from all teams")
    return lock["account.id"].tolist()


def baseline_on_lockbox(lock_ids):
    """Reference: the current best model (pruned logistic + forest) trained on the development rows only."""
    sys.path.insert(0, str(ROOT / "src"))
    from models import Logistic, Forest, FoldFeatures
    from experiments import PRUNE
    from train import _rank
    best = json.loads((ROOT / "results/cv_summary.json").read_text())["best_params"]
    X = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    y = X.pop("label")
    X = X[[c for c in X.columns if not c.startswith(("cons_", "hx_", "geo2_", "hg_"))]]
    lock = X.index.isin(lock_ids)
    ff = FoldFeatures(use_id=True, seed=0).fit(X[~lock], y[~lock])
    keep = [c for c in ff.columns_ if not PRUNE(c)]
    a, b = ff.transform(X[~lock])[keep], ff.transform(X[lock])[keep]
    p = np.mean([_rank(m.fit(a, y[~lock]).predict_proba(b)) for m in (Logistic(**best["logistic"]), Forest(**best["forest"]))], axis=0)
    return lockbox_metrics(pd.Series(p, index=X.index[lock]))


def lockbox_metrics(pred):
    y = pd.read_csv(RAW / "train.csv").set_index("account.id")["label"]
    recent = set(pd.read_csv(RAW / "subscriptions.csv").query("season == '2013-2014'")["account.id"])
    j = y.to_frame().join(pred.rename("p"), how="inner")
    yv, pv = j["label"].to_numpy(), j["p"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(yv[i], pv[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000)) if yv[i].min() != yv[i].max()]
    h = ~j.index.isin(recent)
    return {"lockbox_auroc": round(float(roc_auc_score(yv, pv)), 5), "lockbox_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "lockbox_hard_group_auroc": round(float(roc_auc_score(yv[h], pv[h])), 4), "lockbox_n": int(len(yv)), "lockbox_positives": int(yv.sum())}


# ---------------------------------------------------------------- prompts
SYSTEM = ("You are Team {team}, an expert data scientist in a three-team modelling review for a Kaggle-style binary classification "
          "task. Teams are labelled A, B and C. A neutral operator relays messages between teams and runs code on the private data; "
          "the operator gives no modelling advice. Be evidence-based, specific and honest about uncertainty. The overriding goal is "
          "better performance on UNSEEN data, so treat every source of overfitting (to training data, to validation folds, or to "
          "leaderboard feedback) as a defect.")


def dossier():
    task = (ROOT / "independent/context/kaggle_task.md").read_text()
    data = (ROOT / "eda/external/prompt_and_dossier.md").read_text().split("# DATA DOSSIER", 1)[1]
    code = "\n\n".join(f"### src/{f}\n```python\n{(ROOT / 'src' / f).read_text()}\n```" for f in ["clean.py", "features.py", "select_features.py", "models.py"])
    facts = """## CURRENT MOST ACCURATE MODEL (baseline) — facts measured by the operator
- Model: rank-average of (1) L2 logistic regression (C≈74, standardised, log1p on skewed columns, median impute + missing flags) and
  (2) random forest (500 trees, max_depth 4, min_samples_leaf 2, max_features 0.4, class_weight balanced). Probabilities from each are
  converted to ranks and averaged.
- Features: built by src/features.py (subscription history, single tickets, any-purchase recency, donations, geography, account-ID
  cohort). Inside each CV training fold, src/select_features.py picks 25 (drop >50% missing → mutual information + |Spearman|≥0.95
  de-duplication → random-forest importance; the two account-ID cohort features are always kept). Then 8 are removed by a fixed prune
  list chosen after inspecting permutation importance: all tix_* and ix_* features and 3 of 4 distance features → 17 features.
- Validation: repeated stratified 5-fold × 5 repeats (strata = label × subscribed-in-2013-14). Hyperparameters tuned by random search on
  repeat 0 only, then frozen. Pooled out-of-fold AUROC 0.9835; public leaderboard 0.97785 (best of all submissions).
- Other public leaderboard results: forest alone 0.97679 (CV 0.9801); logistic+forest+MLP pruned 0.97615 (CV 0.9832); logistic+forest+MLP
  unpruned 0.97338 (CV 0.9825); PyTorch MLP ensemble 0.97153 (CV 0.9797); logistic alone 0.97107 (CV 0.9821). A separate pipeline using
  fully nested CV (regularised gradient boosting, no account-ID or donation-timing features): CV 0.9728, public 0.97203.
- Diagnostic: tune-once pipelines dropped 0.003–0.011 (mean ≈0.007) from CV to public; the nested-CV pipeline dropped 0.001.
- Error decomposition: 82% of misordered positive/negative pairs involve the 50 positives who did NOT subscribe in 2013-14 ("hard group",
  6,596 accounts) ranked below other non-subscribers; within-hard-group AUROC ≈0.90–0.92. 282 of 345 2013-14 subscribers renewed.
- Experiments already run (same 25 folds, paired): adding the MLP to the blend +0.000; stronger regularisation (forest depth 3, leaf 10,
  logistic C=1/10) neutral or worse; three explicit interaction terms −0.0004; attendance-consistency features (share of seasons active,
  longest run, lapses, Full-package share, venue loyalty, donations per year) +0.0003 (logistic +0.0008, forest −0.0002); separate
  hard-group segment model −0.0015 overall / −0.010 hard group; forcing in distance-to-last-venue, income percentile, single tickets in
  2013-14 and composer-overlap features −0.0023; removing the two account-ID features −0.005 to −0.010. Train-vs-validation AUROC gaps
  0.004–0.009. Learning curve still rising at 100% of training rows.
- Student decisions (may be questioned, but the student has the final say): keep the account-ID cohort features; keep a "first gift in
  2014" flag (timing risk); prefer about 25 features or fewer.

## ANTI-OVERFITTING RULES OF THIS REVIEW
- A random 20% of training accounts (stratified) is a LOCKBOX hidden from every team. Each team's final script is run once by the
  operator with the lockbox removed from train.csv and placed in test.csv; its lockbox AUROC is the main unbiased score. Teams never see it.
- Any selection or tuning must happen inside the training data the script receives (nested where applicable). No leaderboard probing.
"""
    return f"# TASK BRIEF\n{task}\n\n# DATA DOSSIER\n{data}\n\n{facts}\n\n# BASELINE CODE (read-only reference; importable later)\n{code}"


ENV = """EXECUTION CONTRACT for your implementation script (the operator runs it exactly as written, twice):
- Run 1 (development): DATA_DIR train.csv EXCLUDES the hidden lockbox accounts; DATA_DIR test.csv contains the lockbox IDs.
  Run 2 (full): DATA_DIR is the real competition data (full train.csv, real test.csv). Your script must not care which run it is.
- Read os.environ["DATA_DIR"] (raw files as in the dossier) and write ONLY inside os.environ["OUT_DIR"].
- The baseline modules are importable: sys.path.insert(0, os.environ["BASELINE_SRC"]) then e.g. `from features import build` /
  `from clean import load_all`. In this copy, clean.py reads DATA_DIR and caches into OUT_DIR, so it is safe to use.
- Python 3.14, CPU only (8 cores, 16 GB), 25-minute hard limit. Libraries: numpy, pandas 3, scipy, scikit-learn 1.9, torch 2.13 (CPU),
  pyarrow. No installs, network, subprocess or shell. Set seeds.
- Required outputs in OUT_DIR:
  * results.json: {"models": {"<name>": {"cv_auroc": float, "notes": str}}, "best": "<final model name>", "validation": "<one line>"}
  * oof_final.csv: account.id,pred — out-of-fold prediction of your FINAL model for every account in DATA_DIR/train.csv
  * submission.csv: ID,Predicted for every ID in DATA_DIR/test.csv (same format as sample_submission.csv)
  * diagnostics.json: anything you use to show you are not overfitting (e.g. train vs CV AUROC, fold spread, nested scores)
REPLY FORMAT: a short rationale (≤200 words), then exactly ONE ```python block with the complete script."""

DOC_TEMPLATE = """Then, after a line containing only ===DOCUMENTATION===, write package documentation in this exact neutral template. Third person,
plain prose and bullets, no self-reference, no team letters, no model or company names:
# Methodology
## Objective and summary
## Data pipeline and feature engineering
## Models and hyperparameters
## Validation protocol
## Anti-overfitting measures
## Known limitations"""


# ---------------------------------------------------------------- code execution
def run_script(code, stage_dir, data_dir, src_dir):
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "script.py").write_text(code)
    env = {"DATA_DIR": str(data_dir), "OUT_DIR": str(stage_dir), "BASELINE_SRC": str(src_dir), "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
           "HOME": str(stage_dir), "OMP_NUM_THREADS": "8", "PYTHONHASHSEED": "0"}
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", str(stage_dir / "script.py")], cwd=stage_dir, env=env, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = "timeout", "", "TIMEOUT after 25 min"
    (stage_dir / "stdout.txt").write_text(out or "")
    (stage_dir / "stderr.txt").write_text(err or "")
    return rc, out or "", err or "", time.time() - t


def check_outputs(stage_dir, data_dir):
    notes, rec = [], None
    try:
        results = json.loads((stage_dir / "results.json").read_text())
    except Exception:
        results = None
        notes.append("results.json missing or invalid")
    y = pd.read_csv(data_dir / "train.csv").set_index("account.id")["label"]
    try:
        o = pd.read_csv(stage_dir / "oof_final.csv").set_index("account.id")["pred"].reindex(y.index)
        if o.isna().any():
            notes.append(f"oof_final.csv missing {int(o.isna().sum())} train accounts")
        m = o.notna()
        rec = round(float(roc_auc_score(y[m], o[m])), 5)
    except Exception as e:
        notes.append(f"oof_final.csv unreadable ({e})")
    try:
        s = pd.read_csv(stage_dir / "submission.csv")
        ids = pd.read_csv(data_dir / "test.csv")["ID"]
        if list(s.columns) != ["ID", "Predicted"] or set(s["ID"]) != set(ids) or not np.isfinite(s["Predicted"]).all():
            notes.append("submission.csv does not match the required format/IDs or has NaN")
    except Exception as e:
        notes.append(f"submission.csv unreadable ({e})")
    return results, rec, notes


# ---------------------------------------------------------------- phases
def phase1(st):
    base = dossier()
    ask_text = base + """

PHASE 1 — INDEPENDENT PROPOSAL (you have not seen any other team's ideas). Write, in Markdown, ≤1,300 words:
1. Your explanation of the methodology of the current most accurate baseline model: what it does, why it works, its weaknesses.
2. An evidence-based proposal to improve modelling and feature engineering WITHOUT overfitting: concrete changes, why each should
   generalise, how each will be validated, and which experiments above it takes into account. Number your proposals.
This proposal will be locked before peer review."""
    for t in ORDER:
        ask(st, t, ask_text, f"p1_{t}")
    st["plans"] = {t: st["done"][f"p1_{t}"] for t in ORDER}
    save(st)


def critique_round(st, rnd):
    plans = st["plans"] if rnd == 1 else st["plans_s1"]
    for p in ORDER:
        critics = [c for c in ORDER if c != p]
        crit = {}
        for c in critics:
            crit[c] = ask(st, c, f"""ROUND {rnd} — Team {p} presents its {'locked Phase 1' if rnd == 1 else 'upgraded'} plan:

<plan team="{p}">
{plans[p]}
</plan>

Critique this plan independently (≤500 words): its strongest points, flaws, every overfitting or leakage risk you see, and concrete,
testable changes you would make. Be specific.""", f"r{rnd}_{p}_by_{c}")
        record = f"ROUND {rnd} RECORD — Team {p}'s plan and its two critiques (shared with all teams):\n<plan team=\"{p}\">\n{plans[p]}\n</plan>\n" + \
                 "\n".join(f"<critique by=\"Team {c}\">\n{crit[c]}\n</critique>" for c in critics)
        if f"shared_r{rnd}_{p}" not in st["done"]:
            share(st, record)
            st["done"][f"shared_r{rnd}_{p}"] = True
            save(st)


def synthesis1(st):
    for t in ORDER:
        ask(st, t, """SYNTHESIS 1 — Using all Round 1 critiques and insights (including those of other teams' plans), upgrade YOUR plan.
Output the complete upgraded plan in Markdown (≤1,300 words). State what you changed and why, and what you deliberately did not change.""", f"s1_{t}")
    st["plans_s1"] = {t: st["done"][f"s1_{t}"] for t in ORDER}
    save(st)


def synthesis2(st):
    for t in ORDER:
        ask(st, t, f"""SYNTHESIS 2 — Using all Round 2 feedback, finalise your modelling and feature-engineering architecture. Be exact:
feature list or construction rules, models and hyperparameters (fixed in advance, or tuned only inside inner CV of the training data),
validation protocol, ensembling, and anti-overfitting measures. Write it in Markdown (≤1,200 words).
{DOC_TEMPLATE}""", f"s2_{t}")
    save(st)


def implement(st, lock_ids):
    for t in ORDER:
        tdir = RUN / f"impl_{t}"
        if st["done"].get(f"impl_{t}_final"):
            continue
        prompt = f"""PHASE 4 — IMPLEMENTATION. Implement your finalised architecture as ONE self-contained script.
{ENV}"""
        ok = False
        for attempt in range(MAX_FIX + 1):
            reply = ask(st, t, prompt, f"impl_{t}_{attempt}")
            code = D.extract_code(reply)
            if code is None or D.check_code(code):
                prompt = f"OPERATOR REPORT: script not run ({'no python block' if code is None else 'disallowed operations ' + str(D.check_code(code))}). Resend the complete script."
                continue
            rc, out, err, secs = run_script(code, tdir / "dev", RUN / "data_dev", RUN / "src_dev")
            results, rec, notes = check_outputs(tdir / "dev", RUN / "data_dev")
            log(f"team {t} dev run: exit {rc}, {secs / 60:.1f} min, recomputed OOF AUROC {rec}, issues {notes}")
            if rc == 0 and results and rec is not None and not notes:
                ok = True
                share(st, f"OPERATOR REPORT (development run, raw outcome): exit 0, {secs / 60:.1f} min. Your results.json: {json.dumps(results)}. "
                          f"Operator-recomputed out-of-fold AUROC of oof_final.csv: {rec}. The lockbox score is withheld.", [t])
                break
            prompt = (f"OPERATOR REPORT (development run): exit {rc}, {secs / 60:.1f} min. Issues: {notes}.\nprinted output (tail):\n```\n{out[-4000:]}\n```\n"
                      f"error output (tail):\n```\n{err[-3000:]}\n```\nFix the script and resend the complete script.")
        if not ok:
            st["done"][f"impl_{t}_final"] = {"failed": True}
            save(st)
            continue
        dev_pred = pd.read_csv(tdir / "dev/submission.csv").set_index("ID")["Predicted"]
        lb = lockbox_metrics(dev_pred)
        rc, out, err, secs = run_script(code, tdir / "full", RUN / "data_full", RUN / "src_full")
        _, rec_full, notes_full = check_outputs(tdir / "full", RUN / "data_full")
        log(f"team {t} lockbox AUROC {lb['lockbox_auroc']} {lb['lockbox_ci95']}; full run exit {rc}, OOF {rec_full}, issues {notes_full}")
        st["done"][f"impl_{t}_final"] = {"failed": False, "code": code, "dev_oof_auroc": rec, "full_oof_auroc": rec_full, "full_ok": rc == 0 and not notes_full,
                                         "lockbox": lb, "results": json.loads((tdir / "dev/results.json").read_text())}
        save(st)


def scrub(text):
    text = IDENTITY.sub("[redacted]", text)
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def package(st, base_lb):
    if "mapping" not in st:
        nums = [1, 2, 3]
        random.Random(int(time.time())).shuffle(nums)
        st["mapping"] = {t: n for t, n in zip(ORDER, nums)}
        save(st)
    OUT.mkdir(exist_ok=True)
    (OUT / "README.md").write_text(f"""# Final analysis: blind model comparison

## Goal
Predict, for each account in the competition test set, the probability that the patron buys a 2014-15 orchestra season subscription.
Three modelling packages were developed independently, each after two rounds of structured peer critique. This folder lets a reviewer
compare them without knowing who built which.

## Evaluation metric
Area under the ROC curve (AUROC) on predicted probabilities. Higher is better; 0.5 is random.

## Anti-overfitting constraints every package followed
- A stratified 20% of the 6,941 labelled training accounts ({base_lb['lockbox_n']} accounts, {base_lb['lockbox_positives']} positives) was held out as a
  lockbox before any package was designed. No package's code could train, select features or tune on it. Each final script was
  run once with the lockbox removed; its lockbox AUROC is the main unbiased performance estimate.
- Feature selection and hyperparameter tuning had to happen inside the training data each script received.
- No package saw leaderboard feedback or any other package's results.
- Each script was then re-run unchanged on the full training data to produce a competition submission.

## Reference point
The pre-existing best model (rank-average of a pruned logistic regression and a shallow random forest), trained on the same
non-lockbox rows: lockbox AUROC {base_lb['lockbox_auroc']} (95% bootstrap CI {base_lb['lockbox_ci95'][0]}–{base_lb['lockbox_ci95'][1]}),
hard-group lockbox AUROC {base_lb['lockbox_hard_group_auroc']} (accounts that did not subscribe in 2013-14).
Caution: the reference model was designed before the lockbox existed, using all training rows (its feature pruning and tuning saw the
lockbox accounts), so its lockbox score is optimistic. The three packages had no such access. The lockbox holds only
{base_lb['lockbox_positives']} positives, so differences of about 0.01 are within noise.

## Package contents
Each `Model_Package_N/` holds METHODOLOGY.md, model_code.py, results.json, diagnostics.json and VALIDATION.md (metrics computed the
same way for every package by the operator, including the lockbox score and bootstrap interval).
""")
    for t in ORDER:
        n, imp = st["mapping"][t], st["done"].get(f"impl_{t}_final", {})
        pdir = OUT / f"Model_Package_{n}"
        pdir.mkdir(exist_ok=True)
        doc = st["done"][f"s2_{t}"].split("===DOCUMENTATION===", 1)
        (pdir / "METHODOLOGY.md").write_text(scrub(doc[1] if len(doc) > 1 else doc[0]).strip() + "\n")
        if imp.get("failed", True):
            (pdir / "VALIDATION.md").write_text("# Validation\nThe implementation did not run successfully within the allowed fix attempts.\n")
            continue
        (pdir / "model_code.py").write_text(scrub(imp["code"]))
        for f in ["results.json", "diagnostics.json"]:
            src = RUN / f"impl_{t}" / "dev" / f
            if src.exists():
                (pdir / f).write_text(scrub(src.read_text()))
        lb = imp["lockbox"]
        (pdir / "VALIDATION.md").write_text(f"""# Validation (computed identically for every package by the operator)
| Metric | Value |
|---|---|
| Own cross-validation AUROC (reported in results.json, development rows) | {imp['results']['models'].get(imp['results'].get('best'), {}).get('cv_auroc', 'n/a')} |
| Operator-recomputed out-of-fold AUROC (development rows) | {imp['dev_oof_auroc']} |
| **Lockbox AUROC** (never seen during development) | **{lb['lockbox_auroc']}** |
| Lockbox 95% bootstrap interval | {lb['lockbox_ci95'][0]} – {lb['lockbox_ci95'][1]} |
| Lockbox AUROC, hard group (no 2013-14 subscription) | {lb['lockbox_hard_group_auroc']} |
| Lockbox size / positives | {lb['lockbox_n']} / {lb['lockbox_positives']} |
| Out-of-fold AUROC on full training data (submission run) | {imp['full_oof_auroc']} |
| Reference model lockbox AUROC | {base_lb['lockbox_auroc']} |

Validation scheme as stated by the package: {scrub(imp['results'].get('validation', ''))}
""")
        if imp.get("full_ok"):
            shutil.copy(RUN / f"impl_{t}" / "full" / "submission.csv", ROOT / "submissions" / f"tournament_package_{n}.csv")


def evaluate(st):
    blob = "# README.md\n" + (OUT / "README.md").read_text()
    for n in [1, 2, 3]:
        pdir = OUT / f"Model_Package_{n}"
        blob += f"\n\n# ===== Model_Package_{n} =====\n"
        for f in ["METHODOLOGY.md", "VALIDATION.md", "results.json", "diagnostics.json", "model_code.py"]:
            if (pdir / f).exists():
                body = (pdir / f).read_text()
                blob += f"\n## {f}\n```\n{body[:30000]}\n```\n"
    prompt = blob + """

You are an independent reviewer with no prior involvement. Read the README and inspect the three packages. Score each package from 1 to 10
on (a) methodology, (b) generalisation / anti-overfitting rigour, (c) validation performance (weight the lockbox result most, but account
for its bootstrap uncertainty). Then vote for the single most accurate and robust model.
Reply with ONLY a JSON object:
{"scores": {"1": {"methodology": int, "generalisation": int, "validation": int, "comment": str}, "2": {...}, "3": {...}},
 "vote": 1|2|3, "rationale": str}"""
    for t in ORDER:
        key = f"eval_{t}"
        if key in st["done"]:
            continue
        log(f"blind evaluation by a fresh {TEAMS[t][0]} instance")
        reply, served = D.chat(TEAMS[t][1], [{"role": "user", "content": prompt}], log)
        m = re.search(r"\{.*\}", reply, re.S)
        try:
            st["done"][key] = {"served": served, "verdict": json.loads(m.group(0)) if m else None, "raw": reply}
        except json.JSONDecodeError:
            st["done"][key] = {"served": served, "verdict": None, "raw": reply}
        save(st)


def tally(st, base_lb):
    inv = {n: t for t, n in st["mapping"].items()}
    votes = {n: 0 for n in [1, 2, 3]}
    rows = []
    for t in ORDER:
        v = (st["done"].get(f"eval_{t}") or {}).get("verdict") or {}
        if v.get("vote") in (1, 2, 3, "1", "2", "3"):
            votes[int(v["vote"])] += 1
        rows.append({"evaluator": TEAMS[t][0], "vote": v.get("vote"), "scores": v.get("scores"), "rationale": v.get("rationale")})
    winner = max(votes, key=votes.get)
    summary = {"votes": votes, "winner_package": winner, "winner_persona": TEAMS[inv[winner]][0],
               "mapping": {f"Model_Package_{n}": f"{TEAMS[t][0]} ({TEAMS[t][1]}; served as {st.get('served', {}).get(t)})" for t, n in st["mapping"].items()},
               "lockbox": {f"Model_Package_{n}": (st["done"].get(f"impl_{t}_final") or {}).get("lockbox") for t, n in st["mapping"].items()},
               "reference_lockbox": base_lb, "evaluations": rows}
    (RUN / "results_reveal.json").write_text(json.dumps(summary, indent=1))
    log(f"votes {votes}; winner Model_Package_{winner} = {summary['winner_persona']}")
    return summary


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    st = load_state()
    lock_ids = prepare_data()
    if "base_lockbox" not in st:
        st["base_lockbox"] = baseline_on_lockbox(lock_ids)
        save(st)
        log(f"reference model lockbox: {st['base_lockbox']}")
    if "plans" not in st:
        phase1(st)
    critique_round(st, 1)
    if "plans_s1" not in st:
        synthesis1(st)
    critique_round(st, 2)
    synthesis2(st)
    implement(st, lock_ids)
    package(st, st["base_lockbox"])
    evaluate(st)
    tally(st, st["base_lockbox"])


if __name__ == "__main__":
    main()
