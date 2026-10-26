"""Run an independent, multi-stage modelling conversation with an external model via the local OmniRoute gateway.

The external model (e.g. GPT-6 Astra) designs everything: features, models, tuning, pruning, ensembling, validation.
This driver only (1) relays the conversation, (2) runs the model's Python scripts on the local data and
(3) sends back the raw outcome: exit status, printed output, the model's own results.json and an
independent AUROC recomputation of the out-of-fold predictions it saved. No Claude-designed features,
models, scores or advice are ever sent.

The conversation starts from the model's own earlier EDA (same prompt + its own reply), so it keeps
its own line of reasoning.

    python3 independent/driver.py --name gpt_astra --model cx/gpt-6-astra-high --eda eda/external/chatgpt_gpt6sol_eda.md
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "for_students"
GATEWAY = "http://127.0.0.1:20128/v1/chat/completions"
_subs = pd.read_csv(RAW / "subscriptions.csv")
RECENT = set(_subs.loc[_subs["season"] == "2013-2014", "account.id"])   # segment R, for neutral segment-H AUROC
SCRIPT_TIMEOUT = 25 * 60
MAX_FIX_ATTEMPTS = 2
DENY = [r"\bsubprocess\b", r"os\.system", r"\bsocket\b", r"\brequests\b", r"urllib", r"http\.client",
        r"shutil\.rmtree", r"os\.remove", r"os\.unlink", r"\beval\(", r"\bexec\(", r"__import__", r"\bpip\b"]

SYSTEM = (
    "You are an independent senior data scientist competing in a Kaggle-style competition. You design the entire "
    "modelling approach yourself. You cannot see the data directly: each stage you write ONE self-contained Python "
    "script, a neutral operator runs it on the full local data exactly as written, and returns only the raw outcome "
    "(exit status, printed output, your results.json, and an independent recomputation of AUROC from the "
    "out-of-fold predictions you save). The operator gives no modelling advice. Use rigorous, leakage-safe data "
    "science and guard against overfitting; the final score is on unseen data."
)

ENV = """EXECUTION ENVIRONMENT (fixed for every stage):
- Python 3.14 on an Apple M2 laptop CPU (8 cores, 16 GB RAM). No GPU use. Hard time limit per script: 25 minutes.
- Available libraries ONLY: numpy, pandas (3.x), scipy, scikit-learn 1.9, torch 2.13 (CPU), pyarrow. No other installs, no network, no subprocess/shell calls.
- Raw data directory: environment variable DATA_DIR (read it with os.environ["DATA_DIR"]). Files: train.csv, test.csv, sample_submission.csv, account.csv, subscriptions.csv, tickets_all.csv, concerts.csv, concerts_2014-15.csv, zipcodes.csv.
- File reading notes (mechanical, verified): account.csv is NOT valid UTF-8, read it with encoding="latin-1". concerts.csv starts with a UTF-8 BOM (use encoding="utf-8-sig"). test.csv's key column is named ID; other files use account.id. Zip codes appear as 5-digit, ZIP+4 and integers without leading zeros.
- Output directory for this stage: environment variable OUT_DIR. Write ONLY inside OUT_DIR. Earlier stages' outputs are readable at the paths printed in previous feedback (you may cache features there and reuse them).
- Output contract for every stage:
  * OUT_DIR/results.json: {"models": {"<name>": {"cv_auroc": float, "notes": str}}, "best": "<name>", "validation": "<one-line description of your CV scheme>"}
  * OUT_DIR/oof_<name>.csv for each model you report: columns account.id,pred with one out-of-fold prediction for EVERY train.csv account (6,941 rows). The operator recomputes AUROC from these.
  * Final stage only: OUT_DIR/submission.csv in the exact sample_submission format.
  * Print a short human-readable summary at the end.
- Set random seeds for reproducibility.
REPLY FORMAT: first a short plan/rationale (max ~300 words), then exactly ONE ```python code block containing the complete script."""

STAGES = [
    ("1_design_baseline",
     "STAGE 1 of 4 — Design and baseline. Using your EDA above and the task brief, build your full leakage-safe data "
     "pipeline from the raw files (cleaning + your own feature engineering for train and test), choose a validation "
     "scheme suited to this data, and evaluate at least three different candidate model families with sensible default "
     "settings. Report each model's CV AUROC and save its out-of-fold predictions."),
    ("2_tune",
     "STAGE 2 of 4 — Hyperparameter optimisation. Based on the stage 1 outcome, choose the most promising models "
     "(your call how many) and optimise their hyperparameters with a protocol that avoids optimistic bias. Re-test them "
     "and report the tuned CV AUROC alongside the untuned baselines."),
    ("3_prune",
     "STAGE 3 of 4 — Feature pruning. Prune features for your tuned models using a method you judge appropriate "
     "(it may differ per model), keeping the selection inside your validation so it does not leak. Re-test and report "
     "pruned vs unpruned CV AUROC per model."),
    ("5_segment",
     "STAGE 5 (extra, requested by the student) — Segment modelling. The student wants two segments: segment R = accounts "
     "with a subscription in the 2013-14 season, segment H = every other account (the hard group: few positives among many "
     "accounts). Build a separate, strongly regularised model trained on segment H only, keep or build a model for segment R, "
     "and combine the two into one score in a way you validate. Also engineer, from the billing zip: (a) the distance from the "
     "patron's home to the venue they most recently attended, and (b) neighbourhood income (zipcodes.csv TotalWages / "
     "TaxReturnsFiled), and test whether they help in segment H. Keep feature selection and tuning leakage-safe as in your "
     "earlier stages. Report, for your stage-4 model and for the segmented model, overall CV AUROC and AUROC within segment H "
     "(save OOF for both), and write OUT_DIR/submission.csv for the segmented model."),
    ("4_combine_final",
     "STAGE 4 of 4 — Final model. Find the best combination of your (pruned) models to maximise AUROC on unseen data "
     "while avoiding overfitting (your choice of method, e.g. a single model, averaging, stacking). Validate the final "
     "choice, refit as appropriate, and write OUT_DIR/submission.csv. In your reply's rationale, also state: the final "
     "model, the features it uses, the anti-overfitting techniques you used, and your expected AUROC on unseen data."),
]


STAGES.sort(key=lambda st: st[0])
CORE_STAGES = 4


def key():
    return subprocess.run(["security", "find-generic-password", "-s", "omniroute", "-w"],
                          capture_output=True, text=True).stdout.strip()


def chat(model, messages, log):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 20000}).encode()
    for attempt in range(4):
        req = urllib.request.Request(GATEWAY, body, {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.loads(r.read())
            log(f"  reply from {d.get('model')} in {time.time() - t:.0f}s, usage {d.get('usage')}")
            return d["choices"][0]["message"]["content"], d.get("model")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError) as e:
            msg = e.read().decode()[:400] if isinstance(e, urllib.error.HTTPError) else str(e)
            log(f"  gateway error (attempt {attempt + 1}): {msg}")
            m = re.search(r'"reset_seconds":\s*(\d+)', msg)     # honour a provider cooldown (max 40 min)
            time.sleep(min(int(m.group(1)) + 30, 2400) if m else 30 * (attempt + 1))
    raise RuntimeError("gateway failed 4 times")


def extract_code(text):
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def check_code(code):
    return [p for p in DENY if re.search(p, code)]


def run_script(code, run_dir, stage_dir, data_dir):
    stage_dir.mkdir(parents=True, exist_ok=True)
    script = stage_dir / "script.py"
    script.write_text(code)
    env = {"DATA_DIR": str(data_dir), "OUT_DIR": str(stage_dir), "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
           "HOME": str(run_dir), "OMP_NUM_THREADS": "8", "PYTHONHASHSEED": "0"}
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", str(script)], cwd=stage_dir, env=env, capture_output=True,
                           text=True, timeout=SCRIPT_TIMEOUT)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT after 25 min"
    (stage_dir / "stdout.txt").write_text(out or "")
    (stage_dir / "stderr.txt").write_text(err or "")
    return rc, out or "", err or "", time.time() - t


def assess(stage_dir, final):
    """Neutral checks: contract compliance and independent AUROC of saved out-of-fold predictions."""
    y = pd.read_csv(RAW / "train.csv").set_index("account.id")["label"]
    notes, recomputed = [], {}
    res = stage_dir / "results.json"
    results = json.loads(res.read_text()) if res.exists() else None
    if results is None:
        notes.append("results.json missing")
    for f in sorted(stage_dir.glob("oof_*.csv")):
        try:
            o = pd.read_csv(f).set_index("account.id")["pred"]
            j = y.to_frame().join(o, how="left")
            miss = int(j["pred"].isna().sum())
            if miss:
                notes.append(f"{f.name}: {miss} train accounts missing a prediction")
            recomputed[f.stem[4:]] = round(float(roc_auc_score(j["label"][j["pred"].notna()], j["pred"].dropna())), 5)
            if stage_dir.name.startswith("5"):
                hj = j[~j.index.isin(RECENT) & j["pred"].notna()]
                recomputed[f.stem[4:] + " [segment H]"] = round(float(roc_auc_score(hj["label"], hj["pred"])), 5)
        except Exception as e:  # report, never fix
            notes.append(f"{f.name}: could not be read ({e})")
    if final:
        s = stage_dir / "submission.csv"
        if not s.exists():
            notes.append("submission.csv missing")
        else:
            sub, ss = pd.read_csv(s), pd.read_csv(RAW / "sample_submission.csv")
            if list(sub.columns) != list(ss.columns): notes.append(f"submission header {list(sub.columns)} != {list(ss.columns)}")
            if len(sub) != len(ss) or not sub["ID"].isin(ss["ID"]).all(): notes.append("submission IDs/rows do not match sample_submission")
            if sub["Predicted"].isna().any() or not np.isfinite(sub["Predicted"]).all(): notes.append("submission has NaN/inf")
    return results, recomputed, notes


def feedback(stage, rc, out, err, secs, results, recomputed, notes, stage_dir):
    tail = lambda s, n: s if len(s) <= n else "...[truncated]...\n" + s[-n:]
    parts = [f"OPERATOR REPORT for {stage} (raw outcome, no advice)",
             f"- exit status: {rc}; runtime {secs / 60:.1f} min; outputs in {stage_dir}",
             f"- printed output (tail):\n```\n{tail(out, 6000)}\n```"]
    if rc != 0:
        parts.append(f"- error output (tail):\n```\n{tail(err, 4000)}\n```")
    parts.append(f"- your results.json: {json.dumps(results) if results else 'not found'}")
    parts.append(f"- independent AUROC recomputed from your oof_*.csv files: {json.dumps(recomputed) if recomputed else 'none found'}")
    if notes:
        parts.append("- contract issues: " + "; ".join(notes))
    return "\n".join(parts)


def main(name, model, eda_path, stages, resume=False):
    run_dir = ROOT / "independent" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    if not data_dir.exists():
        shutil.copytree(RAW, data_dir)            # private copy so scripts can't touch the originals
    logf = run_dir / "log.txt"

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")

    task = (ROOT / "independent/context/kaggle_task.md").read_text()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": (ROOT / "eda/external/prompt_and_dossier.md").read_text()},
                {"role": "assistant", "content": Path(eda_path).read_text()}]
    pending = f"{task}\n\n{ENV}\n\n"
    served, start, resume_prompt = set(), 0, None
    tp = run_dir / "transcript.json"
    if resume and tp.exists():
        # Continue the same conversation: completed stages are those with results.json; the unanswered
        # last user message (if any) is re-sent as-is.
        messages = json.loads(tp.read_text())
        if messages[-1]["role"] == "user":
            resume_prompt = messages.pop()["content"]
        start = sum(1 for st, _ in STAGES[:stages] if (run_dir / st / "results.json").exists()
                    and (st[0] not in "45" or (run_dir / st / "submission.csv").exists()))
        if resume_prompt is None and start > 0 and (run_dir / "final_summary.md").exists() and start >= CORE_STAGES:
            pending = ""          # the conversation already ended with a final summary; just add the new stage
        elif resume_prompt is None and start > 0:
            # The last stage ran but its report was never delivered: rebuild it from the saved outputs.
            prev = STAGES[start - 1][0]
            sd = run_dir / prev
            results, recomputed, notes = assess(sd, prev[0] in "45")
            pending = feedback(prev, 0, (sd / "stdout.txt").read_text(), (sd / "stderr.txt").read_text(), 0,
                               results, recomputed, notes, sd) + "\n\n"
        log(f"resuming at stage {start + 1} from saved transcript ({len(messages)} messages)")
    for i, (stage, instr) in enumerate(STAGES[:stages]):
        if i < start:
            continue
        final = stage[0] in "45"
        stage_dir = run_dir / stage
        prompt = pending + instr
        if resume_prompt is not None:
            prompt, resume_prompt = resume_prompt, None
        for attempt in range(MAX_FIX_ATTEMPTS + 1):
            log(f"{stage}: asking {model} (attempt {attempt + 1})")
            messages.append({"role": "user", "content": prompt})
            reply, m = chat(model, messages, log)
            served.add(m)
            messages.append({"role": "assistant", "content": reply})
            (run_dir / "transcript.json").write_text(json.dumps(messages, indent=1))
            code = extract_code(reply)
            if code is None:
                prompt = "OPERATOR REPORT: no ```python block found in your reply. Reply again with the plan and exactly one complete python block."
                continue
            bad = check_code(code)
            if bad:
                prompt = f"OPERATOR REPORT: the script was not run because it uses disallowed operations ({bad}). Remove them and resend the complete script."
                continue
            rc, out, err, secs = run_script(code, run_dir, stage_dir, data_dir)
            results, recomputed, notes = assess(stage_dir, final)
            log(f"{stage}: exit {rc}, {secs / 60:.1f} min, recomputed AUROC {recomputed}, issues {notes}")
            prompt = feedback(stage, rc, out, err, secs, results, recomputed, notes, stage_dir)
            if rc == 0 and results is not None and (not final or (stage_dir / "submission.csv").exists()):
                break
            prompt += "\n\nThe stage did not complete. Fix the script and resend the complete script for this same stage."
        pending = prompt + "\n\n"
    messages.append({"role": "user", "content": pending + "All stages are done. In under 400 words, summarise your final approach, what you learned from each stage's results, and the main risks for unseen data. No code."})
    summary, m = chat(model, messages, log)
    served.add(m)
    messages.append({"role": "assistant", "content": summary})
    (run_dir / "transcript.json").write_text(json.dumps(messages, indent=1))
    (run_dir / ("final_summary.md" if stages <= CORE_STAGES else f"final_summary_stage{stages}.md")).write_text(summary)
    (run_dir / "served_models.json").write_text(json.dumps(sorted(x for x in served if x)))
    log(f"done; requested {model}, gateway-reported serving models {sorted(x for x in served if x)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eda", required=True, help="this model's own earlier EDA reply")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="continue a run from its saved transcript")
    a = ap.parse_args()
    main(a.name, a.model, a.eda, a.stages, a.resume)
