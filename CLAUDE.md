# Project context: orchestra subscription Kaggle (520 course, Fall 2026)

Predict P(subscribe to 2014-15 season) for each account in `for_students/test.csv`. Binary classification, scored by AUROC (public + private leaderboard, averaged for the grade). See DATA_GUIDE.md for the data dictionary and EDA findings; `eda/data_summary.html` holds the charts.

## Submission
- Deliverable is a **GitHub repo**. Its **README.md is the graded 2–4 page write-up** (data pipeline, modelling, validation approach, and why).
- Predictions: probabilities in [0, 1], header `ID,Predicted`, 2,975 rows.
- Up to **10 Kaggle submissions per day**. The private test set differs from the public one, so prefer CV over leaderboard probing.

## Course rules (from Kaggle.pptx), which constrain how to help
- AI may write **code**, but the **logic and decision process must be the student's**. Present options, evidence and trade-offs, and let the student choose features, algorithms and validation strategy. Do not silently make modelling decisions; ask or offer alternatives.
- **Never write, edit or polish README.md** (the graded write-up); AI use there scores 0. Keep AI-generated notes in DATA_GUIDE.md, which must not be copied into README.md. Factual answers about what the code does are fine.
- No collaboration with other students.

## Grading
70 pts from AUROC: >0.53→40, >0.8→50, >0.9→55, >0.94→60, >0.96→65, >0.97→70. **Target: >0.97.** 30 pts write-up. +5 for top 3 on private.

## Student's plan (from the Word doc notes)
1. Choose features → build one pipeline that creates them → apply it to train and test accounts.
2. Break down ambiguous or unclear columns, especially ones whose ordering or meaning is not obvious (e.g. which direction `price.level` and `subscription_tier` run, what high/low means).
3. Feature engineering → choose algorithm(s) → likely an **ensemble** of several models.
4. Design a sound validation approach (cross-validation etc.).

## Data gotchas
- `account.csv` is Latin-1; `concerts.csv` has a BOM; `concerts_2014-15.csv` has mojibake.
- Join key `account.id` (test uses `ID`).
- Billing zips: take first 5 chars, zero-pad; `zipcodes.Zipcode` is an int.
- `tickets_all.price.level` is mixed numeric/text.
- `first.donated` runs to 2014-09, so watch for timing leakage around donations.
- Account ID prefix is a cohort artifact: `001i000000Lh` = 82% positive (mostly 2013-14 subscribers); IDs newer than `Nu` with no history ≈ 7.6%. Using it is the student's decision.
- `subscriptions.section` is 100% null in 2013-14 only; `subscription_tier` = seats × (1 Full, 0.5 Trio/Quartet) for 99.7% of rows.
- ChatGPT/Gemini EDA cross-check and verified findings: `eda/external/REVIEW.md`.
- Raw data in `for_students/` is read-only; write derived data elsewhere.

## Pipeline
- `python3 src/clean.py` → cleaned parquet in `data/processed/`; `python3 src/features.py` → `features_{train,test}.parquet` + `feature_groups.json` (groups: sub_, tix_, any_, don_, pre14_, has_, geo_, id_).
- `build(d, ids, asof)` supports time-shifted builds; it drops donation/ID groups when `asof < 2013`.
- `src/select_features.py`: `select(X, y)` → 25 features (drop >50% missing → mutual-info filter + |Spearman|≥0.95 dedup → RF importance), ID-cohort features always kept; call inside each CV training fold.
- Student decisions (2026-10-25): include ID-cohort features; cap at 25 features; add a PyTorch MLP ensemble (CPU is ~3× faster than MPS here). Ground modelling choices in AIPI 520 lectures (OneDrive `AIPI 520/Actual Lectures/*.pptx`; W1 feature selection, W2 CV/leakage, W3 regularisation, W8 NNs).
- Review deck: `python3 presentation/build.py` → `presentation/data_review.html`. Models: `python3 src/train.py` (tune-once, 5×5 CV, ~22 min) → `results/` + `submissions/{logistic,forest,mlp_ensemble,average_A_B_C}.csv`. Run of 2026-10-26: pooled OOF AUROC logistic 0.9821, forest 0.9801, MLP ensemble 0.9797, A+B+C average 0.9825; without ID features −0.006 to −0.011. Which file to submit is the student's call. Public LB (2026-10-26): forest 0.97679, average 0.97338, mlp_ensemble 0.97153, logistic 0.97107. Final selection (up to 2) is the student's call.

## Deadline
Due ~2026-10-28 (10 days from 2026-10-18).
