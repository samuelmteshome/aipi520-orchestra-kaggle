# Kaggle task brief (from the course's competition page and lecture slides)

- Competition: "Classical Music Meets Classical ML" (course in-class Kaggle).
- Objective: predict which previous patrons of an orchestra will purchase a season subscription (any level) for the upcoming 2014-15 season. Binary classification.
- Submit "soft predictions": the probability that each account in test.csv purchases a subscription. File format must match sample_submission.csv exactly: header `ID,Predicted`, 2,975 rows.
- Metric: Area Under the ROC curve (AUROC).
- The test set is split into a public part (score visible after each submission) and a private part (revealed after the competition ends). The final standing uses the private part; the grade averages public and private performance.
- Up to 10 submissions per day.
- Grade thresholds (AUROC → points out of 70): > 0.53 → 40, > 0.8 → 50, > 0.9 → 55, > 0.94 → 60, > 0.96 → 65, > 0.97 → 70. +5 bonus for top 3 on the private set.
- Files: train.csv (account.id, label), test.csv (ID), sample_submission.csv, account.csv (location info and donation history per patron), subscriptions.csv (previous subscriptions by account: season, location, price level, seats), tickets_all.csv (previously purchased single tickets: season, location, set, price level, seats), concerts.csv (previous concerts by season: title, composer, location), concerts_2014-15.csv (planned 2014-15 sets), zipcodes.csv (zip-code location and demographics).
- Suggested approach from the course: determine possible features; build a data pipeline that creates features for both train and test; clean data; generate features; prepare data; model (feature engineering and selection, evaluate algorithms, tune hyperparameters, possibly ensembles); choose a validation approach.
