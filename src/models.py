"""Model definitions and per-fold preprocessing.

Everything that learns from data (feature selection, imputation, log transform, scaling,
encoding) is fitted on the training rows of a fold only, then applied to the validation
rows (AIPI 520 W1 s50-51, W2_H1 s19).

Models
  zero_rule      constant score                          (W2_H2 s5)
  heuristic      rank by sub_n_last5                     (W1 s19, W2_H1 s8)
  logistic       L1 / L2 logistic regression             (W3_H2 s15-17, W4_H2 s19)
  forest         random forest                           (W1 s33)
  mlp            PyTorch multi-layer perceptron           (W8 s20-24, s62, s67-77, s84-87)
"""
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from select_features import select

NEVER = 99
ID_FEATURES = ["id_cohort_Lh", "id_newer_than_Nu"]
INTERACTIONS = ["ix_recent_x_don_count", "ix_recent_x_tix_seats"]
torch.set_num_threads(4)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


# ---------------------------------------------------------------- features shared by all models
def add_interactions(X):
    """Two pre-specified interaction terms (W3_H1 s22-24): last-season subscriber x donations / tickets."""
    X = X.copy()
    X["ix_recent_x_don_count"] = X["sub_last_season"] * X["don_count"]
    X["ix_recent_x_tix_seats"] = X["sub_last_season"] * X["tix_total_seats"]
    return X


class FoldFeatures:
    """Chooses the 25 features using training rows only. `use_id=False` gives the ablation."""

    def __init__(self, k=25, use_id=True, seed=0):
        self.k, self.use_id, self.seed = k, use_id, seed

    def fit(self, X, y):
        X = add_interactions(X)
        if not self.use_id:
            X = X.drop(columns=ID_FEATURES)
        self.columns_ = select(X, y, k=self.k, seed=self.seed, keep=ID_FEATURES if self.use_id else [])
        return self

    def transform(self, X):
        return add_interactions(X)[self.columns_]


# ---------------------------------------------------------------- preprocessing per model family
def _never_to_nan(X):
    """'Seasons since' columns use 99 for never: add an explicit flag and blank the sentinel."""
    X = X.copy()
    for c in [c for c in X.columns if c.endswith("seasons_since_last")]:
        X[c + "_never"] = (X[c] == NEVER).astype(float)
        X[c] = X[c].where(X[c] != NEVER)
    return X


def _skewed(X):
    num = X.select_dtypes("number")
    return [c for c in num.columns if (num[c].dropna() >= 0).all() and num[c].skew() > 2]


def scaled_preprocessor(X):
    """For logistic regression and the MLP: impute, log1p skewed columns, standardise, one-hot."""
    X = _never_to_nan(X)
    cat = list(X.select_dtypes(["category", "object"]).columns)
    skew = _skewed(X)
    other = [c for c in X.columns if c not in cat and c not in skew]
    num_steps = lambda log: Pipeline(
        ([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if log else [])
        + [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    ct = ColumnTransformer([
        ("skewed", num_steps(True), skew),
        ("numeric", num_steps(False), other),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("never_flags", FunctionTransformer(_never_to_nan)), ("columns", ct)])


def tree_preprocessor(X):
    """For the random forest: median impute (+ missing flags), integer codes for categories."""
    cat = list(X.select_dtypes(["category", "object"]).columns)
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), num),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
    ])


# ---------------------------------------------------------------- the models
class ZeroRule:
    def fit(self, X, y): return self
    def predict_proba(self, X): return np.full(len(X), 0.5)


class Heuristic:
    """Rank by number of seasons subscribed in the last five (the user's recency/frequency rule)."""
    def fit(self, X, y): return self
    def predict_proba(self, X): return X["sub_n_last5"].to_numpy(float)


class Logistic:
    def __init__(self, penalty="l2", C=1.0, class_weight=None, interactions=True):
        self.params = dict(penalty=penalty, C=C, class_weight=class_weight, interactions=interactions)

    def fit(self, X, y):
        if not self.params["interactions"]:
            X = X.drop(columns=[c for c in INTERACTIONS if c in X])
        self.cols_ = list(X.columns)
        self.model_ = Pipeline([
            ("prep", scaled_preprocessor(X)),
            # l1_ratio 1 = L1 (LASSO), 0 = L2 (ridge); sklearn >= 1.8 spelling of `penalty`
            ("lr", LogisticRegression(l1_ratio=1.0 if self.params["penalty"] == "l1" else 0.0, C=self.params["C"],
                                      class_weight=self.params["class_weight"], solver="saga",
                                      max_iter=20000, tol=1e-3)),
        ]).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.cols_])[:, 1]


class Forest:
    def __init__(self, n_estimators=500, max_depth=None, min_samples_leaf=5, max_features="sqrt",
                 class_weight=None, seed=0):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                           max_features=max_features, class_weight=class_weight)
        self.seed = seed

    def fit(self, X, y):
        self.model_ = Pipeline([
            ("prep", tree_preprocessor(X)),
            ("rf", RandomForestClassifier(**self.params, n_jobs=-1, random_state=self.seed)),
        ]).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, 1]


class Net(nn.Module):
    """Fully connected layers with ReLU; optional batch norm (before ReLU) or dropout."""

    def __init__(self, n_in, hidden, dropout=0.0, batch_norm=False):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # one logit; sigmoid gives the probability
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLP:
    """One network. Early stopping on AUROC of an inner 15% split of the training rows (W8 s68)."""

    def __init__(self, hidden=(32, 16), dropout=0.1, batch_norm=False, lr=1e-3, weight_decay=1e-3,
                 batch_size=128, max_epochs=300, patience=25, seed=0):
        self.params = dict(hidden=tuple(hidden), dropout=dropout, batch_norm=batch_norm, lr=lr,
                           weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        self.seed = seed

    def fit(self, X, y):
        p, y = self.params, np.asarray(y)
        torch.manual_seed(self.seed)
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        self.prep_ = scaled_preprocessor(Xa).fit(Xa)          # fitted on inner-training rows only
        ta = torch.tensor(self.prep_.transform(Xa), dtype=torch.float32)
        tb = torch.tensor(self.prep_.transform(Xb), dtype=torch.float32)
        ya_t = torch.tensor(ya, dtype=torch.float32)
        self.net_ = Net(ta.shape[1], p["hidden"], p["dropout"], p["batch_norm"])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, wait, self.best_epoch_ = -1, None, 0, 0
        for epoch in range(p["max_epochs"]):
            self.net_.train()
            for idx in torch.randperm(len(ta)).split(p["batch_size"]):
                opt.zero_grad()
                loss_fn(self.net_(ta[idx]), ya_t[idx]).backward()
                opt.step()
            self.net_.eval()                                    # dropout off, batch-norm frozen (W8 s74)
            with torch.no_grad():
                auc = roc_auc_score(yb, self.net_(tb).numpy())
            if auc > best:
                best, wait, self.best_epoch_ = auc, 0, epoch + 1
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= p["patience"]:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net_(torch.tensor(self.prep_.transform(X), dtype=torch.float32))).numpy()


class MLPEnsemble:
    """'Multiple and in layers': average the probabilities of several networks (configs x seeds)."""

    def __init__(self, configs, seeds=(0, 1, 2)):
        self.configs, self.seeds = configs, seeds

    def fit(self, X, y):
        self.members_ = [MLP(**cfg, seed=s).fit(X, y) for cfg in self.configs for s in self.seeds]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)
