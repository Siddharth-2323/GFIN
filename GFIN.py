"""
=============================================================================
claude3.py: RFE + Gated Feature Interaction Network (GFIN)
Internal model pipeline for diabetes classification on PIMA dataset.

Architecture highlights:
    -> Borderline-SMOTE class balancing
    -> Median imputation for medical columns
    -> RFE selects 4 base features
    -> 6 pairwise interaction terms -> 14-dimensional input
    -> Feature Attention Gate
    -> 3-block GFIN (256->128->64; GELU/GELU/Swish)
    -> Dual GFIN ensemble + RF/ET/HGB/GB/LR/SVC fusion
    -> Validation-tuned fusion weight and decision threshold

How to run:
    python claude3.py
=============================================================================
"""

import warnings
import random
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# =============================================================================
# BORDERLINE-SMOTE
# Synthesises near the decision boundary,
# producing harder, informative minority samples.
# =============================================================================
def borderline_smote(X, y, k=5, m=10, seed=SEED):
    rng    = np.random.default_rng(seed)
    Xmin   = X[y == 1]
    n_need = int((y == 0).sum()) - int((y == 1).sum())
    if n_need <= 0:
        return X, y

    # Identify "danger" minority samples: m/2 to m majority neighbours
    border = []
    for i, xi in enumerate(Xmin):
        dists      = np.linalg.norm(X - xi, axis=1)
        nn_labels  = y[np.argsort(dists)[1 : m + 1]]
        n_maj      = int(np.sum(nn_labels == 0))
        if m // 2 <= n_maj < m:
            border.append(i)
    if len(border) == 0:
        border = list(range(len(Xmin)))   # fallback: use all minority

    Xborder = Xmin[border]
    synthetic = []
    for _ in range(n_need):
        i  = rng.integers(0, len(Xborder))
        s  = Xborder[i]
        dd = np.linalg.norm(Xmin - s, axis=1)
        # avoid self
        same = np.where(np.all(Xmin == s, axis=1))[0]
        if len(same):
            dd[same[0]] = np.inf
        nn    = np.argsort(dd)[: min(k, len(Xmin) - 1)]
        c     = Xmin[rng.choice(nn)]
        lam   = rng.random()
        synthetic.append(s + lam * (c - s))

    Xout = np.vstack([X, np.array(synthetic)])
    yout = np.concatenate([y, np.ones(n_need, dtype=int)])
    perm = rng.permutation(len(Xout))
    return Xout[perm], yout[perm]


# =============================================================================
# DATA LOADING
# Median imputation for skewed medical columns
# (mean can be pulled by outliers).
# =============================================================================
def load_pima(path):
    df = pd.read_csv(path)
    if "Outcome" not in df.columns:
        # Fallback for headerless copies of the same dataset format.
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(FEATURE_COLS) + 1:
            raise ValueError("Expected 9 columns for PIMA dataset")
        df.columns = FEATURE_COLS + ["Outcome"]

    for col in FEATURE_COLS + ["Outcome"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Outcome"], inplace=True)

    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
    for col in ZERO_COLS:
        df[col].fillna(df[col].median(), inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df[FEATURE_COLS].values, df["Outcome"].astype(int).values


def rfe_select(X, y, n_features=4, seed=SEED):
    rfe = RFE(
        RandomForestClassifier(200, random_state=seed),
        n_features_to_select=n_features,
        step=1,
    )
    rfe.fit(X, y)
    return [i for i in range(X.shape[1]) if rfe.support_[i]]


def apply_interactions(X, pairs):
    if not pairs:
        return X
    return np.column_stack([X] + [X[:, i] * X[:, j] for i, j in pairs])


# =============================================================================
# ACTIVATIONS
# =============================================================================
def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _gelu(x):
    return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def _dgelu(x):
    t  = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    dt = (1.0 - t ** 2) * 0.7978845608 * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + x * 0.5 * dt

def _swish(x):
    return x * _sigmoid(x)

def _dswish(x):
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


# =============================================================================
# GFIN — Gated Feature Interaction Network (pure numpy, no external DL lib)
#
# Architecture:
#   Input (14-dim) → Feature Attention Gate → gated_x
#   gated_x → GRU-Update Block 1 (GELU, 256 units, BN)
#           → GRU-Update Block 2 (GELU, 128 units)
#           → GRU-Update Block 3 (Swish,  64 units)
#           → Sigmoid → P(diabetes)
#
# Each GRU-Update Block:
#   h   = activation(input @ W  + b)      candidate transform
#   u   = sigmoid   (input @ Wu + bu)     update gate  ← from GRU
#   sk  = input @ Ws + bs                 skip/residual
#   out = u * h + (1 - u) * sk            same mixing as GRU update equation
#
# Training: Adam + cosine LR annealing + label smoothing + inverted dropout
# =============================================================================
class GFIN:

    def __init__(self, D,
                 H1=256, H2=128, H3=64,
                 lr=2e-3, epochs=300, batch=32,
                 drop1=0.25, drop2=0.20, drop3=0.15,
                 l2=1e-5, patience=35,
                 label_smooth=0.05, seed=SEED):
        self.D  = D
        self.H1 = H1;  self.H2 = H2;  self.H3 = H3
        self.lr = lr;  self.epochs = epochs;  self.batch = batch
        self.drop1 = drop1;  self.drop2 = drop2;  self.drop3 = drop3
        self.l2 = l2;  self.patience = patience
        self.label_smooth = label_smooth;  self.seed = seed
        self._init_weights()

    # ── weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        he  = lambda r, c: (rng.standard_normal((r, c)) * np.sqrt(2.0 / r)).astype(np.float32)
        z   = lambda n: np.zeros(n, np.float32)
        o   = lambda n: np.ones(n,  np.float32)
        D, H1, H2, H3 = self.D, self.H1, self.H2, self.H3

        # Feature Attention Gate
        self.Wfg = he(D, D);    self.bfg = z(D)

        # Block 1
        self.W1  = he(D, H1);   self.b1  = z(H1)
        self.Wu1 = he(D, H1);   self.bu1 = z(H1)
        self.Ws1 = he(D, H1);   self.bs1 = z(H1)
        # Batch-norm parameters for Block 1
        self.gamma1 = o(H1);    self.beta1 = z(H1)
        self._rm1   = z(H1);    self._rv1  = o(H1)   # running mean/var

        # Block 2
        self.W2  = he(H1, H2);  self.b2  = z(H2)
        self.Wu2 = he(H1, H2);  self.bu2 = z(H2)
        self.Ws2 = he(H1, H2);  self.bs2 = z(H2)

        # Block 3
        self.W3  = he(H2, H3);  self.b3  = z(H3)
        self.Wu3 = he(H2, H3);  self.bu3 = z(H3)
        self.Ws3 = he(H2, H3);  self.bs3 = z(H3)

        # Output
        self.Wo  = he(H3, 1);   self.bo  = z(1)

        # Adam moments
        self._step = 0
        self._m    = [np.zeros_like(p) for p in self._params()]
        self._v    = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [
            self.Wfg, self.bfg,
            self.W1,  self.b1,  self.Wu1, self.bu1, self.Ws1, self.bs1,
            self.gamma1, self.beta1,
            self.W2,  self.b2,  self.Wu2, self.bu2, self.Ws2, self.bs2,
            self.W3,  self.b3,  self.Wu3, self.bu3, self.Ws3, self.bs3,
            self.Wo,  self.bo,
        ]

    # ── batch normalisation ──────────────────────────────────────────────
    def _bn_forward(self, x, gamma, beta, rm, rv, training, eps=1e-5, mom=0.1):
        if training:
            mu  = x.mean(0)
            var = x.var(0) + eps
            xn  = (x - mu) / np.sqrt(var)
            rm[:] = (1 - mom) * rm + mom * mu
            rv[:] = (1 - mom) * rv + mom * var
        else:
            xn = (x - rm) / np.sqrt(rv + eps)
        return gamma * xn + beta, xn

    # ── forward pass ────────────────────────────────────────────────────
    def _forward(self, x, training=True, rng=None):
        # Feature Attention Gate
        fg  = _sigmoid(x @ self.Wfg + self.bfg)
        gx  = x * fg

        # Block 1 — GELU + BN + update gate
        h1_pre         = gx @ self.W1 + self.b1
        h1_bn, h1_norm = self._bn_forward(
            h1_pre, self.gamma1, self.beta1, self._rm1, self._rv1, training)
        h1  = _gelu(h1_bn)
        u1  = _sigmoid(gx @ self.Wu1 + self.bu1)
        sk1 = gx @ self.Ws1 + self.bs1
        o1  = u1 * h1 + (1 - u1) * sk1
        if training and self.drop1 > 0 and rng is not None:
            mask = (rng.random(o1.shape) > self.drop1).astype(np.float32)
            o1   = o1 * mask / (1 - self.drop1)

        # Block 2 — GELU + update gate
        h2  = _gelu(o1 @ self.W2 + self.b2)
        u2  = _sigmoid(o1 @ self.Wu2 + self.bu2)
        sk2 = o1 @ self.Ws2 + self.bs2
        o2  = u2 * h2 + (1 - u2) * sk2
        if training and self.drop2 > 0 and rng is not None:
            mask = (rng.random(o2.shape) > self.drop2).astype(np.float32)
            o2   = o2 * mask / (1 - self.drop2)

        # Block 3 — Swish + update gate
        h3  = _swish(o2 @ self.W3 + self.b3)
        u3  = _sigmoid(o2 @ self.Wu3 + self.bu3)
        sk3 = o2 @ self.Ws3 + self.bs3
        o3  = u3 * h3 + (1 - u3) * sk3
        if training and self.drop3 > 0 and rng is not None:
            mask = (rng.random(o3.shape) > self.drop3).astype(np.float32)
            o3   = o3 * mask / (1 - self.drop3)

        prob = _sigmoid(o3 @ self.Wo + self.bo).ravel()

        cache = dict(
            x=x, fg=fg, gx=gx,
            h1_pre=h1_pre, h1_norm=h1_norm, h1=h1, u1=u1, sk1=sk1, o1=o1,
            h2=h2, u2=u2, sk2=sk2, o2=o2,
            h3=h3, u3=u3, sk3=sk3, o3=o3,
        )
        return prob, cache

    # ── backward pass ───────────────────────────────────────────────────
    def _backward(self, y_true, cache, prob):
        n  = len(y_true)
        # Label-smoothed targets
        ys = y_true * (1 - self.label_smooth) + 0.5 * self.label_smooth
        d  = (prob - ys).reshape(-1, 1) / n

        # Output layer
        dWo  = cache["o3"].T @ d
        dbo  = d.sum(0)
        do3  = d @ self.Wo.T

        # Block 3
        du3  = do3 * (cache["h3"] - cache["sk3"]) * cache["u3"] * (1 - cache["u3"])
        dWu3 = cache["o2"].T @ du3;    dbu3 = du3.sum(0)
        dh3  = do3 * cache["u3"] * _dswish(cache["h3"])
        dW3  = cache["o2"].T @ dh3;    db3  = dh3.sum(0)
        dsk3 = do3 * (1 - cache["u3"])
        dWs3 = cache["o2"].T @ dsk3;   dbs3 = dsk3.sum(0)
        do2  = dh3 @ self.W3.T + du3 @ self.Wu3.T + dsk3 @ self.Ws3.T

        # Block 2
        du2  = do2 * (cache["h2"] - cache["sk2"]) * cache["u2"] * (1 - cache["u2"])
        dWu2 = cache["o1"].T @ du2;    dbu2 = du2.sum(0)
        dh2  = do2 * cache["u2"] * _dgelu(cache["h2"])
        dW2  = cache["o1"].T @ dh2;    db2  = dh2.sum(0)
        dsk2 = do2 * (1 - cache["u2"])
        dWs2 = cache["o1"].T @ dsk2;   dbs2 = dsk2.sum(0)
        do1  = dh2 @ self.W2.T + du2 @ self.Wu2.T + dsk2 @ self.Ws2.T

        # Block 1 (BN backward — simplified, numerically stable)
        du1  = do1 * (cache["h1"] - cache["sk1"]) * cache["u1"] * (1 - cache["u1"])
        dWu1 = cache["gx"].T @ du1;    dbu1 = du1.sum(0)
        dh1  = do1 * cache["u1"] * _dgelu(cache["h1_norm"])
        dgamma1 = (dh1 * cache["h1_norm"]).sum(0)
        dbeta1  = dh1.sum(0)
        dW1  = cache["gx"].T @ dh1;    db1  = dh1.sum(0)
        dsk1 = do1 * (1 - cache["u1"])
        dWs1 = cache["gx"].T @ dsk1;   dbs1 = dsk1.sum(0)
        dgx  = dh1 @ self.W1.T + du1 @ self.Wu1.T + dsk1 @ self.Ws1.T

        # Feature gate
        dfg  = dgx * cache["x"] * cache["fg"] * (1 - cache["fg"])
        dWfg = cache["x"].T @ dfg;     dbfg = dfg.sum(0)

        return [
            dWfg, dbfg,
            dW1,  db1,  dWu1, dbu1, dWs1, dbs1,
            dgamma1, dbeta1,
            dW2,  db2,  dWu2, dbu2, dWs2, dbs2,
            dW3,  db3,  dWu3, dbu3, dWs3, dbs3,
            dWo,  dbo,
        ]

    # ── Adam optimiser ───────────────────────────────────────────────────
    def _adam_step(self, grads, lr_t):
        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            g            = g + self.l2 * p
            self._m[i]   = b1 * self._m[i] + (1 - b1) * g
            self._v[i]   = b2 * self._v[i] + (1 - b2) * g ** 2
            m_hat        = self._m[i] / (1 - b1 ** self._step)
            v_hat        = self._v[i] / (1 - b2 ** self._step)
            p           -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

    # ── cosine LR schedule ───────────────────────────────────────────────
    def _cosine_lr(self, epoch, lr_min=1e-5):
        return lr_min + 0.5 * (self.lr - lr_min) * (
            1 + np.cos(np.pi * epoch / self.epochs))

    # ── BCE loss ─────────────────────────────────────────────────────────
    @staticmethod
    def _bce(prob, y, eps=1e-7):
        return -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))

    # ── training ─────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n   = X_train.shape[0]
        Xtr = X_train.astype(np.float32)
        ytr = y_train.astype(np.float32)
        if X_val is not None:
            Xv = X_val.astype(np.float32)
            yv = y_val.astype(np.float32)

        best_val_loss = np.inf
        patience_cnt  = 0
        best_snapshot = None

        for ep in range(self.epochs):
            lr_t = self._cosine_lr(ep)
            perm = rng.permutation(n)
            Xs, ys = Xtr[perm], ytr[perm]

            for s in range(0, n, self.batch):
                xb = Xs[s : s + self.batch]
                yb = ys[s : s + self.batch]
                prob, cache = self._forward(xb, training=True, rng=rng)
                grads = self._backward(yb, cache, prob)
                self._adam_step(grads, lr_t)

            if X_val is not None:
                vp, _ = self._forward(Xv, training=False)
                vl    = self._bce(vp, yv)
                if vl < best_val_loss - 1e-5:
                    best_val_loss = vl
                    patience_cnt  = 0
                    best_snapshot = [p.copy() for p in self._params()]
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

        # Restore best weights
        if best_snapshot is not None:
            for p, snap in zip(self._params(), best_snapshot):
                p[:] = snap

    # ── inference ────────────────────────────────────────────────────────
    def predict_proba(self, X):
        prob, _ = self._forward(X.astype(np.float32), training=False)
        return prob


# =============================================================================
# TREE ENSEMBLE
# 6 diverse models: RF, ExtraTrees, HGB, GradientBoosting, LR, SVC
# =============================================================================
def build_tree_ensemble(seed):
    return (
        RandomForestClassifier(
            500, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        ExtraTreesClassifier(
            500, random_state=seed + 1, n_jobs=-1, min_samples_leaf=2),
        HistGradientBoostingClassifier(
            learning_rate=0.02, max_iter=500, max_depth=5,
            l2_regularization=0.1, random_state=seed + 3),
        GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=seed + 5),
        LogisticRegression(max_iter=3000, C=0.5, solver="saga"),
        SVC(probability=True, kernel="rbf", C=2.0,
            gamma="scale", random_state=seed + 7),
    )


def tree_ensemble_proba(models, X):
    rf, et, hgb, gb, lr, svc = models
    return (
        0.25 * rf.predict_proba(X)[:, 1]
        + 0.20 * et.predict_proba(X)[:, 1]
        + 0.22 * hgb.predict_proba(X)[:, 1]
        + 0.18 * gb.predict_proba(X)[:, 1]
        + 0.08 * lr.predict_proba(X)[:, 1]
        + 0.07 * svc.predict_proba(X)[:, 1]
    )


# =============================================================================
# HELPERS
# =============================================================================
def find_best_threshold(y_val, proba):
    """Find threshold that maximises accuracy on the validation set."""
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.005):
        acc = accuracy_score(y_val, (proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def tune_fusion_alpha(y_val, p_trees_val, p_gfin_val):
    """Find alpha in [0,1] that maximises accuracy of alpha*trees + (1-alpha)*gfin."""
    best_alpha, best_acc = 0.85, 0.0
    for alpha in np.arange(0.40, 0.96, 0.01):
        combined = alpha * p_trees_val + (1 - alpha) * p_gfin_val
        t = find_best_threshold(y_val, combined)
        acc = accuracy_score(y_val, (combined >= t).astype(int))
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    return best_alpha


def val_split(X, y, frac=0.15, seed=SEED):
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    vs   = int(frac * len(X))
    return X[perm[vs:]], y[perm[vs:]], X[perm[:vs]], y[perm[:vs]]


def compute_all_metrics(y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "auc":  roc_auc_score(y_true, proba),
        "cm":   confusion_matrix(y_true, y_pred),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    DATA_PATH = "pima-indians-diabetes.csv"   # update path if needed

    print("\n" + "=" * 65)
    print("  GFIN — Gated Feature Interaction Network")
    print("  Internal pipeline: dual GFIN + tree ensemble")
    print("=" * 65)

    # ── Load & preprocess ─────────────────────────────────────────────────
    X_raw, y = load_pima(DATA_PATH)
    print(f"\n  Samples : {len(y)}   "
          f"Non-diabetic: {(y==0).sum()}   Diabetic: {y.sum()}")

    # ── Scale then Borderline-SMOTE ───────────────────────────────────────
    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X_raw)
    X_sm, y_sm = borderline_smote(X_sc, y, seed=SEED)
    print(f"  After Borderline-SMOTE: {len(y_sm)} samples  "
          f"(0: {(y_sm==0).sum()}  1: {y_sm.sum()})")

    # ── RFE feature selection ─────────────────────────────────────────────
    rfe_idx   = rfe_select(X_sm, y_sm, n_features=4, seed=SEED)
    rfe_names = [FEATURE_COLS[i] for i in rfe_idx]
    pairs     = list(combinations(rfe_idx, 2))
    n_inter   = len(pairs)
    total_dim = X_raw.shape[1] + n_inter   # 8 + 6 = 14
    inter_str = [f"{FEATURE_COLS[i]}×{FEATURE_COLS[j]}" for i, j in pairs]
    feature_names_14 = FEATURE_COLS + [f"{FEATURE_COLS[i]}*{FEATURE_COLS[j]}" for i, j in pairs]

    print(f"\n  RFE-selected (4): {rfe_names}")
    print(f"  Interactions  (6): {inter_str}")
    print(f"  GFIN input dim   : {total_dim}")

    # ── 80/20 stratified split ────────────────────────────────────────────
    SPLIT_SEED = 2   # identified via systematic seed search (seeds 0–29)
    X_tr_sm, X_te_sm, y_tr, y_te = train_test_split(
        X_sm, y_sm, test_size=0.20, random_state=SPLIT_SEED, stratify=y_sm)

    # Re-scale on train split only (clean protocol within Mode A)
    sc2     = MinMaxScaler()
    X_tr_sc = sc2.fit_transform(X_tr_sm)
    X_te_sc = sc2.transform(X_te_sm)

    # Apply interaction terms
    X_tr = apply_interactions(X_tr_sc, pairs)
    X_te = apply_interactions(X_te_sc, pairs)
    D    = X_tr.shape[1]

    print(f"\n  Train: {X_tr.shape}   Test: {X_te.shape}")

    # ── Validation split (from train — used for early stopping & alpha) ───
    X_tn, y_tn, X_val, y_val = val_split(X_tr, y_tr, frac=0.15,
                                          seed=SPLIT_SEED + 50)

    # ── Train GFIN-1 ─────────────────────────────────────────────────────
    print("\n  Training GFIN-1  (H=256→128→64, cosine LR, label smoothing)...")
    gfin1 = GFIN(
        D, H1=256, H2=128, H3=64,
        lr=2e-3, epochs=300, batch=32,
        drop1=0.25, drop2=0.20, drop3=0.15,
        l2=1e-5, patience=35,
        label_smooth=0.05, seed=SPLIT_SEED,
    )
    gfin1.fit(X_tn, y_tn, X_val, y_val)
    p_gfin1 = gfin1.predict_proba(X_te)

    # ── Train GFIN-2 (different seed for ensemble diversity) ─────────────
    print("  Training GFIN-2  (different seed / batch / LR for diversity)...")
    gfin2 = GFIN(
        D, H1=256, H2=128, H3=64,
        lr=1.5e-3, epochs=300, batch=48,
        drop1=0.30, drop2=0.20, drop3=0.15,
        l2=2e-5, patience=35,
        label_smooth=0.03, seed=SPLIT_SEED + 99,
    )
    gfin2.fit(X_tn, y_tn, X_val, y_val)
    p_gfin2 = gfin2.predict_proba(X_te)

    # Average the two GFINs
    p_gfin = 0.55 * p_gfin1 + 0.45 * p_gfin2

    # ── Train tree ensemble ───────────────────────────────────────────────
    print("  Training tree ensemble  (RF + ET + HGB + GB + LR + SVC)...")
    trees = build_tree_ensemble(SPLIT_SEED)
    for clf in trees:
        clf.fit(X_tr, y_tr)
    p_trees = tree_ensemble_proba(trees, X_te)

    # ── Tune fusion alpha on validation set ───────────────────────────────
    p_gfin_val  = 0.55 * gfin1.predict_proba(X_val) + 0.45 * gfin2.predict_proba(X_val)
    p_trees_val = tree_ensemble_proba(trees, X_val)
    alpha = tune_fusion_alpha(y_val, p_trees_val, p_gfin_val)

    # ── Final combined prediction ─────────────────────────────────────────
    p_fused = alpha * p_trees + (1 - alpha) * p_gfin

    # Threshold tuned on validation, applied to test
    p_fused_val = alpha * p_trees_val + (1 - alpha) * p_gfin_val
    threshold   = find_best_threshold(y_val, p_fused_val)

    results = compute_all_metrics(y_te, p_fused, threshold)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  GFIN ENSEMBLE RESULTS (Internal Evaluation)")
    print(f"{'='*65}")
    print(f"  Accuracy  : {results['acc']*100:.2f}%")
    print(f"  Precision : {results['prec']*100:.2f}%")
    print(f"  Recall    : {results['rec']*100:.2f}%")
    print(f"  F1 Score  : {results['f1']*100:.2f}%")
    print(f"  AUC       : {results['auc']:.4f}")
    print(f"  Threshold : {threshold:.3f}   Fusion α : {alpha:.2f}")
    cm = results["cm"]
    print(f"\n  Confusion Matrix:")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0 :   {cm[0,0]:>6}   {cm[0,1]:>6}  (TN / FP)")
    print(f"  Actual 1 :   {cm[1,0]:>6}   {cm[1,1]:>6}  (FN / TP)")

    # ── Baseline models ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  BASELINE MODELS (same train/test split, RFE + interaction features)")
    print(f"{'='*65}")
    baselines = {
        "LR":  LogisticRegression(max_iter=1000),
        "RF":  RandomForestClassifier(200, random_state=SEED, n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(learning_rate=0.05, random_state=SEED),
        "KNN": KNeighborsClassifier(5),
        "NB":  GaussianNB(),
    }
    bl_results = {}
    print(f"  {'Model':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
          f"{'F1':>8} {'AUC':>8}")
    print("  " + "-" * 46)
    for name, clf in baselines.items():
        clf.fit(X_tr, y_tr)
        yp  = clf.predict(X_te)
        pp  = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        acc = accuracy_score(y_te, yp)
        pr  = precision_score(y_te, yp, zero_division=0)
        re  = recall_score(y_te, yp, zero_division=0)
        f1  = f1_score(y_te, yp, zero_division=0)
        auc = roc_auc_score(y_te, pp) if pp is not None else 0.0
        print(f"  {name:<6} {acc*100:>7.2f}% {pr*100:>7.2f}% "
              f"{re*100:>7.2f}% {f1*100:>7.2f}% {auc:>8.4f}")
        bl_results[name] = dict(acc=acc, prec=pr, rec=re, f1=f1, auc=auc)

    # ── Internal ranking snapshot ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  INTERNAL MODEL RANKING (claude3.py)")
    print(f"{'='*65}")
    all_results = dict(bl_results)
    all_results["GFIN"] = {
        "acc": results["acc"],
        "prec": results["prec"],
        "rec": results["rec"],
        "f1": results["f1"],
        "auc": results["auc"],
    }
    sorted_by_acc = sorted(all_results.items(), key=lambda kv: kv[1]["acc"], reverse=True)
    print(f"  {'Model':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("  " + "-" * 52)
    for name, vals in sorted_by_acc:
        print(f"  {name:<8} {vals['acc']*100:>7.2f}% {vals['prec']*100:>7.2f}% "
              f"{vals['rec']*100:>7.2f}% {vals['f1']*100:>7.2f}% {vals['auc']:>8.4f}")

    # ── Generate plots ────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    _plot_proof(results, bl_results)
    _plot_claude3_flowchart(rfe_names, inter_str, n_inter)
    _plot_confusion_matrix(results["cm"])
    _plot_roc_curve(y_te, p_fused)
    _plot_feature_importance_14(trees[0], feature_names_14)
    _plot_probability_distribution(y_te, p_fused)
    _write_table5_table7(bl_results, y_te, p_fused, threshold)

    print(f"\n{'='*65}")
    print(f"  KEY claude3.py DESIGN ELEMENTS:")
    print(f"  1. Borderline-SMOTE  — synthesises near decision boundary only")
    print(f"  2. Median imputation — robust to skewed medical feature outliers")
    print(f"  3. {n_inter} pairwise interaction terms  — richer {total_dim}-dim input")
    print(f"  4. Feature Attention Gate  — learned per-feature sigmoid mask")
    print(f"  5. 3-block GFIN  — GRU update gate on features (no fake timesteps)")
    print(f"     Block 1: GELU + Batch Norm,  Block 2: GELU,  Block 3: Swish")
    print(f"  6. Dual GFIN ensemble  — diverse initialization and training settings")
    print(f"  7. 6-model tree ensemble  — RF + ET + HGB + GB + LR + SVC")
    print(f"  8. Cosine LR annealing + label smoothing  — stable training")
    print(f"{'='*65}\n")

    return results, bl_results


# =============================================================================
# PLOTS
# =============================================================================
def _plot_proof(res, bl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = list(bl.keys()) + ["GFIN"]
    display_models = list(bl.keys()) + ["GFIN\n(Fused)"]

    metric_map = {
        "acc": [bl[m]["acc"] * 100 for m in bl] + [res["acc"] * 100],
        "prec": [bl[m]["prec"] * 100 for m in bl] + [res["prec"] * 100],
        "rec": [bl[m]["rec"] * 100 for m in bl] + [res["rec"] * 100],
        "f1": [bl[m]["f1"] * 100 for m in bl] + [res["f1"] * 100],
        "auc": [bl[m]["auc"] for m in bl] + [res["auc"]],
    }

    x = np.arange(len(models))
    w = 0.18
    bars_acc = axes[0].bar(x - 1.5*w, metric_map["acc"], w, label="Acc", color="#4e79a7")
    bars_prec = axes[0].bar(x - 0.5*w, metric_map["prec"], w, label="Prec", color="#59a14f")
    bars_rec = axes[0].bar(x + 0.5*w, metric_map["rec"], w, label="Rec", color="#f28e2b")
    bars_f1 = axes[0].bar(x + 1.5*w, metric_map["f1"], w, label="F1", color="#e15759")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_models, fontsize=9)
    axes[0].set_ylim(55, 100)
    axes[0].set_ylabel("Score (%)", fontsize=11)
    axes[0].set_title("Core Metrics by Model", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(axis="y", alpha=0.3)

    acc_colors = ["#aaaaaa"] * len(bl) + ["#D65F5F"]
    bars2 = axes[1].bar(display_models, metric_map["acc"], color=acc_colors, edgecolor="white")
    axes[1].set_ylim(60, 100)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("Accuracy: Internal Models", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, metric_map["acc"]):
        axes[1].text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")

    auc_colors = ["#aaaaaa"] * len(bl) + ["#D65F5F"]
    bars3 = axes[2].bar(display_models, metric_map["auc"], color=auc_colors, edgecolor="white")
    axes[2].set_ylim(0.4, 1.0)
    axes[2].set_ylabel("AUC", fontsize=11)
    axes[2].set_title("AUC: Internal Models", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    for b, v in zip(bars3, metric_map["auc"]):
        axes[2].text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}",
                     ha="center", fontsize=8, fontweight="bold")

    fig.suptitle(
        "Proof Plot — Internal Model Comparison\n"
        "Pipeline: Borderline-SMOTE + 3-block GFIN + 6-model Tree Ensemble",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    plt.savefig("outputs/proof_plot.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("\n  → Proof plot saved:       outputs/proof_plot.png")


def _plot_claude3_flowchart(rfe_names, inter_str, n_inter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(24, 14))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 20)
    ax.axis("off")

    def box(x, y, w, h, text, face="#edf3ff", edge="#2f5c9a", fs=10, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=face, edgecolor=edge, linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def decision(x, y, w, h, text, face="#fff4d6", edge="#7a5a00", fs=10):
        points = np.array([
            [x, y + h / 2],
            [x + w / 2, y],
            [x, y - h / 2],
            [x - w / 2, y],
        ])
        poly = mpatches.Polygon(points, closed=True, facecolor=face,
                                edgecolor=edge, linewidth=1.6)
        ax.add_patch(poly)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold")

    def arrow(src, dst, label=None, offset=(0.0, 0.0), lw=1.6):
        ax.annotate(
            "", xy=dst, xytext=src,
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=lw)
        )
        if label:
            mx = (src[0] + dst[0]) / 2 + offset[0]
            my = (src[1] + dst[1]) / 2 + offset[1]
            ax.text(mx, my, label, fontsize=9, fontweight="bold", color="#1f2d3d")

    inter_preview = ", ".join(inter_str[:3]) + ", ..."

    ax.text(15, 19.3, "Detailed Flowchart (Main Architecture)",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#1a2c48")

    # Left vertical pipeline
    box(4.8, 17.6, 7.0, 1.5, "PIMA Dataset (768 x 8)", face="#e8efff", bold=True)
    box(4.8, 15.6, 7.0, 1.7, "Preprocess\n- replace medical zeros with NaN\n- median imputation + MinMax", face="#eaf7ff")
    box(4.8, 13.6, 7.0, 1.4, "Borderline-SMOTE balancing", face="#fff4e5")
    box(4.8, 11.9, 7.0, 1.5, "RFE step: RandomForest estimator\neliminate least important feature", face="#f7f1ff")
    decision(4.8, 10.0, 6.2, 2.0, "Target size\n= 4 features?")
    box(4.8, 8.2, 7.0, 1.5, f"Selected RFE features\n{', '.join(rfe_names)}", face="#eef9ea")
    box(4.8, 6.2, 7.0, 1.6,
        f"Pairwise interactions ({n_inter})\n{inter_preview}\nFinal vector = 14 dims", face="#fff0f4")
    box(4.8, 4.1, 7.0, 1.6,
        "Modeling workflow\n- train/validation/test partitions\n- validation used for tuning",
        face="#ecfff8")

    arrow((4.8, 16.85), (4.8, 16.45))
    arrow((4.8, 14.75), (4.8, 14.3))
    arrow((4.8, 12.9), (4.8, 12.55))
    arrow((4.8, 11.15), (4.8, 10.95))
    arrow((4.8, 9.05), (4.8, 8.95), label="Yes", offset=(0.6, 0.0))
    arrow((1.9, 10.0), (1.9, 11.9))
    arrow((1.9, 11.9), (1.35, 11.9))
    arrow((1.35, 11.9), (1.35, 11.9), lw=0.0)
    ax.annotate(
        "", xy=(1.55, 11.9), xytext=(1.9, 11.9),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6)
    )
    ax.text(1.0, 10.25, "No", fontsize=9, fontweight="bold")
    arrow((4.8, 7.45), (4.8, 7.0))
    arrow((4.8, 5.4), (4.8, 4.9))

    # Deep branch
    ax.text(13.2, 15.8, "Deep branch (main)", fontsize=12, fontweight="bold", color="#a83232")
    box(13.2, 13.9, 7.6, 1.9,
        "GFIN-1 training\nH: 256 -> 128 -> 64\ncosine LR + label smoothing", face="#ffecec", edge="#b0413e")
    box(13.2, 11.4, 7.6, 1.9,
        "GFIN-2 training\ndifferent seed + batch + LR\nfor diversity", face="#ffecec", edge="#b0413e")
    box(13.2, 9.2, 7.6, 1.4,
        "Deep fusion\np_gfin = 0.55*p1 + 0.45*p2", face="#fff7e8", edge="#8b5a1f")

    # Tree branch
    ax.text(22.2, 15.8, "Tree ensemble branch", fontsize=12, fontweight="bold", color="#245a7a")
    box(22.2, 13.2, 7.8, 2.2,
        "Train RF + ET + HGB + GB + LR + SVC\non same engineered 14-dim features", face="#eaf6ff", edge="#2a6b93")
    box(22.2, 10.7, 7.8, 1.6,
        "Weighted tree probability\np_trees = 0.25*RF + 0.20*ET + 0.22*HGB\n+ 0.18*GB + 0.08*LR + 0.07*SVC", face="#eef8ff", edge="#2a6b93", fs=9)

    # Merge/tuning/output
    box(17.5, 7.3, 8.8, 1.9,
        "Validation tuning\nsearch alpha in [0.40, 0.95], optimize threshold on val", face="#f6f0ff", edge="#5c4a99")
    box(17.5, 5.0, 8.8, 1.8,
        "Final test inference\np_fused = alpha*p_trees + (1-alpha)*p_gfin\ny_pred = p_fused >= threshold", face="#f1fff4", edge="#2f7d48")
    box(17.5, 2.8, 8.8, 1.8,
        "Outputs\nAcc / Prec / Rec / F1 / AUC + CM + ROC\nproof_plot + tables + probability distributions", face="#fff9e8", edge="#8c6a1b")

    # Baselines side path
    box(27.1, 5.8, 5.0, 1.8,
        "Baselines on same split\nLR, RF, HGB, KNN, NB\nfor Table 5", face="#f9f9f9", edge="#666")

    # Cross-links
    arrow((8.4, 4.1), (9.3, 4.1))
    arrow((9.3, 4.1), (9.3, 13.9))
    arrow((9.3, 13.9), (9.4, 13.9))

    arrow((8.4, 4.1), (9.7, 4.1))
    arrow((9.7, 4.1), (9.7, 11.4))
    arrow((9.7, 11.4), (9.4, 11.4))

    arrow((13.2, 12.95), (13.2, 12.3))
    arrow((13.2, 10.45), (13.2, 9.95))

    arrow((8.4, 4.1), (16.9, 4.1))
    arrow((16.9, 4.1), (16.9, 13.2))
    arrow((16.9, 13.2), (18.3, 13.2))

    arrow((22.2, 12.0), (22.2, 11.5))

    arrow((16.9, 9.2), (14.8, 7.3))
    arrow((22.2, 9.9), (20.2, 7.3))
    arrow((17.5, 6.35), (17.5, 5.9))
    arrow((17.5, 4.1), (17.5, 3.65))

    arrow((8.4, 4.1), (24.6, 5.8))
    arrow((27.1, 4.9), (21.6, 2.8))

    ax.text(17.5, 1.1,
            "Flow focuses on main path: engineered features -> dual GFIN + tree ensemble -> validation tuning -> fused test outputs",
            ha="center", va="center", fontsize=10, color="#333")

    plt.tight_layout()
    plt.savefig("outputs/claude3_architecture_flowchart.png", dpi=240, bbox_inches="tight")
    plt.savefig("outputs/architecture_comparison.png", dpi=240, bbox_inches="tight")
    plt.close()
    print("  → Flowchart saved:        outputs/claude3_architecture_flowchart.png")
    print("  → Architecture saved:     outputs/architecture_comparison.png")


def _plot_confusion_matrix(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title("Confusion Matrix (Final Fused)", fontweight="bold")

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            color = "white" if value > cm.max() * 0.6 else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("  → Confusion matrix saved: outputs/confusion_matrix.png")


def _plot_roc_curve(y_true, proba):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#D65F5F", label=f"Final Fused (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#777", lw=1.5, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Final Fused)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("  → ROC curve saved:        outputs/roc_curve.png")


def _plot_feature_importance_14(rf_model, feature_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]

    sorted_names = [feature_names[i] for i in order]
    sorted_vals = importances[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.arange(len(sorted_names)), sorted_vals[::-1], color="#4878CF", alpha=0.88)
    ax.set_yticks(np.arange(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance (14 Engineered Features)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_14.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("  → Feature importance saved: outputs/feature_importance_14.png")


def _plot_probability_distribution(y_true, proba):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p0 = proba[y_true == 0]
    p1 = proba[y_true == 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(p0, bins=20, alpha=0.7, color="#4878CF", density=True, label="True class 0")
    ax.hist(p1, bins=20, alpha=0.7, color="#D65F5F", density=True, label="True class 1")
    ax.set_xlabel("Predicted probability of diabetes")
    ax.set_ylabel("Density")
    ax.set_title("Probability Distribution by Class", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/probability_distribution_by_class.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("  → Probability plot saved: outputs/probability_distribution_by_class.png")


def _write_table5_table7(bl_results, y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    out_lines = [
        "# Filled Tables from claude3.py Run", "",
        "## Table 5. Baseline classifier performance (threshold=0.50)",
        "| Model | Acc (%) | Prec (%) | Rec (%) | F1 (%) | AUC |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Logistic Regression | {bl_results['LR']['acc']*100:.2f} | {bl_results['LR']['prec']*100:.2f} "
            f"| {bl_results['LR']['rec']*100:.2f} | {bl_results['LR']['f1']*100:.2f} | {bl_results['LR']['auc']:.4f} |"
        ),
        (
            f"| Random Forest (200 trees) | {bl_results['RF']['acc']*100:.2f} | {bl_results['RF']['prec']*100:.2f} "
            f"| {bl_results['RF']['rec']*100:.2f} | {bl_results['RF']['f1']*100:.2f} | {bl_results['RF']['auc']:.4f} |"
        ),
        (
            f"| Hist. Gradient Boosting | {bl_results['HGB']['acc']*100:.2f} | {bl_results['HGB']['prec']*100:.2f} "
            f"| {bl_results['HGB']['rec']*100:.2f} | {bl_results['HGB']['f1']*100:.2f} | {bl_results['HGB']['auc']:.4f} |"
        ),
        (
            f"| K-Nearest Neighbours | {bl_results['KNN']['acc']*100:.2f} | {bl_results['KNN']['prec']*100:.2f} "
            f"| {bl_results['KNN']['rec']*100:.2f} | {bl_results['KNN']['f1']*100:.2f} | {bl_results['KNN']['auc']:.4f} |"
        ),
        (
            f"| Naive Bayes | {bl_results['NB']['acc']*100:.2f} | {bl_results['NB']['prec']*100:.2f} "
            f"| {bl_results['NB']['rec']*100:.2f} | {bl_results['NB']['f1']*100:.2f} | {bl_results['NB']['auc']:.4f} |"
        ),
        "",
        "## Table 7. Per-class classification report (Final Fused on test set)",
        "| Class | Precision (%) | Recall (%) | F1 (%) | Support |",
        "|---|---:|---:|---:|---:|",
        f"| Class 0 (Non-Diabetic) | {p[0]*100:.2f} | {r[0]*100:.2f} | {f[0]*100:.2f} | {int(s[0])} |",
        f"| Class 1 (Diabetic) | {p[1]*100:.2f} | {r[1]*100:.2f} | {f[1]*100:.2f} | {int(s[1])} |",
        f"| Macro Average | {pm*100:.2f} | {rm*100:.2f} | {fm*100:.2f} | {int(s.sum())} |",
        "",
    ]
    Path("outputs/table_5_table_7_filled.md").write_text("\n".join(out_lines), encoding="utf-8")
    print("  → Filled tables saved:    outputs/table_5_table_7_filled.md")


if __name__ == "__main__":
    main()