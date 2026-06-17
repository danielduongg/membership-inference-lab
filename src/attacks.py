"""
attacks.py -- membership-inference attacks and metrics.

Two attacks:
  * naive   -- threshold the model's confidence (log-prob it assigns to the TRUE
               label). Members tend to get higher confidence than non-members.
  * shadow  -- the Shokri et al. attack: train K "shadow" models on known in/out
               splits of auxiliary (pool) data, learn what a member's confidence
               vector looks like, and apply that learned attacker to the target.

We report ROC-AUC, balanced accuracy, and -- crucially -- TPR @ 1% FPR. As Carlini et
al. argue, average-case metrics understate privacy risk; what matters is how confidently
an attacker can identify *some* training members at a low false-positive rate.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from .target import train_target


def tpr_at_fpr(labels, scores, fpr_target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    k = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(k, 0)])


def _metrics(labels, scores) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    auc = float(roc_auc_score(labels, scores))
    thr = np.median(scores)
    bal = float(balanced_accuracy_score(labels, (scores >= thr).astype(int)))
    fpr, tpr, _ = roc_curve(labels, scores)
    return dict(auc=auc, balanced_acc=bal,
                tpr_at_1pct_fpr=tpr_at_fpr(labels, scores, 0.01),
                tpr_at_10pct_fpr=tpr_at_fpr(labels, scores, 0.10),
                roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()})


def _true_class_logprob(model, X, y):
    P = model.predict_proba(X)
    return np.log(P[np.arange(len(X)), y] + 1e-12)


def _attack_features(P, y):
    tp = P[np.arange(len(P)), y]
    ent = -(P * np.log(P + 1e-12)).sum(1)
    srt = np.sort(P, 1)
    margin = srt[:, -1] - srt[:, -2]
    return np.column_stack([np.log(tp + 1e-12), ent, srt[:, -1], margin])


def naive_attack(model, X, y, members, nonmembers) -> dict:
    s = _true_class_logprob(model, X, y)
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    scores = np.concatenate([s[members], s[nonmembers]])
    return _metrics(labels, scores)


def shadow_attack(model, X, y, members, nonmembers, pool,
                  max_depth=None, n_shadows: int = 10, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    n_in = len(members)
    AX, AY = [], []
    for k in range(n_shadows):
        p = rng.permutation(pool)
        s_in, s_out = p[:n_in], p[n_in:2 * n_in]
        sm = train_target(X, y, s_in, max_depth=max_depth, seed=k + 1)
        AX.append(_attack_features(sm.predict_proba(X[s_in]), y[s_in]))
        AY.append(np.ones(len(s_in)))
        AX.append(_attack_features(sm.predict_proba(X[s_out]), y[s_out]))
        AY.append(np.zeros(len(s_out)))
    atk = LogisticRegression(max_iter=2000).fit(np.vstack(AX), np.concatenate(AY))

    ft = np.vstack([_attack_features(model.predict_proba(X[members]), y[members]),
                    _attack_features(model.predict_proba(X[nonmembers]), y[nonmembers])])
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    scores = atk.predict_proba(ft)[:, 1]
    return _metrics(labels, scores)
