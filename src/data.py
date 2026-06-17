"""
data.py -- reproducible dataset + membership split.

We use a deliberately *hard* synthetic classification task (many classes, modest
class separation, no label noise). Download-based image/text corpora aren't reachable
offline here, and a hard task is the cleanest way to make a model memorize its training
set -- which is exactly the condition that drives membership-inference leakage. The same
attack code applies unchanged to real models; see the README.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

SEED = 20260617


def make_dataset(n_samples: int = 10000, seed: int = SEED):
    X, y = make_classification(
        n_samples=n_samples, n_features=50, n_informative=12, n_redundant=8,
        n_classes=10, n_clusters_per_class=2, class_sep=0.8, flip_y=0.0,
        random_state=seed)
    return X, y


def membership_split(n: int, n_members: int = 800, n_nonmembers: int = 1000,
                     seed: int = SEED):
    """Disjoint index sets: members (target train), non-members (held out),
    and a shadow pool (auxiliary data the attacker uses to calibrate)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    members = idx[:n_members]
    nonmembers = idx[n_members:n_members + n_nonmembers]
    pool = idx[n_members + n_nonmembers:]
    return {"members": members, "nonmembers": nonmembers, "pool": pool}
