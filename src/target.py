"""target.py -- the model whose privacy we attack (RandomForest; max_depth = the
overfitting / model-complexity knob)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_target(X, y, members, max_depth=None, n_estimators=200, seed=0):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=seed, n_jobs=-1)
    clf.fit(X[members], y[members])
    return clf


def utility(clf, X, y, members, nonmembers) -> dict:
    return dict(
        train_acc=float(accuracy_score(y[members], clf.predict(X[members]))),
        test_acc=float(accuracy_score(y[nonmembers], clf.predict(X[nonmembers]))),
    )
