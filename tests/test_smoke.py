"""Fast smoke test: an overfit model leaks membership; regularization reduces it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import naive_attack  # noqa: E402
from src.data import make_dataset, membership_split  # noqa: E402
from src.target import train_target  # noqa: E402


def test_leakage_and_defense():
    X, y = make_dataset(n_samples=4000)
    sp = membership_split(len(X), n_members=400, n_nonmembers=500)
    overfit = train_target(X, y, sp["members"], max_depth=None)
    reg = train_target(X, y, sp["members"], max_depth=3)
    auc_o = naive_attack(overfit, X, y, sp["members"], sp["nonmembers"])["auc"]
    auc_r = naive_attack(reg, X, y, sp["members"], sp["nonmembers"])["auc"]
    assert auc_o > 0.8, auc_o
    assert auc_r < auc_o, (auc_r, auc_o)


if __name__ == "__main__":
    test_leakage_and_defense()
    print("smoke test passed")
