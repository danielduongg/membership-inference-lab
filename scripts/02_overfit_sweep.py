"""Step 2: sweep model complexity (max_depth); leakage tracks the overfitting gap."""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import naive_attack  # noqa: E402
from src.data import make_dataset, membership_split  # noqa: E402
from src.plots import plot_leakage_vs_complexity  # noqa: E402
from src.target import train_target, utility  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

X, y = make_dataset()
sp = membership_split(len(X))
DEPTHS = [3, 5, 8, 12, None]

rows, labels, auc_l, tpr_l, gap_l = [], [], [], [], []
for md in DEPTHS:
    clf = train_target(X, y, sp["members"], max_depth=md)
    u = utility(clf, X, y, sp["members"], sp["nonmembers"])
    a = naive_attack(clf, X, y, sp["members"], sp["nonmembers"])
    gap = u["train_acc"] - u["test_acc"]
    lab = "None" if md is None else str(md)
    rows.append((lab, u["train_acc"], u["test_acc"], gap, a["auc"], a["tpr_at_1pct_fpr"]))
    labels.append("∞" if md is None else str(md))
    auc_l.append(a["auc"]); tpr_l.append(a["tpr_at_1pct_fpr"]); gap_l.append(gap)
    print(f"max_depth={lab:<4} train={u['train_acc']:.3f} test={u['test_acc']:.3f} "
          f"gap={gap:+.3f}  AUC={a['auc']:.3f}  TPR@1%FPR={a['tpr_at_1pct_fpr']:.3f}")

with open(RES / "overfit_sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["max_depth", "train_acc", "test_acc", "gap", "attack_auc", "tpr_at_1pct_fpr"])
    w.writerows(rows)
plot_leakage_vs_complexity(labels, auc_l, tpr_l, gap_l,
                           RES / "figures" / "leakage_vs_complexity.png")
print("wrote results/overfit_sweep.csv + figures/leakage_vs_complexity.png")
