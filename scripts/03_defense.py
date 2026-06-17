"""Step 3: regularization as a defense -- less leakage, at a utility cost."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import naive_attack  # noqa: E402
from src.data import make_dataset, membership_split  # noqa: E402
from src.plots import plot_defense  # noqa: E402
from src.target import train_target, utility  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

X, y = make_dataset()
sp = membership_split(len(X))
configs = [("Unregularized (max_depth=None)", None), ("Regularized (max_depth=4)", 4)]

labels, leak, acc, detail = [], [], [], []
for name, md in configs:
    clf = train_target(X, y, sp["members"], max_depth=md)
    u = utility(clf, X, y, sp["members"], sp["nonmembers"])
    a = naive_attack(clf, X, y, sp["members"], sp["nonmembers"])
    labels.append("Unregularized" if md is None else "Regularized")
    leak.append(a["auc"]); acc.append(u["test_acc"])
    detail.append(dict(config=name, max_depth=("None" if md is None else md),
                       test_acc=u["test_acc"], train_acc=u["train_acc"],
                       attack_auc=a["auc"], tpr_at_1pct_fpr=a["tpr_at_1pct_fpr"]))
    print(f"{name}: test_acc={u['test_acc']:.3f}  attack_AUC={a['auc']:.3f}  "
          f"TPR@1%FPR={a['tpr_at_1pct_fpr']:.3f}")

json.dump(detail, open(RES / "defense.json", "w"), indent=2)
plot_defense(labels, leak, acc, RES / "figures" / "defense.png")
print("wrote results/defense.json + figures/defense.png")
