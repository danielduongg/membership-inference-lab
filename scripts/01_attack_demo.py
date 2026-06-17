"""Step 1: attack one overfit target with the naive AND shadow attacks; log-ROC."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import naive_attack, shadow_attack  # noqa: E402
from src.data import make_dataset, membership_split  # noqa: E402
from src.plots import plot_log_roc  # noqa: E402
from src.target import train_target, utility  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)
DEPTH = 8

X, y = make_dataset()
sp = membership_split(len(X))
clf = train_target(X, y, sp["members"], max_depth=DEPTH)
util = utility(clf, X, y, sp["members"], sp["nonmembers"])

naive = naive_attack(clf, X, y, sp["members"], sp["nonmembers"])
shadow = shadow_attack(clf, X, y, sp["members"], sp["nonmembers"], sp["pool"],
                       max_depth=DEPTH, n_shadows=10)

out = dict(max_depth=DEPTH, utility=util,
           naive={k: naive[k] for k in ["auc", "balanced_acc", "tpr_at_1pct_fpr", "tpr_at_10pct_fpr"]},
           shadow={k: shadow[k] for k in ["auc", "balanced_acc", "tpr_at_1pct_fpr", "tpr_at_10pct_fpr"]})
json.dump(out, open(RES / "attack_demo.json", "w"), indent=2)
plot_log_roc([("Naive (confidence threshold)", naive["roc"], naive["auc"]),
              ("Shadow models (Shokri)", shadow["roc"], shadow["auc"])],
             RES / "figures" / "log_roc.png")

print(f"Target RandomForest(max_depth={DEPTH}): "
      f"train={util['train_acc']:.3f} test={util['test_acc']:.3f} "
      f"gap={util['train_acc']-util['test_acc']:+.3f}")
print(f"  NAIVE  AUC={naive['auc']:.3f}  balacc={naive['balanced_acc']:.3f}  "
      f"TPR@1%FPR={naive['tpr_at_1pct_fpr']:.3f}")
print(f"  SHADOW AUC={shadow['auc']:.3f}  balacc={shadow['balanced_acc']:.3f}  "
      f"TPR@1%FPR={shadow['tpr_at_1pct_fpr']:.3f}")
