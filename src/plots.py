"""plots.py -- figures for the report (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_leakage_vs_complexity(depth_labels, auc, tpr1, gap, path):
    x = np.arange(len(depth_labels))
    fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
    ax1.plot(x, [a * 100 for a in auc], "o-", color="#c0392b", lw=2, label="Attack ROC-AUC")
    ax1.plot(x, [t * 100 for t in tpr1], "^-", color="#8e44ad", lw=2, label="Attack TPR @ 1% FPR")
    ax1.set_ylabel("Attack metric (%)")
    ax1.set_ylim(0, 102)
    ax2 = ax1.twinx()
    ax2.plot(x, [g * 100 for g in gap], "s--", color="#2980b9", lw=2, label="Train-test gap")
    ax2.set_ylabel("Overfitting: train-test accuracy gap (%)", color="#2980b9")
    ax2.set_ylim(0, 102)
    ax1.set_xticks(x)
    ax1.set_xticklabels(depth_labels)
    ax1.set_xlabel("Model complexity  (RandomForest max_depth)")
    ax1.set_title("Privacy leakage tracks overfitting")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_roc(curves, path):
    """Carlini-style log-scale ROC to spotlight the low-FPR region."""
    plt.figure(figsize=(5.6, 5))
    for label, roc, auc in curves:
        fpr = np.asarray(roc["fpr"])
        tpr = np.asarray(roc["tpr"])
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([1e-3, 1], [1e-3, 1], "--", color="gray", lw=1, label="chance")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-3, 1)
    plt.ylim(1e-3, 1)
    plt.xlabel("False positive rate (log)")
    plt.ylabel("True positive rate (log)")
    plt.title("Membership-inference ROC (low-FPR region matters)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_defense(labels, leak_auc, test_acc, path):
    x = np.arange(len(labels))
    w = 0.38
    plt.figure(figsize=(6, 4.6))
    plt.bar(x - w / 2, [a * 100 for a in leak_auc], w, label="Attack ROC-AUC (leakage)",
            color="#c0392b")
    plt.bar(x + w / 2, [a * 100 for a in test_acc], w, label="Test accuracy (utility)",
            color="#27ae60")
    plt.xticks(x, labels)
    plt.ylabel("Percent")
    plt.ylim(0, 105)
    plt.title("Regularization trades utility for privacy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
