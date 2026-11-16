#!/usr/bin/env python3
"""
plot_roc_mc.py -- ROC curves of the multiclass model written by train_deeptau_mc.py.

  python plot_roc_mc.py mc_deeptau_hl_test_scores.npz
  python plot_roc_mc.py mc_deeptau_hl_test_scores.npz --no-show      # only save the PNGs

Two figures (saved next to the scores file):
  <name>_roc_tau_vs_each.png   tau efficiency vs the efficiency for jet / muon / electron, using the
                               discriminator D = p_tau / (p_tau + p_other) on tau and <other> events only
                               (linear axes on the left, log x-axis on the right)
  <name>_roc_one_vs_rest.png   one-vs-rest ROC of each of the four classes (score = its own probability)
Needs: numpy, matplotlib.
"""
import argparse

import numpy as np


def roc(score, y):
    """(fpr, tpr, auc) with y == 1 the signal"""
    o = np.argsort(-score, kind="stable")
    ys = y[o]
    tpr = np.concatenate([[0], np.cumsum(ys == 1) / max((ys == 1).sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(ys == 0) / max((ys == 0).sum(), 1)])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", nargs="?", default="mc_deeptau_test_scores.npz")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening a window")
    a = ap.parse_args()

    import matplotlib
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.scores)
    prob, y = z["prob"], z["label"]
    names = [str(n) for n in z["class_names"]]
    tau = names.index("tau")
    base = a.scores[:-4] if a.scores.endswith(".npz") else a.scores

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    print("tau vs each class (D = p_tau / (p_tau + p_other)):")
    for c, name in enumerate(names):
        if c == tau:
            continue
        m = (y == tau) | (y == c)
        d = prob[m, tau] / np.maximum(prob[m, tau] + prob[m, c], 1e-12)
        fpr, tpr, auc = roc(d, (y[m] == tau).astype(np.int64))
        print("  tau vs %-9s AUC = %.4f   (%d tau, %d %s events)" % (name, auc, (y == tau).sum(), (y == c).sum(), name))
        for ax in axes:
            ax.plot(fpr, tpr, label="tau vs %s (AUC = %.4f)" % (name, auc))
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="random")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-4, 1); axes[1].set_ylim(0, 1)
    for ax, title in zip(axes, ("linear", "log x-axis")):
        ax.set_xlabel("background efficiency (false positive rate)")
        ax.set_ylabel("tau efficiency (true positive rate)")
        ax.set_title("tau vs each class (%s)" % title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    f1 = base + "_roc_tau_vs_each.png"
    fig.savefig(f1, dpi=300)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    print("one-vs-rest:")
    for c, name in enumerate(names):
        fpr, tpr, auc = roc(prob[:, c], (y == c).astype(np.int64))
        print("  %-9s AUC = %.4f" % (name, auc))
        ax2.plot(fpr, tpr, label="%s vs rest (AUC = %.4f)" % (name, auc))
    ax2.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xlabel("false positive rate"); ax2.set_ylabel("true positive rate")
    ax2.set_title("One-vs-rest ROC")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="lower right")
    fig2.tight_layout()
    f2 = base + "_roc_one_vs_rest.png"
    fig2.savefig(f2, dpi=300)
    print("saved %s and %s" % (f1, f2))
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
