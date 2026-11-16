#!/usr/bin/env python3
"""
Tau-jet vs QCD-jet classification with three classifiers, each tuned with GridSearchCV:

    1. XGBoost            (xgboost.XGBClassifier)
    2. Random Forest      (sklearn RandomForestClassifier)
    3. Gradient Boosting  (sklearn GradientBoostingClassifier)

All three use the same events, the same train/test split, the same CV folds and the same
scoring (ROC AUC), so their 3 ROC curves / 3 AUCs are directly comparable.

Usage
    python bdt_tau_qcd_3models.py
    python bdt_tau_qcd_3models.py --signal "tau/*.root" --bkg "qcd/*.root"
    python bdt_tau_qcd_3models.py --quick              # tiny grids, fast test of the whole chain
    python bdt_tau_qcd_3models.py --max-events 50000   # cap events per class (speeds up GB a lot)

Outputs (in --outdir, default bdt_results/)
    roc_xgboost.png  roc_random_forest.png  roc_gradient_boosting.png   (one ROC per model)
    roc_comparison.png                    all 3 ROCs together (linear and log-x)
    feature_importance.png, confusion_matrices.png, score_distributions.png
    feature_distributions.png             signal vs QCD for every variable (sanity check)
    summary.csv, best_params.json, grid_<model>.csv, <model>.json / .joblib
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("xgboost is not installed:  pip install xgboost")

# ----------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------
SIGNAL_FILES = ["WZAnalyzer_output_merged_tau.root"]   # tau  = label 1
BKG_FILES    = ["WZAnalyzer_output_merged_qcd.root"]   # QCD  = label 0
TREE = "demo/Events"
SEED = 42

# Features, and for each one how to recognise the analyzer's "failed" sentinel value.
FEATURES = ["leadingJetThetaJ", "leadingJetLambdaJ", "leadingJetEpsilonJ",
            "nTracks", "tau1", "tau31"]
PT_BRANCH = "leadingJetPt"      # NOT a feature by default (see --use-pt); still loaded for plots
FAILED = {
    "leadingJetThetaJ":   lambda v: v < 0,       # sentinel -1   (theta_J = 0 is a valid value)
    "leadingJetLambdaJ":  lambda v: v <= -90,    # sentinel -99
    "leadingJetEpsilonJ": lambda v: v < 0,       # sentinel -1
    "nTracks":            lambda v: v < 0,       # sentinel -1
    "tau1":               lambda v: v < 0,       # sentinel -999
    "tau31":              lambda v: v < 0,       # sentinel -999
    PT_BRANCH:            lambda v: v <= 0,
}


# ----------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------
def load_sample(patterns, label, tree_name, branches):
    import uproot  # imported here so the rest of the file can be tested without it
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        sys.exit(f"No files match {patterns}")
    frames = []
    for f in files:
        with uproot.open(f) as fh:
            try:
                tree = fh[tree_name]
            except KeyError:
                sys.exit(f"{f}: tree '{tree_name}' not found. Keys: {fh.keys()}")
            missing = [b for b in branches if b not in tree.keys()]
            if missing:
                sys.exit(f"{f}: missing branches {missing}\nAvailable: {tree.keys()}")
            df = tree.arrays(branches, library="pd")
        print(f"  {os.path.basename(f)}: {len(df):,} events")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = label
    return df


def failure_report(sig, bkg, features):
    """Fraction of events where each variable holds the 'failed' sentinel, per sample."""
    print(f"\n{'variable':<22}{'failed in tau':>15}{'failed in QCD':>15}")
    for c in features:
        fs = FAILED[c](sig[c]).mean() * 100
        fb = FAILED[c](bkg[c]).mean() * 100
        print(f"{c:<22}{fs:>14.2f}%{fb:>14.2f}%")
    print("(a big difference between the two columns means the failure itself separates the classes)")


def balanced_weights(y):
    """Per-event weights giving tau and QCD the same total weight (mean weight ~ 1)."""
    y = np.asarray(y)
    n, n1 = len(y), int(y.sum())
    n0 = n - n1
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


# ----------------------------------------------------------------------------------------
# MODELS AND GRIDS
# ----------------------------------------------------------------------------------------
def get_models(quick):
    if quick:
        xgb_grid = {"n_estimators": [100, 200], "max_depth": [3, 5]}
        rf_grid = {"n_estimators": [100], "max_depth": [6, 10]}
        gb_grid = {"n_estimators": [100], "max_depth": [3, 5]}
    else:
        xgb_grid = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5], "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0], "min_child_weight": [1, 5]}          # 64
        rf_grid = {"n_estimators": [100, 300], "max_depth": [6, 10, 15],
                   "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", None]}       # 36
        gb_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                   "max_depth": [3, 5], "subsample": [0.8, 1.0]}                          # 16
    return {
        "XGBoost": dict(
            slug="xgboost", color="darkorange", grid=xgb_grid,
            est=XGBClassifier(eval_metric="logloss", tree_method="hist",
                              n_jobs=1, random_state=SEED)),
        "Random Forest": dict(
            slug="random_forest", color="seagreen", grid=rf_grid,
            est=RandomForestClassifier(n_jobs=1, random_state=SEED)),
        "Gradient Boosting": dict(
            slug="gradient_boosting", color="crimson", grid=gb_grid,
            est=GradientBoostingClassifier(random_state=SEED)),
    }


def bkg_rejection(fpr, tpr, eff, n_bkg):
    """1 / (QCD mistag rate) at a given tau efficiency.
    If no QCD test event passes, the value is capped at n_bkg (a lower limit)."""
    f = max(float(np.interp(eff, tpr, fpr)), 1.0 / n_bkg)
    return 1.0 / f


# ----------------------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------------------
def plot_single_roc(path, name, res):
    plt.figure(figsize=(6, 5))
    plt.plot(res["fpr"], res["tpr"], color=res["color"],
             label=f"{name} (AUC = {res['auc_test']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate (QCD mistagged as tau)")
    plt.ylabel("True Positive Rate (tau efficiency)")
    plt.title(f"ROC Curve: Tau vs QCD - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_comparison(path, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, logx in zip(axes, (False, True)):
        for name, r in results.items():
            ax.plot(r["fpr"], r["tpr"], color=r["color"],
                    label=f"{name} (AUC = {r['auc_test']:.3f})")
        if not logx:
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate (QCD mistagged as tau)")
        ax.set_ylabel("True Positive Rate (tau efficiency)")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1)
            ax.set_title("ROC comparison (log x: low-mistag region)")
        else:
            ax.set_title("ROC comparison (test set)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_importances(path, results, features):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        imp = np.asarray(r["importances"], dtype=float)
        order = np.argsort(imp)
        ax.barh(np.array(features)[order], imp[order], color=r["color"])
        ax.set_xlabel("Importance")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusions(path, results):
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\n(threshold 0.5)")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["QCD", "Tau"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["QCD", "Tau"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_scores(path, results, y_train, y_test):
    """Train (filled) vs test (outline) score distributions: a gap = overtraining."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5))
    bins = np.linspace(0, 1, 41)
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        for lab, col, txt in ((1, "crimson", "Tau"), (0, "steelblue", "QCD")):
            ax.hist(r["p_train"][y_train == lab], bins=bins, density=True, histtype="stepfilled",
                    alpha=0.35, color=col, label=f"{txt} train")
            ax.hist(r["p_test"][y_test == lab], bins=bins, density=True, histtype="step",
                    lw=1.6, color=col, label=f"{txt} test")
        ax.set_xlabel("BDT score"); ax.set_ylabel("Normalized events")
        ax.set_title(name)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_distributions(path, sig, bkg, cols):
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        s = sig[c][~FAILED[c](sig[c])].to_numpy(float)
        b = bkg[c][~FAILED[c](bkg[c])].to_numpy(float)
        both = np.concatenate([s, b])
        if len(both) == 0:
            ax.set_title(f"{c}: no valid entries")
            continue
        lo, hi = np.percentile(both, [0.5, 99.5])
        if c == "nTracks":
            bins = np.arange(-0.5, min(hi, 40) + 1.5, 1.0)
        else:
            bins = np.linspace(lo, hi if hi > lo else lo + 1, 50)
        ax.hist(s, bins=bins, density=True, histtype="step", lw=1.6, color="crimson", label="Tau")
        ax.hist(b, bins=bins, density=True, histtype="step", lw=1.6, color="steelblue", label="QCD")
        ax.set_title(c)
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", nargs="+", default=SIGNAL_FILES, help="tau file(s) / glob(s)")
    ap.add_argument("--bkg", nargs="+", default=BKG_FILES, help="QCD file(s) / glob(s)")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--outdir", default="bdt_results")
    ap.add_argument("--missing", choices=["keep", "drop"], default="keep",
                    help="keep: events with a failed variable stay, the sentinel value is a "
                         "regular (extreme) number the trees can split on (default).  "
                         "drop: remove every event where ANY variable failed.")
    ap.add_argument("--use-pt", action="store_true",
                    help="also use leadingJetPt as a feature (off by default: it can let the "
                         "BDT learn the sample's pT spectrum instead of tau-vs-QCD substructure)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds in every GridSearchCV")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel grid-search jobs")
    ap.add_argument("--max-events", type=int, default=None, help="cap on events per class")
    ap.add_argument("--quick", action="store_true", help="tiny grids for a fast test")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    features = FEATURES + ([PT_BRANCH] if args.use_pt else [])
    branches = list(dict.fromkeys(features + [PT_BRANCH]))

    # ---------------- load ----------------
    print("Loading signal (tau) ...")
    sig = load_sample(args.signal, 1, args.tree, branches)
    print("Loading background (QCD) ...")
    bkg = load_sample(args.bkg, 0, args.tree, branches)
    print(f"\nRead {len(sig):,} tau and {len(bkg):,} QCD events")

    failure_report(sig, bkg, features)
    plot_feature_distributions(os.path.join(args.outdir, "feature_distributions.png"),
                               sig, bkg, branches)

    # ---------------- clean ----------------
    df = pd.concat([sig, bkg], ignore_index=True)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    n_nan = int(df[features].isna().any(axis=1).sum())
    if n_nan:
        print(f"\nDropping {n_nan:,} events with NaN/inf in a feature")
        df = df.dropna(subset=features)
    if args.missing == "drop":
        failed_any = np.zeros(len(df), dtype=bool)
        for c in features:
            failed_any |= FAILED[c](df[c]).to_numpy()
        print(f"--missing drop: removing {int(failed_any.sum()):,} events with a failed variable "
              f"(tau: {int(failed_any[df['label'].to_numpy() == 1].sum()):,}, "
              f"QCD: {int(failed_any[df['label'].to_numpy() == 0].sum()):,})")
        df = df[~failed_any]
    if args.max_events:
        df = pd.concat([g.sample(min(len(g), args.max_events), random_state=SEED)
                        for _, g in df.groupby("label")])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_tau, n_qcd = int((df["label"] == 1).sum()), int((df["label"] == 0).sum())
    print(f"\nUsing {n_tau:,} tau and {n_qcd:,} QCD events  ({len(features)} features: {features})")
    if min(n_tau, n_qcd) < 500:
        print("WARNING: very few events in one class; CV results will be noisy.")

    X, y = df[features], df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y)
    y_train_a, y_test_a = y_train.to_numpy(), y_test.to_numpy()
    w_train = balanced_weights(y_train_a)
    print(f"Training samples: {len(X_train):,}   Testing samples: {len(X_test):,}")

    # ---------------- tune + evaluate each model ----------------
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    models = get_models(args.quick)
    results, best_params = {}, {}

    for name, spec in models.items():
        n_comb = int(np.prod([len(v) for v in spec["grid"].values()]))
        print(f"\n{'=' * 78}\n{name}: GridSearchCV over {n_comb} combinations x {args.cv} folds\n{'=' * 78}")
        t0 = time.time()
        gs = GridSearchCV(spec["est"], spec["grid"], scoring="roc_auc", cv=cv,
                          n_jobs=args.n_jobs, refit=True, verbose=1, return_train_score=True)
        gs.fit(X_train, y_train, sample_weight=w_train)
        best = gs.best_estimator_
        print(f"done in {time.time() - t0:.0f} s")
        print(f"Best parameters: {gs.best_params_}")
        print(f"Best CV AUC:     {gs.best_score_:.4f}")
        best_params[name] = gs.best_params_

        pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").to_csv(
            os.path.join(args.outdir, f"grid_{spec['slug']}.csv"), index=False)

        p_train = best.predict_proba(X_train)[:, 1]
        p_test = best.predict_proba(X_test)[:, 1]
        pred = (p_test >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test_a, p_test)
        res = dict(
            color=spec["color"], cv_auc=gs.best_score_,
            auc_train=roc_auc_score(y_train_a, p_train),
            auc_test=roc_auc_score(y_test_a, p_test),
            p_train=p_train, p_test=p_test, fpr=fpr, tpr=tpr,
            cm=confusion_matrix(y_test_a, pred),
            importances=best.feature_importances_,
            rej={e: bkg_rejection(fpr, tpr, e, int((y_test_a == 0).sum())) for e in (0.3, 0.5, 0.7, 0.9)},
        )
        results[name] = res

        print(f"Train AUC: {res['auc_train']:.4f}   Test AUC: {res['auc_test']:.4f}")
        if res["auc_train"] - res["auc_test"] > 0.05:
            print("  note: train AUC is much higher than test AUC -> overfitting; "
                  "try shallower trees / fewer estimators")
        print(f"Accuracy: {accuracy_score(y_test_a, pred):.4f}  "
              f"Precision (macro): {precision_score(y_test_a, pred, average='macro'):.4f}  "
              f"Recall (macro): {recall_score(y_test_a, pred, average='macro'):.4f}  "
              f"F1 (macro): {f1_score(y_test_a, pred, average='macro'):.4f}")
        print(f"Confusion matrix (rows = true QCD, tau; threshold 0.5):\n{res['cm']}")

        plot_single_roc(os.path.join(args.outdir, f"roc_{spec['slug']}.png"), name, res)
        if spec["slug"] == "xgboost":
            best.save_model(os.path.join(args.outdir, "xgboost.json"))
        else:
            joblib.dump(best, os.path.join(args.outdir, f"{spec['slug']}.joblib"))

    # ---------------- comparison ----------------
    plot_roc_comparison(os.path.join(args.outdir, "roc_comparison.png"), results)
    plot_importances(os.path.join(args.outdir, "feature_importance.png"), results, features)
    plot_confusions(os.path.join(args.outdir, "confusion_matrices.png"), results)
    plot_scores(os.path.join(args.outdir, "score_distributions.png"), results, y_train_a, y_test_a)

    rows = []
    for name, r in results.items():
        rows.append({"model": name, "cv_auc": r["cv_auc"], "train_auc": r["auc_train"],
                     "test_auc": r["auc_test"],
                     **{f"1/FPR@{int(e * 100)}%": r["rej"][e] for e in r["rej"]}})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "best_params.json"), "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)

    print(f"\n{'=' * 78}\nSUMMARY (test set; 1/FPR = QCD rejection at that tau efficiency, "
          f"capped at {int((y_test_a == 0).sum()):,} = number of QCD test events)\n{'=' * 78}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}" if v < 100 else f"{v:.1f}"))
    print(f"\nPlots, tables and models saved in {args.outdir}/")


if __name__ == "__main__":
    main()