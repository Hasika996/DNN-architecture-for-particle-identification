#!/usr/bin/env python3
"""
smoke_test_mc.py -- end-to-end check of the multiclass pipeline on FABRICATED data.

Writes toy mc_hl_*.npz and grids_*.npz files for the four samples (jet, tau, muon, electron) directly
(no ROOT needed), then runs train_deeptau_mc.py in --mode hl and --mode full and plot_roc_mc.py. It only
proves that the plumbing works and the network can learn something; the numbers mean nothing physically.

    python smoke_test_mc.py              # needs torch; a few minutes on a CPU
    python smoke_test_mc.py --dry-run    # no torch: only checks the data path (loading, rows, normalisation)
Needs the files build_grids.py, export_hl_mc.py, train_deeptau_mc.py, plot_roc_mc.py in the same folder.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

import build_grids as bg

try:
    import export_hl_mc as eh
except ImportError:
    import export_hl as eh

SAMPLES = ["qcd", "tau", "mumu", "ee"]
N_PER_SAMPLE = {"qcd": 900, "tau": 600, "mumu": 500, "ee": 400}


def toy_sample(out_dir, key, n, rng):
    """one sample: hl npz (like export_hl_mc.py) + grids npz (like build_grids.py), n tree entries each"""
    names = list(eh.HL_TAU)
    ev_list, ctr = [], []
    hl = np.zeros((n, len(names)), np.float32)
    selected = np.ones(n, np.int8)
    for i in range(n):
        ctr_pt = float(rng.uniform(25, 150))
        ctr.append(ctr_pt)
        if key == "tau":
            k, width, ty = rng.randint(2, 6), 0.03, rng.choice([0, 1, 2], 5, p=[0.45, 0.2, 0.35])
        elif key == "qcd":
            k, width = rng.randint(8, 40), 0.15
            ty = rng.choice([0, 1, 2, 3, 4], k, p=[0.4, 0.25, 0.3, 0.03, 0.02])
        elif key == "mumu":
            k, width = rng.randint(2, 6), 0.03
            ty = np.concatenate([[6], rng.choice([0, 2, 4], k - 1)])
        else:   # ee
            k, width = rng.randint(2, 6), 0.03
            ty = np.concatenate([[5], rng.choice([0, 2, 3], k - 1)])
        k = len(ty) if key == "tau" else k
        ty = np.asarray(ty)[:k]
        feat = {f: np.zeros(k, np.float32) for f in bg.FEATURES}
        feat["pt"] = (ctr_pt * rng.dirichlet(np.ones(k))).astype(np.float32) + 0.1
        feat["deta"] = rng.normal(0, width, k).astype(np.float32)
        feat["dphi"] = rng.normal(0, width, k).astype(np.float32)
        feat["charge"] = np.where(np.isin(ty, [0, 3, 4, 5, 6]), rng.choice([-1, 1], k), 0).astype(np.float32)
        feat["fromPV"] = rng.randint(0, 4, k).astype(np.float32)
        feat["dxy"] = rng.normal(0, 0.02, k).astype(np.float32)
        feat["hcalFraction"] = rng.uniform(0, 1, k).astype(np.float32)
        ev = {"gc_" + f: feat[f] for f in bg.FEATURES}
        ev["gc_type"] = ty.astype(np.int32)
        ev_list.append(ev)
        # high-level variables: standard normal, a few columns with class-dependent shifts
        row = rng.normal(0, 1, len(names)).astype(np.float32)
        row[names.index("tau_pt")] = ctr_pt
        row[names.index("tau_eta")] = rng.uniform(-2.3, 2.3)
        row[names.index("tau_mass")] = abs(rng.normal({"tau": 0.8, "qcd": 3.0, "mumu": 0.2, "ee": 0.4}[key], 0.4))
        row[names.index("chargedIsoPtSum")] = abs(rng.normal({"tau": 0.5, "qcd": 6.0, "mumu": 0.3, "ee": 0.4}[key], 1.0))
        row[names.index("rho")] = rng.uniform(5, 30)
        for nm in ("tau_n_charged_prongs", "tau_n_neutral_prongs", "tau_dxy_valid", "tau_inside_ecal_crack",
                   "tau_ip3d_valid", "tau_dz_sig_valid", "tau_n_photons"):
            row[names.index(nm)] = rng.randint(0, 3)
        if rng.rand() < 0.1:          # an event without a tau candidate: not selected, sentinel values
            selected[i] = 0
            rho = row[names.index("rho")]
            row[:] = -999.0
            row[names.index("rho")] = rho
        hl[i] = row
    sp, _ = bg.build_sparse(iter(ev_list))
    sp["run"] = np.ones(n, np.int64); sp["lumi"] = np.ones(n, np.int64); sp["event"] = np.arange(n)
    sp["grid_centre_eta"] = np.zeros(n, np.float32); sp["grid_centre_phi"] = np.zeros(n, np.float32)
    sp["grid_centre_pt"] = np.array(ctr, np.float32)
    sp["grid_centre_is_tau"] = np.ones(n, np.int64)
    sp["feature_names"] = np.array(bg.FEATURES); sp["type_names"] = np.array(bg.TYPE_NAMES)
    np.savez_compressed(os.path.join(out_dir, "grids_%s.npz" % key), **sp)
    np.savez_compressed(os.path.join(out_dir, "mc_hl_%s.npz" % key), hl=hl, names=np.array(names),
                        selected=selected, tau_pt=hl[:, names.index("tau_pt")], tau_eta=hl[:, names.index("tau_eta")],
                        run=sp["run"], lumi=sp["lumi"], event=sp["event"])


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit("FAILED: " + " ".join(cmd))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no torch: only check the data path")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    d = tempfile.mkdtemp()
    rng = np.random.RandomState(3)
    for key in SAMPLES:
        toy_sample(d, key, N_PER_SAMPLE[key], rng)
    print("toy files written to", d)
    train = [sys.executable, os.path.join(here, "train_deeptau_mc.py")]
    hl_args = []
    for key in SAMPLES:
        hl_args += ["--hl-" + key, os.path.join(d, "mc_hl_%s.npz" % key)]
    gr_args = []
    for key in SAMPLES:
        gr_args += ["--grids-" + key, os.path.join(d, "grids_%s.npz" % key)]
    extra = ["--dry-run"] if a.dry_run else []
    run(train + ["--mode", "hl"] + hl_args + ["--epochs", "6", "--batch", "64", "--out", os.path.join(d, "run_hl")] + extra)
    run(train + ["--mode", "full"] + hl_args + gr_args +
        ["--epochs", "2", "--batch", "32", "--scale", "0.25", "--workers", "0", "--out", os.path.join(d, "run_full")] + extra)
    if not a.dry_run:
        run([sys.executable, os.path.join(here, "plot_roc_mc.py"), os.path.join(d, "run_hl_test_scores.npz"), "--no-show"])
    print("SMOKE TEST PASSED (outputs in %s)" % d)


if __name__ == "__main__":
    main()
