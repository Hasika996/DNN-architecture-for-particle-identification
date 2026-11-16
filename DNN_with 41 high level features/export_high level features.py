#!/usr/bin/env python3
"""
export_hl.py -- pull the high-level inputs (41 tau variables; 47 with --jets) of the
tau-centred events out of a WZAnalyzer ROOT file.

Default: 41 columns, same order as hl_tau.npz / hl_qcd.npz (the set the old model used).
With --jets: 47 columns, same order as hl_tau1.npz / hl_qcd1.npz (adds the 6 jet variables).

  python3 export_hl.py --root WZAnalyzer_output_merged_tau.root --out hl_tau.npz
  python3 export_hl.py --root WZAnalyzer_output_merged_qcd.root --out hl_qcd.npz
  python3 export_hl.py --root ... --out ... --jets      # 47 columns

Needs: numpy, uproot, awkward (normal Python 3, not cmsenv's).

Row selection (written as "selected" = 1):
    tauFound == 1 and tauHLIndex >= 0 and grid_centre_is_tau == 1 and grid_centre_pt > 0
    and tau_pt finite and > 0
Only these events have a real tau candidate with real features AND a tau-centred grid.
Events without one carry the -999 sentinel; training on them would let the network
separate the classes by "is the sentinel there?", i.e. leak the label.

Per-tau vs per-event branches are detected from the branch structure: a branch with one
entry per tau is read at the analyzer's selected tau (tauHLIndex); a flat branch is read
as is. The script prints which branches were treated as per-tau so you can check them.
"""
import argparse

import numpy as np

HL_NAMES = [
    # --- tau candidate (41) ---
    "tau_pt", "tau_eta", "tau_phi", "tau_mass", "tau_charge", "tau_E_over_pt",
    "chargedIsoPtSum", "chargedIsoPtSumdR03_over_dR05", "footprintCorrection",
    "neutralIsoPtSum", "neutralIsoPtSumWeight_over_neutralIsoPtSum",
    "neutralIsoPtSumWeightdR03_over_neutralIsoPtSum", "neutralIsoPtSumdR03_over_dR05",
    "photonPtSumOutsideSignalCone", "puCorrPtSum",
    "tau_dxy_pca_x", "tau_dxy_pca_y", "tau_dxy_pca_z", "tau_dxy", "tau_dxy_sig",
    "tau_flightLength_x", "tau_flightLength_y", "tau_flightLength_z", "tau_flightLength_sig",
    "tau_n_charged_prongs", "tau_n_neutral_prongs", "tau_dxy_valid", "tau_inside_ecal_crack",
    "tau_ip3d", "tau_ip3d_sig", "tau_ip3d_valid", "tau_dz", "tau_dz_sig", "tau_dz_sig_valid",
    "tau_pt_weighted_deta_strip", "tau_pt_weighted_dphi_strip",
    "tau_pt_weighted_dr_signal", "tau_pt_weighted_dr_iso",
    "tau_leadingTrackNormChi2", "tau_n_photons",
    "rho",
    # --- jet variables (6) ---
    "leadingJetLogThetaJ", "leadingJetLambdaJ", "leadingJetEpsilonJ",
    "nTracks", "tau1", "tau31",
]
assert len(HL_NAMES) == 47 and len(set(HL_NAMES)) == 47
HL_TAU = HL_NAMES[:41]    # default set: 41 tau variables, no jet variables
HL_JET = HL_NAMES[41:]    # the 6 jet variables, only with --jets

META = ["run", "lumi", "event", "tauFound", "tauHLIndex", "grid_centre_is_tau", "grid_centre_pt"]
SENTINEL = -999.0


def export(path, out_path, tree="Events", jets=False):
    names = HL_NAMES if jets else HL_TAU
    import awkward as ak
    import uproot

    with uproot.open(path) as f:
        t = f[tree]
        missing = [b for b in names + META if b not in t.keys()]
        if missing:
            raise SystemExit("branches missing in %s: %s" % (path, missing))
        a = t.arrays(names + META, library="ak")

    n = len(a["run"])
    idx = ak.to_numpy(a["tauHLIndex"]).astype(np.int64)
    hl = np.full((n, len(names)), SENTINEL, dtype=np.float32)
    per_tau = []
    for j, name in enumerate(names):
        v = a[name]
        if v.ndim > 1:  # one entry per tau: take the analyzer's selected tau
            per_tau.append(name)
            sel = ak.firsts(v[ak.local_index(v, axis=1) == idx])
            hl[:, j] = ak.to_numpy(ak.fill_none(sel, SENTINEL)).astype(np.float32)
        else:               # per-event value
            hl[:, j] = ak.to_numpy(v).astype(np.float32)
    print("read at tauHLIndex (per-tau):", per_tau)

    meta = {k: ak.to_numpy(a[k]) for k in META}
    selected = ((meta["tauFound"] == 1) & (idx >= 0) & (meta["grid_centre_is_tau"] == 1)
                & (meta["grid_centre_pt"] > 0))
    ipt, ieta = names.index("tau_pt"), names.index("tau_eta")
    selected &= np.isfinite(hl[:, ipt]) & (hl[:, ipt] > 0)
    np.savez_compressed(out_path, hl=hl, names=np.array(names), selected=selected.astype(np.int8),
                        tau_pt=hl[:, ipt], tau_eta=hl[:, ieta],
                        run=meta["run"], lumi=meta["lumi"], event=meta["event"])
    print("%s: %d events, %d selected (%.1f%%), %d high-level features"
          % (path, n, selected.sum(), 100.0 * selected.mean() if n else 0, len(names)))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default="Events")
    ap.add_argument("--jets", action="store_true", help="also write the 6 jet variables (47 columns)")
    a = ap.parse_args()
    export(a.root, a.out, a.tree, a.jets)