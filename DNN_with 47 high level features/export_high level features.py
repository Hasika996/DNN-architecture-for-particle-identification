#!/usr/bin/env python3
"""
export_hl_mc.py -- pull the high-level inputs (41 tau variables; 47 with --jets) of the
tau-centred events out of a WZAnalyzer ROOT file.

Default: 41 columns (no jet variables).
With --jets: 47 columns (adds the 6 jet variables).

  python3 export_hl_mc.py --root WZAnalyzer_output_merged_tau.root  --out mc_hl_tau.npz  --jets
  python3 export_hl_mc.py --root WZAnalyzer_output_merged_qcd.root  --out mc_hl_qcd.npz  --jets
  python3 export_hl_mc.py --root WZAnalyzer_output_merged_mumu.root --out mc_hl_mumu.npz --jets
  python3 export_hl_mc.py --root WZAnalyzer_output_merged_ee.root   --out mc_hl_ee.npz   --jets

Needs: numpy, uproot, awkward (normal Python 3, not cmsenv's).

Row selection (written as "selected" = 1):
    tauFound == 1 and tauHLIndex >= 0 and grid_centre_is_tau == 1 and grid_centre_pt > 0
    and tau_pt finite and > 0

SENTINELS: the analyzer uses different "not computed" values for some jet variables
(-99 for log(thetaJ) and lambdaJ, -1 for epsilonJ and nTracks). The training script only
treats -999 as missing, so all of them are converted to -999 here.
After exporting, the fraction of missing values per variable (selected rows only) is printed:
compare it between the four samples -- a large difference lets the network separate
classes by "is the value missing?".
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

# The analyzer's own "not computed" values for the jet variables (tau1 / tau31 already use -999).
JET_SENTINELS = {
    "leadingJetLogThetaJ": -99.0,
    "leadingJetLambdaJ": -99.0,
    "leadingJetEpsilonJ": -1.0,
    "nTracks": -1.0,
}


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

    # convert the analyzer's jet-variable sentinels to the common -999 sentinel
    for name, s in JET_SENTINELS.items():
        if name in names:
            j = names.index(name)
            hl[hl[:, j] == np.float32(s), j] = SENTINEL

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

    # fraction of missing (-999 or non-finite) values per variable, selected rows only
    hs = hl[selected]
    if len(hs):
        print("missing fraction per variable (selected rows; only variables with any missing shown):")
        shown = False
        for j, name in enumerate(names):
            frac = float(np.mean(~np.isfinite(hs[:, j]) | (np.abs(hs[:, j] - SENTINEL) < 1e-3)))
            if frac > 0 or name in HL_JET:
                print("  %-48s %.4f" % (name, frac))
                shown = True
        if not shown:
            print("  (none)")
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default="Events")
    ap.add_argument("--jets", action="store_true", help="also write the 6 jet variables (47 columns)")
    a = ap.parse_args()
    export(a.root, a.out, a.tree, a.jets)
