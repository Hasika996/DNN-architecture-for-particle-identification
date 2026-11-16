#!/usr/bin/env python3
"""
build_grids.py -- inner / outer grid inputs for the DeepTau-style DNN
(CMS DeepTau paper, arXiv:2201.08458, Sec. 4.1.1, Fig. 2, Table 2).

Reads candidate-level branches (gc_*) written by the modified WZAnalyzer.cc, bins
candidates into the two grids and writes a SPARSE .npz (only occupied cells are stored).

Memory-efficient ROOT processing:
  * Pass 1 counts the events and occupied (cell, type) slots.
  * Pass 2 writes sparse arrays to temporary disk-backed NumPy memmaps.
  * The memmaps are compressed into the requested .npz file.

Grid geometry:
  inner grid : 11 x 11 cells of 0.02 x 0.02 (cell index -5..+5)
  outer grid : 21 x 21 cells of 0.05 x 0.05 (cell index -10..+10)

Both grids are centred on the grid centre (tau axis, or leading-jet axis if no tau
was found -- see grid_centre_is_tau). Cell 0 is centred on the axis. Per cell and
per particle type only the highest-pT particle is kept. The grids overlap; the
highest-pT rule is applied to each grid separately.

Usage:
  python build_grids_memory_efficient.py --root WZAnalyzer_output_merged_tau.root --out grids_tau.npz
  python build_grids_memory_efficient.py --selftest
  python build_grids_memory_efficient.py --report

Optional ROOT chunk size:
  python build_grids_memory_efficient.py --root input.root --tree demo/Events \\
      --out grids.npz --step-size "20 MB"

Reading a ROOT file needs: pip install uproot numpy

Loading the result:
  sp = np.load("grids_tau.npz")
  x = densify(sp, "inner", rows)
  x = densify(sp, "outer", rows)
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np

# --- must match GridCandType in WZAnalyzer.cc ----------------------------------------
TYPE_NAMES = [
    "ch_hadron", "neutral_hadron", "photon", "pf_electron",
    "pf_muon", "electron", "muon"
]
N_TYPES = len(TYPE_NAMES)

# --- must match GRID_FEAT_NAMES in WZAnalyzer.cc -------------------------------------
FEATURES = [
    "pt", "deta", "dphi", "charge", "fromPV",
    "pvAssociationQuality", "puppiWeight", "puppiWeightNoLep", "lostInnerHits", "numberOfPixelHits",
    "vertex_dx", "vertex_dy", "vertex_dz", "vertex_dx_tauFL", "vertex_dy_tauFL", "vertex_dz_tauFL",
    "dxy", "dxy_sig", "dz", "dz_sig", "track_chi2_ndof", "hcalFraction", "leadChargedHadrCand",
    "ele_hcalOverEcal", "ele_sigmaIetaIeta", "ele_e1x5_over_e5x5", "ele_r9", "ele_sc_etaWidth", "ele_sc_phiWidth",
    "ele_deltaEtaSuperClusterTrackAtVtx", "ele_deltaPhiSuperClusterTrackAtVtx",
    "ele_deltaEtaSeedClusterTrackAtCalo", "ele_deltaPhiSeedClusterTrackAtCalo",
    "ele_deltaEtaEleClusterTrackAtCalo", "ele_deltaPhiEleClusterTrackAtCalo",
    "ele_eSuperClusterOverP", "ele_eSeedClusterOverP",
    "ele_eEleClusterOverPout", "ele_fbrem", "ele_numberOfBrems", "ele_ecalEnergy",
    "ele_gsf_ndof", "ele_gsf_numberOfValidHits", "ele_gsf_numberOfLostHits", "ele_ecalDrivenSeed",
    "mu_globalNormChi2", "mu_innerNumberOfValidHits", "mu_numberOfValidMuonHits", "mu_numberOfMatchedStations",
    "mu_segmentCompatibility", "mu_caloCompatibility",
]
N_FEAT = len(FEATURES)
IDX = {n: i for i, n in enumerate(FEATURES)}
I_PT, I_DETA, I_DPHI = IDX["pt"], IDX["deta"], IDX["dphi"]

# --- features available for each particle type --------------------------------------
_KIN = ["pt", "deta", "dphi"]
_PF_TRACK = [
    "charge", "fromPV", "pvAssociationQuality", "puppiWeight", "puppiWeightNoLep",
    "lostInnerHits", "numberOfPixelHits",
    "vertex_dx", "vertex_dy", "vertex_dz", "vertex_dx_tauFL", "vertex_dy_tauFL", "vertex_dz_tauFL",
    "dxy", "dxy_sig", "dz", "dz_sig", "track_chi2_ndof"
]
_TRK = ["charge", "dxy", "dxy_sig", "dz", "dz_sig", "track_chi2_ndof"]
TYPE_FEATURES = {
    "ch_hadron": _KIN + _PF_TRACK + ["hcalFraction", "leadChargedHadrCand"],
    "neutral_hadron": _KIN + ["puppiWeight", "puppiWeightNoLep", "hcalFraction"],
    "photon": _KIN + _PF_TRACK,
    "pf_electron": _KIN + _PF_TRACK,
    "pf_muon": _KIN + _PF_TRACK,
    "electron": _KIN + _TRK + [n for n in FEATURES if n.startswith("ele_")],
    "muon": _KIN + _TRK + [n for n in FEATURES if n.startswith("mu_")],
}

# The paper's three per-cell blocks (Fig. 3).
BLOCKS = {
    "egamma": ["electron", "pf_electron", "photon"],
    "muon": ["muon", "pf_muon"],
    "hadrons": ["ch_hadron", "neutral_hadron"],
}

# --- grid geometry from the paper ----------------------------------------------------
GRIDS = {"inner": (11, 0.02), "outer": (21, 0.05)}

CAND_BRANCHES = ["gc_type"] + ["gc_" + n for n in FEATURES]
META_BRANCHES = [
    "run", "lumi", "event", "grid_centre_eta", "grid_centre_phi",
    "grid_centre_pt", "grid_centre_is_tau"
]


# =====================================================================================
# Core: one event, one grid
# =====================================================================================
def cell_index(d, size, n_cells):
    """Return cell indices and a mask saying which displacements are inside the grid.

    Cell k covers [(k - 0.5) * size, (k + 0.5) * size), so cell 0 is centred
    on the grid axis.
    """
    half = n_cells // 2
    idx = np.floor(np.asarray(d, dtype=np.float64) / size + 0.5).astype(np.int64)
    return idx, (idx >= -half) & (idx <= half)


def _empty():
    return (np.zeros(0, np.int64),) * 3 + (np.zeros((0, N_FEAT), np.float32), 0)


def grid_slots(ctype, feats, grid):
    """Get occupied (cell, type) slots for one event and one grid.

    Returns ieta, iphi, type, feature matrix, and the number of candidates
    inside the grid window before applying the highest-pT-per-slot rule.
    """
    n_cells, size = GRIDS[grid]
    half = n_cells // 2
    if len(ctype) == 0:
        return _empty()

    ie, ok_e = cell_index(feats[:, I_DETA], size, n_cells)
    ip, ok_p = cell_index(feats[:, I_DPHI], size, n_cells)
    keep = np.nonzero(ok_e & ok_p)[0]
    if keep.size == 0:
        return _empty()

    # Highest pT first; stable sort means the first candidate wins a pT tie.
    order = keep[np.argsort(-feats[keep, I_PT], kind="stable")]
    key = ((ie[order] + half) * n_cells + (ip[order] + half)) * N_TYPES + ctype[order]
    _, first = np.unique(key, return_index=True)
    sel = order[first]

    return (
        ie[sel],
        ip[sel],
        ctype[sel].astype(np.int64),
        feats[sel].astype(np.float32),
        int(keep.size),
    )


# =====================================================================================
# Event conversion and dense conversion helpers
# =====================================================================================
def event_arrays(ev):
    """Convert one event's gc_* arrays into particle types and a feature matrix."""
    ctype = np.asarray(ev["gc_type"], dtype=np.int64)
    feats = np.zeros((len(ctype), N_FEAT), dtype=np.float64)
    for j, name in enumerate(FEATURES):
        feats[:, j] = np.asarray(ev["gc_" + name], dtype=np.float64)
    return ctype, feats


def densify_full(sp, grid, rows):
    """Dense tensor: (len(rows), n, n, N_TYPES, N_FEAT)."""
    n_cells, _ = GRIDS[grid]
    half = n_cells // 2
    rows = np.asarray(rows, dtype=np.int64)

    ev = sp[grid + "_ev"]
    ie = sp[grid + "_ieta"].astype(np.int64)
    ip = sp[grid + "_iphi"].astype(np.int64)
    ty = sp[grid + "_type"].astype(np.int64)
    x = sp[grid + "_x"]

    out = np.zeros((len(rows), n_cells, n_cells, N_TYPES, N_FEAT), dtype=np.float32)
    lo = np.searchsorted(ev, rows, side="left")
    hi = np.searchsorted(ev, rows, side="right")

    for k in range(len(rows)):
        if hi[k] > lo[k]:
            s = slice(lo[k], hi[k])
            out[k, ie[s] + half, ip[s] + half, ty[s], :] = x[s]

    return out


def block_layout(block):
    """Return [(particle type, feature names)] in the block's channel order."""
    return [(t, TYPE_FEATURES[t]) for t in BLOCKS[block]]


def block_channels(block):
    """Return channel names, such as 'ch_hadron/valid' and 'ch_hadron/pt'."""
    names = []
    for t, feats in block_layout(block):
        names += [t + "/valid"] + [t + "/" + f for f in feats]
    return names


def densify(sp, grid, rows, rel_pt=True):
    """Return the three dense DNN input blocks for selected event rows.

    Output is a dict with 'egamma', 'muon', and 'hadrons' arrays. Use this
    function per batch; the full dense dataset can be very large.

    If rel_pt=True, pT channels are divided by the grid-centre pT.
    """
    full = densify_full(sp, grid, rows)

    if rel_pt and "grid_centre_pt" in sp:
        cpt = np.asarray(sp["grid_centre_pt"], dtype=np.float32)[np.asarray(rows, dtype=np.int64)]
        cpt = np.where(cpt > 0, cpt, 1.0).astype(np.float32)
        full = full.copy()
        full[..., I_PT] /= cpt[:, None, None, None]

    out = {}
    for block in BLOCKS:
        parts = []
        for t, feats in block_layout(block):
            ti = TYPE_NAMES.index(t)
            valid = (full[..., ti, I_PT] != 0).astype(np.float32)[..., None]
            parts.append(valid)
            parts.append(full[..., ti, :][..., [IDX[f] for f in feats]])
        out[block] = np.concatenate(parts, axis=-1)

    return out


def zero_columns(sp, chunk_rows=100000):
    """Find features that are zero for every stored particle of each type.

    Work in chunks so this diagnostic does not make a large RAM copy.
    """
    res = {}
    ty = sp["outer_type"]
    x = sp["outer_x"]

    has_particles = np.zeros(N_TYPES, dtype=bool)
    nonzero = np.zeros((N_TYPES, N_FEAT), dtype=bool)

    for start in range(0, len(ty), chunk_rows):
        end = min(start + chunk_rows, len(ty))
        ty_chunk = ty[start:end]
        x_chunk = x[start:end]

        for ti in range(N_TYPES):
            mask = ty_chunk == ti
            if mask.any():
                has_particles[ti] = True
                nonzero[ti] |= np.any(x_chunk[mask] != 0, axis=0)

    for ti, t in enumerate(TYPE_NAMES):
        if not has_particles[ti]:
            res[t] = ["(no particles of this type in the file)"]
        else:
            res[t] = [f for f in TYPE_FEATURES[t] if not nonzero[ti, IDX[f]]]

    return res


def print_summary(stats):
    n = stats["n_events"]
    print(
        "events: %d   candidates in window (all types): %d  (%.1f per event)"
        % (n, stats["n_cand"], stats["n_cand"] / max(n, 1))
    )

    for g, (nc, _) in GRIDS.items():
        slots = stats[g + "_slots"]
        lost = 1.0 - slots / max(stats["n_in_window"][g], 1)
        occ = stats[g + "_cells"] / float(max(n, 1) * nc * nc)
        print(
            "  %-5s %2dx%-2d : %.2f%% of cells occupied, %.1f occupied (cell,type) slots per event, "
            "%.1f%% of in-grid candidates lost to the highest-pT-per-cell rule"
            % (g, nc, nc, 100.0 * occ, slots / max(n, 1), 100.0 * lost)
        )

    print(
        "  (paper, its own tau samples: 1.7% of inner and 7.1% of outer cells occupied; "
        "up to ~10% of occupied cells hold >1 particle of a type)"
    )


def print_report():
    print("per-particle inputs per type (maximum set; 'valid' = occupied-slot flag)")
    for t in TYPE_NAMES:
        print("  %-15s %2d  (valid + %d features)" % (t, 1 + len(TYPE_FEATURES[t]), len(TYPE_FEATURES[t])))

    tot = {b: sum(1 + len(TYPE_FEATURES[t]) for t in ts) for b, ts in BLOCKS.items()}
    print("  per-cell channels per block:", tot, " total:", sum(tot.values()))


# =====================================================================================
# ROOT input
# =====================================================================================
def iter_events(path, tree="Events", step="50 MB"):
    """Yield (event dict, metadata dict) one event at a time from a ROOT tree."""
    import uproot

    with uproot.open(path) as f:
        t = f[tree]
        for chunk in t.iterate(
            CAND_BRANCHES + META_BRANCHES,
            step_size=step,
            library="np",
        ):
            n = len(chunk["run"])
            for i in range(n):
                ev = {k: chunk[k][i] for k in CAND_BRANCHES}
                meta = {k: chunk[k][i] for k in META_BRANCHES}
                yield ev, meta


def allocate_disk_array(tmpdir, name, dtype, shape):
    """Create an array backed by a temporary .npy file instead of RAM."""
    shape = tuple(shape)
    dtype = np.dtype(dtype)

    if int(np.prod(shape, dtype=np.int64)) == 0:
        return np.zeros(shape, dtype=dtype)

    path = os.path.join(tmpdir, name + ".npy")
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def run_root(path, out_path, tree, step_size="50 MB"):
    """Build sparse grid arrays in two passes using disk-backed storage."""

    # ------------------------------------------------------------------------------
    # Pass 1: count all events and the number of stored grid slots.
    # ------------------------------------------------------------------------------
    print("\nPASS 1/2: counting events and occupied grid slots...")

    first_meta = None
    stats = {
        "n_events": 0,
        "n_cand": 0,
        "n_in_window": {g: 0 for g in GRIDS},
    }
    for g in GRIDS:
        stats[g + "_slots"] = 0
        stats[g + "_cells"] = 0

    for ev, meta in iter_events(path, tree, step_size):
        if first_meta is None:
            first_meta = {k: np.asarray(v) for k, v in meta.items()}

        ctype, feats = event_arrays(ev)
        stats["n_events"] += 1
        stats["n_cand"] += len(ctype)

        for g in GRIDS:
            ie, ip, ty, x, n_in = grid_slots(ctype, feats, g)
            stats["n_in_window"][g] += n_in
            stats[g + "_slots"] += len(ie)

            # Count occupied spatial cells (ignoring particle type), event by event.
            if len(ie):
                nc = GRIDS[g][0]
                half = nc // 2
                cell_key = (
                    (ie.astype(np.int64) + half) * nc
                    + (ip.astype(np.int64) + half)
                )
                stats[g + "_cells"] += len(np.unique(cell_key))

    n_events = stats["n_events"]
    if n_events == 0:
        raise RuntimeError("No events found in the ROOT tree.")

    print("Events:", n_events)
    for g in GRIDS:
        print("  %s: %d occupied (cell,type) slots" % (g, stats[g + "_slots"]))

    # Estimate the size of the disk-backed sparse arrays.
    bytes_per_slot = N_FEAT * np.dtype(np.float32).itemsize + 4 + 1 + 1 + 1
    estimated_bytes = sum(stats[g + "_slots"] for g in GRIDS) * bytes_per_slot
    print(
        "Approximate temporary grid-array size: %.2f GiB (excluding metadata and final NPZ)"
        % (estimated_bytes / (1024.0 ** 3))
    )

    # ------------------------------------------------------------------------------
    # Pass 2: allocate disk-backed arrays and fill them sequentially.
    # ------------------------------------------------------------------------------
    print("\nPASS 2/2: writing grid arrays to temporary disk-backed files...")

    output_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="grid_tmp_", dir=output_dir) as tmpdir:
        sp = {"n_events": np.array(n_events)}

        # Sparse arrays for both grids.
        for g in GRIDS:
            n_slots = stats[g + "_slots"]
            sp[g + "_ev"] = allocate_disk_array(tmpdir, g + "_ev", np.int32, (n_slots,))
            sp[g + "_ieta"] = allocate_disk_array(tmpdir, g + "_ieta", np.int8, (n_slots,))
            sp[g + "_iphi"] = allocate_disk_array(tmpdir, g + "_iphi", np.int8, (n_slots,))
            sp[g + "_type"] = allocate_disk_array(tmpdir, g + "_type", np.int8, (n_slots,))
            sp[g + "_x"] = allocate_disk_array(
                tmpdir, g + "_x", np.float32, (n_slots, N_FEAT)
            )

        # Metadata arrays.
        for k in META_BRANCHES:
            sp[k] = allocate_disk_array(
                tmpdir,
                "meta_" + k,
                first_meta[k].dtype,
                (n_events,),
            )

        sp["feature_names"] = np.array(FEATURES)
        sp["type_names"] = np.array(TYPE_NAMES)

        ptr = {g: 0 for g in GRIDS}
        rows_written = 0

        try:
            for row, (ev, meta) in enumerate(iter_events(path, tree, step_size)):
                if row >= n_events:
                    raise RuntimeError("ROOT event count changed between the two passes.")

                for k in META_BRANCHES:
                    sp[k][row] = meta[k]

                ctype, feats = event_arrays(ev)

                for g in GRIDS:
                    ie, ip, ty, x, n_in = grid_slots(ctype, feats, g)
                    start = ptr[g]
                    end = start + len(ie)

                    if end > start:
                        sp[g + "_ev"][start:end] = row
                        sp[g + "_ieta"][start:end] = ie
                        sp[g + "_iphi"][start:end] = ip
                        sp[g + "_type"][start:end] = ty
                        sp[g + "_x"][start:end] = x

                    ptr[g] = end

                rows_written = row + 1
                if rows_written % 10000 == 0:
                    print("  processed %d / %d events" % (rows_written, n_events))

            if rows_written != n_events:
                raise RuntimeError(
                    "ROOT event count changed between passes: counted %d, read %d."
                    % (n_events, rows_written)
                )

            for g in GRIDS:
                if ptr[g] != stats[g + "_slots"]:
                    raise RuntimeError(
                        "%s slot count mismatch: expected %d, wrote %d."
                        % (g, stats[g + "_slots"], ptr[g])
                    )

            # Flush disk-backed arrays before compression.
            for arr in sp.values():
                if isinstance(arr, np.memmap):
                    arr.flush()

            print("\nCompressing arrays into:", out_path)
            np.savez_compressed(out_path, **sp)

            print_summary(stats)
            cen = sp["grid_centre_is_tau"]
            print(
                "grid centre = tau axis in %d / %d events (rest: leading-jet axis)"
                % (int(np.count_nonzero(cen == 1)), len(cen))
            )

            print("features that are 0 for every particle of a type:")
            for t, cols in zero_columns(sp).items():
                print("  %-15s %s" % (t, ", ".join(cols) if cols else "-"))

            print("\nWrote", out_path)

        finally:
            # Windows requires mapped files to be closed before TemporaryDirectory
            # removes the temporary files.
            for arr in sp.values():
                if isinstance(arr, np.memmap):
                    arr.flush()
                    mmap_obj = getattr(arr, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()


# =====================================================================================
# Self-test (independent brute-force reference + geometry + schema check)
# =====================================================================================
def _reference_dense(ctype, feats, grid):
    """Independent, deliberately naive grid implementation used only by self-test."""
    n_cells, size = GRIDS[grid]
    half = n_cells // 2
    out = np.zeros((n_cells, n_cells, N_TYPES, N_FEAT), dtype=np.float32)
    filled = np.zeros((n_cells, n_cells, N_TYPES), dtype=bool)

    order = sorted(range(len(ctype)), key=lambda i: -feats[i, I_PT])
    for i in order:
        de, dp = feats[i, I_DETA], feats[i, I_DPHI]
        ce = cp = None
        for k in range(-half, half + 1):
            if (k - 0.5) * size <= de < (k + 0.5) * size:
                ce = k
            if (k - 0.5) * size <= dp < (k + 0.5) * size:
                cp = k
        if ce is None or cp is None:
            continue
        if not filled[ce + half, cp + half, ctype[i]]:
            filled[ce + half, cp + half, ctype[i]] = True
            out[ce + half, cp + half, ctype[i], :] = feats[i]

    return out


def selftest(cc_path=None):
    import os

    import tempfile as _tempfile

    rng = np.random.RandomState(7)
    fails = 0

    def check(name, cond):
        nonlocal fails
        print("  [%s] %s" % ("ok" if cond else "FAIL", name))
        fails += (not cond)

    # Schema checks.
    print("schema")
    cc_path = cc_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "WZAnalyzer.cc")
    if os.path.exists(cc_path):
        with open(cc_path, "r", encoding="utf-8") as src_file:
            src = src_file.read()
        m = re.search(r"GRID_FEAT_NAMES\[GF_NFEAT\]\s*=\s*\{(.*?)\};", src, re.S)
        cc_names = re.findall(r'"([^"]+)"', m.group(1)) if m else []
        check(
            "feature names/order identical to GRID_FEAT_NAMES in %s (%d)"
            % (os.path.basename(cc_path), len(cc_names)),
            cc_names == FEATURES,
        )
        m = re.search(r"enum GridCandType \{(.*?)\};", src, re.S)
        cc_types = re.findall(r"GC_[A-Z_]+", m.group(1)) if m else []
        want = [
            "GC_CH_HADRON", "GC_NEUTRAL_HADRON", "GC_PHOTON", "GC_PF_ELECTRON",
            "GC_PF_MUON", "GC_ELECTRON", "GC_MUON"
        ]
        check("particle-type order identical to GridCandType in the analyzer", cc_types == want)
    else:
        print("  (skipped: %s not found next to this script)" % cc_path)

    check(
        "every per-type feature exists, no duplicates",
        all(all(nm in IDX for nm in fs) and len(set(fs)) == len(fs) for fs in TYPE_FEATURES.values()),
    )
    check(
        "every feature is used by at least one type",
        set(f for fs in TYPE_FEATURES.values() for f in fs) == set(FEATURES),
    )
    check(
        "blocks cover every type exactly once",
        sorted(t for ts in BLOCKS.values() for t in ts) == sorted(TYPE_NAMES),
    )

    # Grid geometry.
    print("geometry (inner 11x11 @0.02, outer 21x21 @0.05)")

    def where(de, dp, grid):
        f = np.zeros((1, N_FEAT))
        f[0, I_PT] = 1.0
        f[0, I_DETA] = de
        f[0, I_DPHI] = dp
        ie, ip, ty, x, n = grid_slots(np.array([2]), f, grid)
        return None if n == 0 else (int(ie[0]), int(ip[0]))

    check(
        "axis (0,0) is in cell (0,0) of both grids",
        where(0, 0, "inner") == (0, 0) and where(0, 0, "outer") == (0, 0),
    )
    check(
        "inner cell 0 spans +-0.01",
        where(0.0099, 0, "inner") == (0, 0)
        and where(0.0101, 0, "inner") == (1, 0)
        and where(-0.0099, 0, "inner") == (0, 0)
        and where(-0.0101, 0, "inner") == (-1, 0),
    )
    check(
        "inner grid reaches +-0.11 (cell +-5) and nothing beyond",
        where(0.109, -0.109, "inner") == (5, -5)
        and where(0.111, 0, "inner") is None
        and where(0, -0.111, "inner") is None,
    )
    check(
        "outer cell 0 spans +-0.025",
        where(0.0249, 0, "outer") == (0, 0)
        and where(0.0251, 0, "outer") == (1, 0),
    )
    check(
        "outer grid reaches +-0.525 (cell +-10) and nothing beyond",
        where(0.524, -0.524, "outer") == (10, -10)
        and where(0.526, 0, "outer") is None
        and where(0, -0.526, "outer") is None,
    )
    check(
        "candidate at (0.15, 0) is only in outer grid",
        where(0.15, 0, "inner") is None and where(0.15, 0, "outer") == (3, 0),
    )

    # Random events against the independent reference implementation.
    print("random events vs. naive reference implementation")
    keys = ["gc_type"] + ["gc_" + f for f in FEATURES]
    events = []
    for _ in range(300):
        n = rng.randint(0, 80)
        ev = {"gc_" + f: rng.randn(n) for f in FEATURES}
        ev["gc_type"] = rng.randint(0, N_TYPES, n)
        ev["gc_pt"] = np.round(rng.exponential(5.0, n), 1) + 0.1
        for k in ("gc_deta", "gc_dphi"):
            ev[k] = (
                rng.uniform(-0.6, 0.6, n) * (rng.rand(n) < 0.5)
                + rng.uniform(-0.1, 0.1, n) * (rng.rand(n) >= 0.5)
            )
        for j in range(min(n, 6)):
            edge = rng.choice([0.0, 0.01, 0.025, 0.03, 0.075, 0.11, 0.525])
            ev["gc_deta"][j] = rng.choice([-1, 1]) * edge + rng.choice([-1, 1]) * 1e-4
        for k in keys[1:]:
            ev[k] = np.asarray(ev[k]).astype(np.float32)
        events.append(ev)

    events.append({k: np.zeros(0) for k in keys})
    sp, stats = build_sparse_for_selftest(iter(events))
    sp["grid_centre_pt"] = rng.uniform(20, 200, len(events)).astype(np.float32)
    rows_all = np.arange(len(events))
    dense = {g: densify_full(sp, g, rows_all) for g in GRIDS}
    bad = {g: 0 for g in GRIDS}

    for r, ev in enumerate(events):
        ctype, feats = event_arrays(ev)
        for g in GRIDS:
            if not np.allclose(dense[g][r], _reference_dense(ctype, feats, g), atol=1e-6):
                bad[g] += 1

    check("inner grid: all %d events identical to reference" % len(events), bad["inner"] == 0)
    check("outer grid: all %d events identical to reference" % len(events), bad["outer"] == 0)

    # Properties.
    print("properties")
    n_in = stats["n_in_window"]
    check("every inner-window candidate is also in outer window", n_in["inner"] <= n_in["outer"])
    check(
        "at most one particle per (event, cell, type)",
        all(
            len(np.unique(
                ((sp[g + "_ev"].astype(np.int64) * 50 + sp[g + "_ieta"].astype(np.int64)) * 50
                 + sp[g + "_iphi"].astype(np.int64)) * 10 + sp[g + "_type"].astype(np.int64)
            )) == len(sp[g + "_ev"])
            for g in GRIDS
        ),
    )

    f = np.zeros((3, N_FEAT))
    f[:, I_PT] = [1.0, 7.0, 3.0]
    f[:, I_DETA] = [0.001, 0.002, -0.003]
    f[:, I_DPHI] = 0.0
    ie, ip, ty, x, n = grid_slots(np.array([0, 0, 0]), f, "inner")
    check(
        "3 charged hadrons in one cell -> only the 7 GeV one is kept",
        len(ie) == 1 and n == 3 and abs(x[0, I_PT] - 7.0) < 1e-6,
    )

    ie, ip, ty, x, n = grid_slots(np.array([0, 2, 0]), f, "inner")
    check("photon sharing cell with charged hadron is also kept", len(ie) == 2)

    f3 = np.zeros((2, N_FEAT))
    f3[:, I_PT] = 5.0
    f3[:, I_DETA] = [0.001, 0.002]
    f3[:, IDX["charge"]] = [1, -1]
    ie, ip, ty, x, n = grid_slots(np.array([0, 0]), f3, "inner")
    check("equal pT -> first candidate in list is kept", len(ie) == 1 and x[0, IDX["charge"]] == 1)

    # Dense block tensor layouts.
    print("dense per-block tensors")
    rows = np.array([5, 0, 123, 299, 300])

    for g, (nc, _) in GRIDS.items():
        d = densify(sp, g, rows, rel_pt=False)
        d_rel = densify(sp, g, rows, rel_pt=True)
        shapes_ok = all(
            d[b].shape == (len(rows), nc, nc, len(block_channels(b)))
            for b in BLOCKS
        )
        check(
            "%s: block shapes (N,%d,%d,channels) = %s"
            % (g, nc, nc, {b: d[b].shape[-1] for b in BLOCKS}),
            shapes_ok,
        )

        full = dense[g][rows]
        ok = True
        for b in BLOCKS:
            for k, name in enumerate(block_channels(b)):
                t, fname = name.split("/")
                ti = TYPE_NAMES.index(t)
                want = (
                    (full[..., ti, I_PT] != 0).astype(np.float32)
                    if fname == "valid"
                    else full[..., ti, IDX[fname]]
                )
                if not np.array_equal(d[b][..., k], want):
                    ok = False
        check("%s: every channel equals raw cell/type/feature value" % g, ok)

        pt_ch = block_channels("hadrons").index("ch_hadron/pt")
        cpt = sp["grid_centre_pt"][rows]
        check(
            "%s: rel_pt divides pt by centre pT" % g,
            np.allclose(
                d_rel["hadrons"][..., pt_ch],
                d["hadrons"][..., pt_ch] / cpt[:, None, None],
                rtol=1e-5,
            ),
        )

    check(
        "empty event gives an all-zero tensor",
        all(not densify(sp, g, [len(events) - 1])[b].any() for g in GRIDS for b in BLOCKS),
    )

    # NPZ round trip.
    print("file round trip")
    p = os.path.join(_tempfile.mkdtemp(), "t.npz")
    np.savez_compressed(p, **sp)
    sp2 = np.load(p)
    same = all(
        np.array_equal(densify(sp2, g, rows)[b], densify(sp, g, rows)[b])
        for g in GRIDS for b in BLOCKS
    )
    check("saved/loaded sparse file densifies identically", same)
    print_summary(stats)
    print("\nSELFTEST", "PASSED" if fails == 0 else "FAILED (%d)" % fails)
    return fails


def build_sparse_for_selftest(events):
    """Small in-memory builder used by self-test only (not for large ROOT datasets)."""
    acc = {g: {"ev": [], "ie": [], "ip": [], "ty": [], "x": []} for g in GRIDS}
    n_events = 0
    n_in_window = {g: 0 for g in GRIDS}
    n_cand = 0

    for row, ev in enumerate(events):
        ctype, feats = event_arrays(ev)
        n_cand += len(ctype)

        for g in GRIDS:
            ie, ip, ty, x, n_in = grid_slots(ctype, feats, g)
            n_in_window[g] += n_in
            a = acc[g]
            a["ev"].append(np.full(len(ie), row, dtype=np.int32))
            a["ie"].append(ie.astype(np.int8))
            a["ip"].append(ip.astype(np.int8))
            a["ty"].append(ty.astype(np.int8))
            a["x"].append(x)

        n_events += 1

    out = {"n_events": np.array(n_events)}
    stats = {"n_events": n_events, "n_cand": n_cand, "n_in_window": n_in_window}

    for g, a in acc.items():
        cat = lambda k, dt: np.concatenate(a[k]) if a[k] else np.zeros((0,), dt)
        out[g + "_ev"] = cat("ev", np.int32)
        out[g + "_ieta"] = cat("ie", np.int8)
        out[g + "_iphi"] = cat("ip", np.int8)
        out[g + "_type"] = cat("ty", np.int8)
        out[g + "_x"] = (
            np.concatenate(a["x"])
            if a["x"]
            else np.zeros((0, N_FEAT), np.float32)
        )
        stats[g + "_slots"] = len(out[g + "_ev"])

        nc = GRIDS[g][0]
        half = nc // 2
        cell_key = (
            (out[g + "_ev"].astype(np.int64) * nc
             + (out[g + "_ieta"].astype(np.int64) + half)) * nc
            + (out[g + "_iphi"].astype(np.int64) + half)
        )
        stats[g + "_cells"] = len(np.unique(cell_key))

    return out, stats


def build_sparse(events):
    """Build sparse grids in memory from an iterable of event dictionaries.

    This function is kept for compatibility with small-sample workflows such as
    smoke_test_mc.py, which fabricates a few hundred events and calls
    ``build_grids.build_sparse(...)`` directly.

    IMPORTANT: this function accumulates its result in RAM. Use ``run_root``
    (the two-pass disk-backed ROOT workflow) for large production datasets.
    """
    return build_sparse_for_selftest(events)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", help="ROOT file written by the modified WZAnalyzer")
    ap.add_argument("--out", default="grids.npz", help="output sparse-grid .npz file")
    ap.add_argument("--tree", default="Events", help="ROOT tree path (default: Events)")
    ap.add_argument(
        "--step-size",
        default="50 MB",
        help='ROOT reading chunk size (default: "50 MB"; try "20 MB" if RAM is limited)',
    )
    ap.add_argument("--selftest", action="store_true", help="run internal tests; no ROOT file needed")
    ap.add_argument("--report", action="store_true", help="print per-type feature counts and exit")
    a = ap.parse_args()

    if a.report:
        print_report()
        return

    if a.selftest:
        sys.exit(1 if selftest() else 0)

    if not a.root:
        ap.error("give --root FILE.root (or --selftest / --report)")

    run_root(a.root, a.out, a.tree, a.step_size)


if __name__ == "__main__":
    main()
