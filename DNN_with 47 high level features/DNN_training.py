#!/usr/bin/env python3
"""
train_deeptau_mc.py -- MULTICLASS DeepTau-style classifier (CMS DeepTau paper, arXiv:2201.08458, Fig. 3).
Four classes, one per sample (label in brackets):
    jet (0) = QCD sample     tau (1) = tau sample     muon (2) = mumu sample     electron (3) = ee sample
The tau candidate is the one the analyzer selected (tauHLIndex); every sample uses its own tau candidates.

Inputs (made by export_hl_mc.py and build_grids.py, one file per sample, same ROOT file for both):
  --hl-qcd / --hl-tau / --hl-mumu / --hl-ee        mc_hl_*.npz  (hl, names, selected)   high-level variables
  --grids-qcd / ... / --grids-ee                   grids_*.npz  (only for --mode full)
Rows used: selected == 1 in the hl file (and grid_centre_is_tau == 1 in the grids file, --mode full).

Modes
  --mode hl    high-level variables only (a fast baseline; no grids needed)
  --mode full  high-level + inner/outer grids (the full network; needs build_grids.py in this folder)

In --mode full, grid features that only the CMSSW_10_6_30 build fills are zeroed in all samples (--mask-grid-features).
The number of high-level inputs is read from the .npz (41 with export_hl_mc.py, 47 with --jets).

Network (sizes from the paper's Fig. 3, as in train_deeptau.py)
  high-level N -> 113 -> 80 -> 57
  per grid: 1x1 cell layers (e/gamma 207-129-104, muon 154-96-77, hadrons 92-57-46, merged 227-141-88-64),
            inner grid 5 conv layers 3x3 x64, outer grid 10 conv layers 3x3 x64
  concatenate 57 + 64 + 64 -> 4 x 200 -> 4 outputs (softmax)
  every layer except the last: linear/conv -> BatchNorm -> PReLU -> Dropout(0.2)

Usage
  python train_deeptau_mc.py --mode hl --hl-qcd mc_hl_qcd.npz --hl-tau mc_hl_tau.npz \
         --hl-mumu mc_hl_mumu.npz --hl-ee mc_hl_ee.npz --epochs 20 --out mc_deeptau_hl
  add --dry-run to only load the data, print the class counts and fit the normalisation (no torch needed).
Outputs: <out>.pt, <out>_norm.npz, <out>_test_scores.npz (prob, label, class_names), <out>_last.pt
Needs: numpy, torch  (build_grids.py in the same folder for --mode full).
"""
import argparse
import os
import time

import numpy as np

try:
    from build_grids import BLOCKS, FEATURES, GRIDS, block_channels, densify
except ImportError:          # only needed for --mode full
    BLOCKS = FEATURES = GRIDS = block_channels = densify = None

try:
    import torch
    import torch.nn as nn
except ImportError:          # the data pipeline below works without torch (--dry-run)
    torch = None

    class _NoNN(object):
        Module = object
    nn = _NoNN()

CLASS_NAMES = ["jet", "tau", "muon", "electron"]      # label = position; tau = 1 as in the binary script
SAMPLE_KEYS = ["qcd", "tau", "mumu", "ee"]            # the sample that provides each class
N_CLASSES = len(CLASS_NAMES)
SENTINEL = -999.0                                     # the analyzer's "not available" value
CLIP = 10.0                                           # standardised inputs are clipped to +-CLIP

BLOCK_WIDTHS = {"egamma": [207, 129, 104], "muon": [154, 96, 77], "hadrons": [92, 57, 46]}
MERGE_WIDTHS = [227, 141, 88, 64]
INNER_CONV_LAYERS, OUTER_CONV_LAYERS = 5, 10
DROPOUT = 0.2

# Grid inputs that the CMSSW_7_6_7 analyzer leaves at 0 (GRID_HAVE_* switches off: PV_ASSOC_QUALITY, PUPPI,
# LOST_INNER_HITS, PIXEL_HITS, HCAL_FRACTION) but the CMSSW_10_6_30 analyzer fills. tau and qcd come from the
# first build and mumu and ee from the second, so these channels would tell the samples apart. They are set
# to 0 in ALL samples by default (--mask-grid-features '' to keep them, e.g. after re-running on one build).
DEFAULT_MASK = ["pvAssociationQuality", "puppiWeight", "puppiWeightNoLep", "lostInnerHits",
                "numberOfPixelHits", "hcalFraction"]


def W(width, scale):
    """Layer width times --scale (1.0 = the paper's sizes; smaller only for quick tests)."""
    return max(4, int(round(width * scale)))


# =========================================================================================
# data (numpy only)
# =========================================================================================
def load_sample(hl_path, grids_path, max_events, need_grids):
    """-> (sparse grids dict or None, rows used, high-level matrix of those rows, variable names)"""
    z = np.load(hl_path)
    names = [str(n) for n in z["names"]]
    sel = z["selected"] == 1
    sp = None
    if need_grids:
        sp = dict(np.load(grids_path))
        if len(sp["grid_centre_is_tau"]) != len(sel):
            raise SystemExit("%s has %d events but %s has %d: they were not made from the same ROOT file"
                             % (hl_path, len(sel), grids_path, len(sp["grid_centre_is_tau"])))
        sel = sel & (sp["grid_centre_is_tau"] == 1)
    rows = np.nonzero(sel)[0]
    if max_events:
        rows = rows[:max_events]
    hl = z["hl"][rows].astype(np.float32)
    print("  %-22s %7d selected rows of %d   (%d variables)" % (hl_path, len(rows), len(sel), len(names)))
    return sp, rows, hl, names


def describe(names, hls):
    """pT / eta of the tau candidates per class: large differences let the network cheat on kinematics."""
    ipt, ieta = names.index("tau_pt"), names.index("tau_eta")
    print("tau-candidate kinematics per class (pT quartiles in GeV, mean |eta|):")
    for c, hl in enumerate(hls):
        q = np.percentile(hl[:, ipt], [25, 50, 75])
        print("  %-9s n=%7d   pT %6.1f %6.1f %6.1f   mean|eta| %.2f"
              % (CLASS_NAMES[c], len(hl), q[0], q[1], q[2], np.abs(hl[:, ieta]).mean()))


def fit_hl_norm(hl):
    bad = ~np.isfinite(hl) | (np.abs(hl - SENTINEL) < 1e-3)
    h = np.where(bad, np.nan, hl.astype(np.float64))
    mean = np.nanmean(h, axis=0)
    std = np.nanstd(h, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_hl_norm(hl, norm):
    bad = ~np.isfinite(hl) | (np.abs(hl - SENTINEL) < 1e-3)
    z = (np.where(bad, 0.0, hl) - norm["mean"]) / norm["std"]
    z[bad] = 0.0                                  # missing -> the mean
    return np.clip(z, -CLIP, CLIP).astype(np.float32)


def check_and_mask_grid_features(sps, to_mask):
    """Print which grid features are non-zero in which sample, then zero `to_mask` in every sample."""
    nz = []                                    # per sample: bool per feature, "any particle has a non-zero value"
    for sp in sps:
        nz.append((np.asarray(sp["outer_x"]) != 0).any(axis=0))
    nz = np.array(nz)                          # (n_samples, n_features)
    print("grid features that are non-zero for at least one particle (1 = filled, 0 = all zero):")
    print("  %-26s" % "" + "".join("%8s" % k for k in SAMPLE_KEYS))
    for j, f in enumerate(FEATURES):
        if f in to_mask or (0 < nz[:, j].sum() < len(sps)):
            print("  %-26s" % (f + ("  [masked]" if f in to_mask else "")) + "".join("%8d" % int(v) for v in nz[:, j]))
    other = [f for j, f in enumerate(FEATURES) if f not in to_mask and 0 < nz[:, j].sum() < len(sps)]
    if other:
        print("WARNING: these grid features are filled in some samples but all zero in others (not masked): %s"
              % ", ".join(other))
    for f in to_mask:
        if f not in FEATURES:
            raise SystemExit("--mask-grid-features: unknown grid feature '%s'" % f)
    idx = [FEATURES.index(f) for f in to_mask]
    if idx:
        for sp in sps:
            for g in ("inner", "outer"):
                sp[g + "_x"][:, idx] = 0
        print("set to 0 in every sample: " + ", ".join(to_mask))


class ChannelNorm:
    """Per-channel standardisation of one block of one grid, over OCCUPIED cells only.
    Empty cells stay exactly 0; the 'valid' channels are left as 0/1."""

    def __init__(self, names):
        self.names = names
        types = [n.split("/")[0] for n in names]
        self.is_valid = np.array([n.endswith("/valid") for n in names])
        self.vidx = np.array([names.index(t + "/valid") for t in types])
        self.mean = np.zeros(len(names), np.float32)
        self.std = np.ones(len(names), np.float32)

    def begin(self):
        k = len(self.names)
        self._s1, self._s2, self._n = np.zeros(k), np.zeros(k), np.zeros(k)

    def update(self, x):                           # x: (B, n, n, C); running sums, nothing is kept
        occ = x[..., self.vidx]
        x64 = x.astype(np.float64)
        self._s1 += (x64 * occ).sum(axis=(0, 1, 2))
        self._s2 += (x64 * x64 * occ).sum(axis=(0, 1, 2))
        self._n += occ.sum(axis=(0, 1, 2))

    def finish(self):
        s1, s2, n = self._s1, self._s2, self._n
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s1 / n, 0.0)
            var = np.where(n > 0, s2 / n - mean ** 2, 1.0)
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(std > 1e-8, std, 1.0)
        mean = np.where(self.is_valid, 0.0, mean)
        std = np.where(self.is_valid, 1.0, std)
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    def __call__(self, x):
        occ = x[..., self.vidx]
        z = np.clip((x - self.mean) / self.std, -CLIP, CLIP) * occ
        return np.where(self.is_valid, x, z).astype(np.float32)


class GridData:
    """All selected events of the four samples; builds normalised dense batches on the fly
    (the full dense outer grid would be tens of GB)."""

    def __init__(self, samples):
        # samples: list of (sp, rows, hl_raw, label)
        self.sp = [s[0] for s in samples]
        self.src = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(samples)])
        self.row = np.concatenate([s[1] for s in samples]).astype(np.int64)
        self.y = np.concatenate([np.full(len(s[1]), s[3], np.int64) for s in samples])
        self.hl_raw = np.concatenate([s[2] for s in samples]).astype(np.float32)
        self.hl_norm = None
        self.chan_norm = None           # {grid: {block: ChannelNorm}}

    def __len__(self):
        return len(self.y)

    def dense(self, gidx, normalise=True):
        """(gidx reordered by source, {grid: {block: (B, n, n, C)}})"""
        gidx = np.asarray(gidx, dtype=np.int64)
        order, parts = [], {g: {b: [] for b in BLOCKS} for g in GRIDS}
        for s in np.unique(self.src[gidx]):
            m = self.src[gidx] == s
            order.append(gidx[m])
            for g in GRIDS:
                d = densify(self.sp[s], g, self.row[gidx[m]])
                for b in BLOCKS:
                    x = d[b]
                    if normalise and self.chan_norm is not None:
                        x = self.chan_norm[g][b](x)
                    parts[g][b].append(x)
        order = np.concatenate(order)
        return order, {g: {b: np.concatenate(parts[g][b], axis=0) for b in BLOCKS} for g in GRIDS}

    def batch(self, gidx):
        order, grids = self.dense(gidx)
        hl = apply_hl_norm(self.hl_raw[order], self.hl_norm)
        return hl, grids, self.y[order]


def fit_channel_norms(data, idx, chunk=200):
    norms = {g: {b: ChannelNorm(block_channels(b)) for b in BLOCKS} for g in GRIDS}
    for g in GRIDS:
        for b in BLOCKS:
            norms[g][b].begin()
    for i in range(0, len(idx), chunk):            # streaming: only one chunk is in memory at a time
        _, grids = data.dense(idx[i:i + chunk], normalise=False)
        for g in GRIDS:
            for b in BLOCKS:
                norms[g][b].update(grids[g][b])
        del grids
    for g in GRIDS:
        for b in BLOCKS:
            norms[g][b].finish()
    return norms


def split_indices(y, seed=1, frac=(0.7, 0.15, 0.15)):
    """stratified train / val / test split of the global event indices (every class split the same way)"""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        i = rng.permutation(np.nonzero(y == c)[0])
        a, b = int(frac[0] * len(i)), int((frac[0] + frac[1]) * len(i))
        tr.append(i[:a]); va.append(i[a:b]); te.append(i[b:])
    return [np.sort(np.concatenate(p)) for p in (tr, va, te)]


class ArrayLoader:
    """--mode hl: batches of the already normalised high-level matrix."""

    def __init__(self, hl, y, batch, shuffle, seed=0):
        self.hl, self.y, self.batch, self.shuffle = hl, y, batch, shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        n = len(self.y)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        stop = n - n % self.batch if self.shuffle else n      # no size-1 batch in training (BatchNorm)
        for i in range(0, stop, self.batch):
            b = idx[i:i + self.batch]
            yield self.hl[b], None, self.y[b]


# =========================================================================================
# metrics (numpy only)
# =========================================================================================
def roc(score, y):
    """(fpr, tpr, auc) with y == 1 the signal"""
    o = np.argsort(-score, kind="stable")
    ys = y[o]
    tpr = np.concatenate([[0], np.cumsum(ys == 1) / max((ys == 1).sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(ys == 0) / max((ys == 0).sum(), 1)])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def eff_at_mistag(fpr, tpr, mistag):
    return float(np.interp(mistag, fpr, tpr))


def one_vs_rest_auc(prob, y):
    return [roc(prob[:, c], (y == c).astype(np.int64))[2] for c in range(N_CLASSES)]


def tau_vs(prob, y, other):
    """tau-vs-<other class> discriminator D = p_tau / (p_tau + p_other), on the tau and <other> events only"""
    m = (y == 1) | (y == other)
    d = prob[m, 1] / np.maximum(prob[m, 1] + prob[m, other], 1e-12)
    return roc(d, (y[m] == 1).astype(np.int64))


def report(prob, y):
    pred = prob.argmax(axis=1)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    print("accuracy %.4f" % (np.trace(cm) / max(cm.sum(), 1)))
    print("confusion matrix (rows = true class, columns = predicted):")
    print("  %-10s" % "" + "".join("%10s" % n for n in CLASS_NAMES))
    for c in range(N_CLASSES):
        print("  %-10s" % CLASS_NAMES[c] + "".join("%10d" % v for v in cm[c]))
    aucs = one_vs_rest_auc(prob, y)
    print("one-vs-rest AUC:  " + "   ".join("%s %.4f" % (n, a) for n, a in zip(CLASS_NAMES, aucs)))
    for other in (0, 2, 3):
        fpr, tpr, auc = tau_vs(prob, y, other)
        print("tau vs %-8s AUC %.4f   tau eff at mistag 1e-1: %.3f   1e-2: %.3f   1e-3: %.3f"
              % (CLASS_NAMES[other], auc, eff_at_mistag(fpr, tpr, 0.1),
                 eff_at_mistag(fpr, tpr, 0.01), eff_at_mistag(fpr, tpr, 0.001)))
    return aucs


# =========================================================================================
# network (torch)
# =========================================================================================
def layer1d(cin, cout, p=DROPOUT):
    return nn.Sequential(nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.PReLU(cout), nn.Dropout(p))


def layer2d(cin, cout, k, p=DROPOUT):
    return nn.Sequential(nn.Conv2d(cin, cout, k), nn.BatchNorm2d(cout), nn.PReLU(cout), nn.Dropout(p))


class HLNet(nn.Module):
    """--mode hl: the high-level branch of the paper's network followed by the 4 x 200 head."""

    def __init__(self, n_hl, n_out=N_CLASSES, scale=1.0):
        super().__init__()
        a, b, c, h = W(113, scale), W(80, scale), W(57, scale), W(200, scale)
        self.hl = nn.Sequential(layer1d(n_hl, a), layer1d(a, b), layer1d(b, c))
        self.head = nn.Sequential(layer1d(c, h), layer1d(h, h), layer1d(h, h), layer1d(h, h), nn.Linear(h, n_out))

    def forward(self, hl, inner=None, outer=None):
        return self.head(self.hl(hl))


class CellReducer(nn.Module):
    """1x1 layers applied cell by cell: three blocks, concatenated, then a common stack -> 64."""

    def __init__(self, in_channels, scale=1.0):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for b, widths in BLOCK_WIDTHS.items():
            layers, c = [], in_channels[b]
            for w in widths:
                layers.append(layer2d(c, W(w, scale), 1)); c = W(w, scale)
            self.blocks[b] = nn.Sequential(*layers)
        merge, c = [], sum(W(w[-1], scale) for w in BLOCK_WIDTHS.values())          # 104 + 77 + 46 = 227
        for w in MERGE_WIDTHS:
            merge.append(layer2d(c, W(w, scale), 1)); c = W(w, scale)
        self.merge = nn.Sequential(*merge)

    def forward(self, x):                           # x[block]: (B, C_block, n, n)
        return self.merge(torch.cat([self.blocks[b](x[b]) for b in BLOCK_WIDTHS], dim=1))


class DeepTauMC(nn.Module):
    def __init__(self, in_channels_inner, in_channels_outer, n_hl, n_out=N_CLASSES, scale=1.0):
        super().__init__()
        a, b, c, h, f = W(113, scale), W(80, scale), W(57, scale), W(200, scale), W(64, scale)
        self.hl = nn.Sequential(layer1d(n_hl, a), layer1d(a, b), layer1d(b, c))
        self.inner_cells = CellReducer(in_channels_inner, scale)
        self.outer_cells = CellReducer(in_channels_outer, scale)
        self.inner_conv = nn.Sequential(*[layer2d(f, f, 3) for _ in range(INNER_CONV_LAYERS)])   # 11 -> 1
        self.outer_conv = nn.Sequential(*[layer2d(f, f, 3) for _ in range(OUTER_CONV_LAYERS)])   # 21 -> 1
        self.head = nn.Sequential(layer1d(c + 2 * f, h), layer1d(h, h), layer1d(h, h), layer1d(h, h),
                                  nn.Linear(h, n_out))

    def forward(self, hl, inner, outer):
        a = self.hl(hl)
        b = self.inner_conv(self.inner_cells(inner)).flatten(1)
        c = self.outer_conv(self.outer_cells(outer)).flatten(1)
        return self.head(torch.cat([a, b, c], dim=1))


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_torch(hl, grids, y, device):
    hl_t = torch.from_numpy(np.ascontiguousarray(hl)).to(device)
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(device)
    if grids is None:
        return hl_t, None, None, y_t

    def conv(g):   # (B, n, n, C) -> (B, C, n, n)
        return {b: torch.from_numpy(np.ascontiguousarray(grids[g][b].transpose(0, 3, 1, 2))).to(device)
                for b in BLOCKS}
    return hl_t, conv("inner"), conv("outer"), y_t


def _identity(x):
    return x


class BatchDS(torch.utils.data.Dataset if torch is not None else object):
    """Module level (not a local class), so it can also be pickled by 'spawn'/'forkserver'."""

    def __init__(self, data, idx):
        self.data, self.idx = data, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, positions):                # a list of positions (batch sampler)
        return self.data.batch(self.idx[np.asarray(positions)])


def make_loader(data, idx, batch_size, shuffle, workers):
    import multiprocessing
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    sampler = RandomSampler(range(len(idx))) if shuffle else SequentialSampler(range(len(idx)))
    kw = {}
    if workers > 0 and "fork" not in multiprocessing.get_all_start_methods():
        print("note: this system cannot 'fork' (Windows): using --workers 0")
        workers = 0
    if workers > 0:
        kw["multiprocessing_context"] = "fork"        # shares the dataset instead of pickling it into every worker
    return DataLoader(BatchDS(data, idx), sampler=BatchSampler(sampler, batch_size, drop_last=shuffle),
                      batch_size=None, num_workers=workers, collate_fn=_identity, **kw)


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for hl, grids, y in loader:
            hl_t, inner, outer, _ = to_torch(hl, grids, y, device)
            probs.append(torch.softmax(model(hl_t, inner, outer), dim=1).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["hl", "full"], default="hl")
    for k in SAMPLE_KEYS:
        ap.add_argument("--hl-" + k, required=True, help="mc_hl_%s.npz from export_hl_mc.py" % k)
        ap.add_argument("--grids-" + k, help="grids file of the %s sample from build_grids.py (--mode full)" % k)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2, help="data-loading processes (--mode full)")
    ap.add_argument("--scale", type=float, default=1.0, help="layer widths x this (1.0 = paper sizes; smaller for tests)")
    ap.add_argument("--mask-grid-features", default=",".join(DEFAULT_MASK),
                    help="comma-separated grid features set to 0 in ALL samples (--mode full); '' keeps everything")
    ap.add_argument("--max-events", type=int, default=0, help="use at most this many rows per sample (smoke test)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="mc_deeptau")
    ap.add_argument("--resume", action="store_true", help="continue from <out>_last.pt if it exists")
    ap.add_argument("--dry-run", action="store_true", help="load the data and fit the normalisation, then stop (no torch needed)")
    a = ap.parse_args()
    full = a.mode == "full"
    if not a.dry_run and torch is None:
        raise SystemExit("torch is not installed (pip install torch)")
    if full:
        if densify is None:
            raise SystemExit("--mode full needs build_grids.py in the same folder")
        missing = [k for k in SAMPLE_KEYS if not getattr(a, "grids_" + k)]
        if missing:
            raise SystemExit("--mode full needs --grids-<sample> for: " + ", ".join(missing))

    print("loading ...")
    loaded = [load_sample(getattr(a, "hl_" + k), getattr(a, "grids_" + k), a.max_events or None, full)
              for k in SAMPLE_KEYS]
    names = loaded[0][3]
    for k, s in zip(SAMPLE_KEYS, loaded):
        if s[3] != names:
            raise SystemExit("the %s file has different variable names than the %s file" % (k, SAMPLE_KEYS[0]))
    n_hl = len(names)
    describe(names, [s[2] for s in loaded])
    np.random.seed(a.seed)

    if full:
        check_and_mask_grid_features([s[0] for s in loaded], [f for f in a.mask_grid_features.split(",") if f])
        data = GridData([(s[0], s[1], s[2], c) for c, s in enumerate(loaded)])
        y = data.y
    else:
        hl_all = np.concatenate([s[2] for s in loaded])
        y = np.concatenate([np.full(len(s[2]), c, np.int64) for c, s in enumerate(loaded)])
    tr, va, te = split_indices(y)
    counts = np.bincount(y[tr], minlength=N_CLASSES)
    print("events: train %d  val %d  test %d" % (len(tr), len(va), len(te)))
    print("train events per class: " + "  ".join("%s %d" % (n, c) for n, c in zip(CLASS_NAMES, counts)))

    # normalisation from the TRAINING events only
    hl_norm = fit_hl_norm((data.hl_raw if full else hl_all)[tr])
    norm_out = {"hl_mean": hl_norm["mean"], "hl_std": hl_norm["std"], "names": np.array(names)}
    if full:
        data.hl_norm = hl_norm
        rng = np.random.RandomState(2)
        sub = np.sort(np.concatenate([rng.choice(tr[y[tr] == c], min(1500, int((y[tr] == c).sum())), replace=False)
                                      for c in range(N_CLASSES)]))
        print("fitting per-channel normalisation on %d training events ..." % len(sub))
        data.chan_norm = fit_channel_norms(data, sub)
        for g in GRIDS:
            for b in BLOCKS:
                for k in ("mean", "std"):
                    norm_out["%s_%s_%s" % (g, b, k)] = getattr(data.chan_norm[g][b], k)
    np.savez(a.out + "_norm.npz", **norm_out)
    weights = len(tr) / (float(N_CLASSES) * np.maximum(counts, 1))
    print("class weights in the loss: " + "  ".join("%s %.2f" % (n, w) for n, w in zip(CLASS_NAMES, weights)))
    if a.dry_run:
        print("dry run finished: data and normalisation are fine, nothing was trained")
        return

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if full:
        in_ch = {g: {b: len(block_channels(b)) for b in BLOCKS} for g in GRIDS}
        model = DeepTauMC(in_ch["inner"], in_ch["outer"], n_hl, N_CLASSES, a.scale).to(device)
        train_loader = make_loader(data, tr, a.batch, True, a.workers)
        val_loader = make_loader(data, va, a.batch, False, a.workers)
        test_loader = make_loader(data, te, a.batch, False, a.workers)
    else:
        hlz = apply_hl_norm(hl_all, hl_norm)
        model = HLNet(n_hl, N_CLASSES, a.scale).to(device)
        train_loader = ArrayLoader(hlz[tr], y[tr], a.batch, True, a.seed)
        val_loader = ArrayLoader(hlz[va], y[va], a.batch, False)
        test_loader = ArrayLoader(hlz[te], y[te], a.batch, False)
    print("mode: %s   device: %s   high-level inputs: %d   trainable parameters: %d"
          % (a.mode, device, n_hl, n_params(model)))

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

    best, start = -1.0, 1
    last_path = a.out + "_last.pt"
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, start = ck["best"], ck["epoch"] + 1
        print("resumed from %s: %d epochs done, best val macro AUC %.4f" % (last_path, ck["epoch"], best))
    for ep in range(start, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for hl, grids, yb in train_loader:
            hl_t, inner, outer, yt = to_torch(hl, grids, yb, device)
            opt.zero_grad()
            loss = loss_fn(model(hl_t, inner, outer), yt)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        prob, yv = predict(model, val_loader, device)
        aucs = one_vs_rest_auc(prob, yv)
        score = float(np.mean(aucs))                   # macro-average one-vs-rest AUC
        sched.step(score)
        print("epoch %2d  train loss %.4f  val macro AUC %.4f  acc %.4f  (%.0fs)   %s"
              % (ep, tot / max(nb, 1), score, float((prob.argmax(1) == yv).mean()), time.time() - t0,
                 " ".join("%s %.3f" % (n[:3], v) for n, v in zip(CLASS_NAMES, aucs))))
        if score > best:
            best = score
            torch.save(model.state_dict(), a.out + ".pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "epoch": ep}, last_path)

    model.load_state_dict(torch.load(a.out + ".pt", map_location=device))
    prob, yt_ = predict(model, test_loader, device)
    print("\nTEST (best val macro AUC %.4f)" % best)
    report(prob, yt_)
    np.savez(a.out + "_test_scores.npz", prob=prob, label=yt_, class_names=np.array(CLASS_NAMES))
    print("saved %s.pt, %s_norm.npz, %s_test_scores.npz" % (a.out, a.out, a.out))


if __name__ == "__main__":
    main()