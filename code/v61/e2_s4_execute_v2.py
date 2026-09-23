#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 / S4 v2: protocol-faithful execution after independent audit (2026-09-21).

Fixes vs v1 (audit items 1-4, see outputs/corrections_and_lessons_20260921.md):
  F1 same visible set per target: BOTH arms see the same 11 covisible channels
     (only the CURRENT target masked/omitted), per frozen protocol 5.3.
  F2 one moments set per outer fold, estimated ONLY from that fold's training
     cores (pixel-pooled sufficient stats), applied to all train+test windows.
  F3 pooled variance = Σsq/Σn − mean² (never averaged SDs).
  F4 cell-level aggregation: per-cell means over nuclei-polygon support
     (rasterized onto the 128×128 model grid; approximation disclosed),
     patient value = equal-weight mean over cells; window-pixel means kept as
     secondary. D2 = appended-core sensitivity (moments/ridge from all 27
     primary cores), relabelled per audit item 12.

Outputs -> outputs/e2/s4_v2/: patient_predictions_cells.csv (+window-level
secondary), run_manifest.json. v1 kept as SUPERSEDED audit trail.
"""
import csv
import glob
import hashlib
import json
import os
import sys
import time

ROOT = r"<data-root>\virtual cell and tissue"
sys.path.insert(0, os.path.join(ROOT, "code", "Virtues"))
os.chdir(os.path.join(ROOT, "code", "Virtues"))
sys.path.insert(0, r"<project-root>\scripts")

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

from e0_streaming_adapter import load_model, EMB_DIR
from virtues.utils.utils import load_marker_embedding_dict

DATA = r"<project-root>\external_data\S-BIAD3612"
CAND = r"<project-root>\outputs\e2\gating\window_candidates.csv"
OUTD = r"<project-root>\outputs\e2\s4_v2"
PX = 0.5066976883697085
WIN_NATIVE = 253
WIN_MODEL = 128
SCALE = WIN_MODEL / WIN_NATIVE
SEED = 20260920
D2 = "Core_D2"
TARGETS = ["CD45", "CD31", "CD68", "CD163", "Ki67", "aSMA"]
VISIBLE = {"CD45": 69, "CD31": 41, "CD68": 53, "CD163": 32, "Ki67": 29, "aSMA": 6,
           "PanCK": 50, "Podoplanin": 9, "GranzymeB": 54, "CD8": 40, "CD11b": 65, "CD15": 35}
UNIPROT = {"CD45": "P08575", "CD31": "P16284", "CD68": "P34810", "CD163": "Q86VB7",
           "Ki67": "P46013", "aSMA": "P62736", "PanCK": "P08727", "Podoplanin": "Q86YL7",
           "GranzymeB": "P10144", "CD8": "P01732", "CD11b": "P11215", "CD15": "P31997"}


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def blur3(x):
    ax = torch.arange(3).float() - 1
    g = torch.exp(-(ax ** 2) / 2.0); g /= g.sum()
    w = (g[:, None] @ g[None, :]).reshape(1, 1, 3, 3).expand(x.shape[0], 1, 3, 3).contiguous()
    return F.conv2d(F.pad(x.unsqueeze(0), (1, 1, 1, 1), mode="reflect"), w, groups=x.shape[0])[0]


def load_windows():
    rows = list(csv.DictReader(open(CAND, encoding="utf-8")))
    out = {}
    for r in rows:
        if r["size_regime"] != "A":
            continue
        out.setdefault(r["core_id"], []).append(
            dict(idx=int(r["window_idx"]), r0=int(r["r0_px"]), c0=int(r["c0_px"])))
    for k in out:
        out[k].sort(key=lambda w: w["idx"])
    return out


def rasterize_cells(core, r0, c0):
    """Cell-object support on the 128-grid, EXCLUDING features classified
    'Not_cells' (audit fix 2026-09-21: classification filter; geometry = the
    segmentation objects as delivered, not nuclei-restricted). PIL 'I'-mode
    polygon fill (correct fill semantics incl. edge-crossing polygons).
    Returns int32 label image (0=background) and count of included objects."""
    from PIL import Image, ImageDraw
    path = os.path.join(DATA, "annotations", "cell_nuclei_segmentation_geojson",
                        f"{core}_cell_nuclei.geojson")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    img = Image.new("I", (WIN_MODEL, WIN_MODEL), 0)
    draw = ImageDraw.Draw(img)
    ci = 0
    for feat in gj["features"]:
        props = feat.get("properties") or {}
        cls = (props.get("classification") or {})
        name = cls.get("name") if isinstance(cls, dict) else None
        if name == "Not_cells" or (cls is None and not props.get("objectType")):
            continue  # excluded: Not_cells and unclassified objects
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        pts = [((x - c0) * SCALE, (y - r0) * SCALE) for x, y in geom["coordinates"][0]]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if max(xs) < 0 or min(xs) > WIN_MODEL or max(ys) < 0 or min(ys) > WIN_MODEL:
            continue
        if min(xs) < -20 or max(xs) > WIN_MODEL + 20 or min(ys) < -20 or max(ys) > WIN_MODEL + 20:
            continue  # far outside
        ci += 1
        draw.polygon(pts, fill=ci)
    return np.array(img, dtype=np.int32), ci


def main():
    os.makedirs(OUTD, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    md = load_marker_embedding_dict(EMB_DIR)
    markers = [m for m in VISIBLE if UNIPROT[m] in md]
    midx = torch.tensor([md[UNIPROT[m]] for m in markers], dtype=torch.long)
    tpos = {t: markers.index(t) for t in TARGETS}
    print(f"visible set: {len(markers)} markers", flush=True)

    wins = load_windows()
    cores = sorted(wins.keys())
    primary = [c for c in cores if c != D2]
    print(f"cores: {len(cores)} ({len(primary)} primary + D2 appended)", flush=True)

    model = load_model()

    # ---- pass 1: per-core windows (log1p domain) + sufficient stats + cell support
    core_win, core_stats, cell_lab = {}, {}, {}
    t0 = time.time()
    for core in cores:
        with tifffile.TiffFile(os.path.join(DATA, "images", "cores", f"{core}.ome.tif")) as tf:
            arr = np.stack([tf.pages[VISIBLE[m]].asarray().astype(np.float32) for m in markers])
        t = blur3(torch.from_numpy(arr))
        q99 = torch.from_numpy(np.percentile(t.numpy().reshape(t.shape[0], -1), 99, axis=1).astype(np.float32))
        tl = torch.log1p(torch.minimum(t, q99[:, None, None]))
        s = tl.sum(dim=(1, 2)).double().numpy()
        sq = (tl.double() ** 2).sum(dim=(1, 2)).numpy()
        n = float(tl.shape[1] * tl.shape[2])
        core_stats[core] = (s, sq, n)
        ws = []
        for w in wins[core]:
            win = tl[:, w["r0"]:w["r0"] + WIN_NATIVE, w["c0"]:w["c0"] + WIN_NATIVE].unsqueeze(0)
            win = F.interpolate(win, size=(WIN_MODEL, WIN_MODEL), mode="bilinear",
                                align_corners=False, antialias=True)[0]
            lab, ncells = rasterize_cells(core, w["r0"], w["c0"])
            ws.append(dict(win_log=win, lab=lab, n_cells=ncells))
        core_win[core] = ws
        del arr, t, tl
    print(f"pass1 done {time.time()-t0:.0f}s", flush=True)

    def fold_moments(train_cores):
        s = np.sum([core_stats[c][0] for c in train_cores], axis=0)
        sq = np.sum([core_stats[c][1] for c in train_cores], axis=0)
        n = np.sum([core_stats[c][2] for c in train_cores])
        mu = s / n
        var = np.maximum(sq / n - mu ** 2, 1e-12)
        return mu.astype(np.float32), np.sqrt(var).astype(np.float32)

    rows_cell, rows_win = [], []
    n_fwd = 0
    tF = time.time()
    # ---- VirTues arm (per test core: moments from its fold's training cores)
    for core in cores:
        train_cores = primary if core == D2 else [c for c in primary if c != core]
        mu, sd = fold_moments(train_cores)
        mu_t = torch.from_numpy(mu); sd_t = torch.from_numpy(sd)
        branch = "appended_D2" if core == D2 else "primary"
        for widx, wd in enumerate(core_win[core]):
            win_z = ((wd["win_log"] - mu_t[:, None, None]) / sd_t[:, None, None]).float()
            lab = wd["lab"]
            for tname in TARGETS:
                mask = torch.zeros(len(markers), WIN_MODEL // 8, WIN_MODEL // 8, dtype=torch.bool)
                mask[tpos[tname]] = True
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    o = model([win_z.cuda()], [midx.cuda()], [mask.cuda()])
                n_fwd += 1
                rec = o.decoded_multiplex[0].float().cpu()
                pred_log = rec[tpos[tname]] * sd_t[tpos[tname]] + mu_t[tpos[tname]]
                meas_log = wd["win_log"][tpos[tname]]
                # cell-level aggregation (primary unit)
                for cid in range(1, wd["n_cells"] + 1):
                    m = lab == cid
                    if m.sum() < 4:
                        continue
                    rows_cell.append(dict(branch=branch, core_id=core, window=widx + 1,
                                          target=tname, arm="virtues", cell=cid,
                                          meas=float(meas_log[m].mean()),
                                          pred=float(pred_log[m].mean())))
                rz = float(np.corrcoef(rec[tpos[tname]].numpy().ravel(),
                                       win_z[tpos[tname]].numpy().ravel())[0, 1])
                rows_win.append(dict(branch=branch, core_id=core, window=widx + 1, target=tname,
                                     arm="virtues", win_mean_meas=float(meas_log.mean()),
                                     win_mean_pred=float(pred_log.mean()), pixel_r_z=rz))
        print(f"virtues {core} done ({time.time()-tF:.0f}s)", flush=True)

    # ---- ridge arm: per outer fold, ONE moments set; features = same 11 covisibles per target
    def feats_z(wd, mu, sd):
        return ((wd["win_log"] - torch.from_numpy(mu)[:, None, None]) /
                torch.from_numpy(sd)[:, None, None]).float()

    for core in cores:
        train_cores = primary if core == D2 else [c for c in primary if c != core]
        mu, sd = fold_moments(train_cores)
        Xtr_pool = {t: [] for t in TARGETS}
        for c in train_cores:
            for wd in core_win[c]:
                wz = feats_z(wd, mu, sd)
                flat = wz.reshape(len(markers), -1).T
                for t in TARGETS:
                    keepcols = [i for i in range(len(markers)) if i != tpos[t]]
                    Xtr_pool[t].append(flat[:, keepcols])
        for t in TARGETS:
            keepcols = [i for i in range(len(markers)) if i != tpos[t]]
            Xtr = np.concatenate(Xtr_pool[t])
            ycols = []
            for c in train_cores:
                for wd in core_win[c]:
                    wz = feats_z(wd, mu, sd)
                    ycols.append(wz[tpos[t]].numpy().ravel())
            ytr = np.concatenate(ycols)
            w_ = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
            branch = "appended_D2" if core == D2 else "primary"
            for widx, wd in enumerate(core_win[core]):
                wz = feats_z(wd, mu, sd)
                flat = wz.reshape(len(markers), -1).T
                pred_z = (flat[:, keepcols] @ w_).reshape(WIN_MODEL, WIN_MODEL)
                pred_log = pred_z * sd[tpos[t]] + mu[tpos[t]]
                meas_log = wd["win_log"][tpos[t]]
                lab = wd["lab"]
                for cid in range(1, wd["n_cells"] + 1):
                    m = lab == cid
                    if m.sum() < 4:
                        continue
                    rows_cell.append(dict(branch=branch, core_id=core, window=widx + 1,
                                          target=tname if False else t, arm="ridge_lopo",
                                          cell=cid, meas=float(meas_log[m].mean()),
                                          pred=float(pred_log[m].mean())))
                rz = float(np.corrcoef(pred_z.ravel(), wz[tpos[t]].numpy().ravel())[0, 1])
                rows_win.append(dict(branch=branch, core_id=core, window=widx + 1, target=t,
                                     arm="ridge_lopo", win_mean_meas=float(meas_log.mean()),
                                     win_mean_pred=float(pred_log.mean()), pixel_r_z=rz))
        print(f"ridge {core} done ({time.time()-tF:.0f}s)", flush=True)

    for name, rows, fname in [("cells", rows_cell, "patient_predictions_cells.csv"),
                              ("window_secondary", rows_win, "window_predictions_secondary.csv")]:
        with open(os.path.join(OUTD, fname), "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"written {fname} ({len(rows)} rows)", flush=True)
    manifest = dict(
        mode="v2_after_audit",
        audit="E0_E2_new_results_independent_audit_20260921.md items 1-4 fixed",
        fixes=["F1 same 11-covisible set both arms per target",
               "F2 one moments set per outer fold from training cores only",
               "F3 pooled variance from sufficient stats",
               "F4 cell-level (nuclei-polygon) aggregation primary; window means secondary; D2=appended-core sensitivity"],
        seed=SEED, alpha=1.0, win_native=WIN_NATIVE, win_model=WIN_MODEL,
        visible_markers=markers, targets=TARGETS,
        n_primary=len(primary), n_forwards=n_fwd,
        candidates_sha16=sha16(CAND),
        weights_sha16=sha16(os.path.join(ROOT, "data", "weights", "virtues-sp32", "model.safetensors")),
        cell_support="nuclei polygons rasterized on 128 grid (128/253 scale) — approximation to author cell regions, disclosed",
        timing_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUTD, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    # mark v1 superseded
    open(os.path.join(r"<project-root>\outputs\e2\s4", "SUPERSEDED_BY_v2.txt"), "w", encoding="utf-8").write(
        "v1 results superseded by s4_v2 (audit 2026-09-21: arm asymmetry, ridge fold moments, SD averaging, window-mean unit). Kept as audit trail.\n")
    print("DONE s4_v2", flush=True)


if __name__ == "__main__":
    main()
