#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 / S4: protocol-transfer execution on S-BIAD3612 (laryngeal CODEX).

Pipeline per frozen protocol (gate_checklist 2026-09-21, regime A windows):
  window (128um physical, ~253 native px) -> bilinear resample to 128x128
  -> blur k=3 (reflect) -> per-core per-channel q99 clip -> log1p
  -> z with LOPO moments (other patients only; convention A: 27 cores = 27 patients,
     D2 = sensitivity branch)
  VirTues arm : patch-level full-channel mask on target -> model.forward
  Ridge arm   : closed-form ridge (alpha=1.0, Einhaus convention) fit on the
     other patients' windows, covisible channels -> target, in log1p space
  Scoring     : patient-level window-mean on log1p common scale (primary),
     z-space pixel Pearson r (spatial quality) recorded as secondary.

Modes:
  --smoke  run the pipeline on the CURRENT window_candidates.csv as-is,
           results marked SMOKE_NOT_FOR_PAPER (pipeline validation only)
  (default) require the validated window set (agent-fixed + main-window ok)

Outputs (under outputs/e2/s4[_smoke]/): patient_predictions.csv,
window_pixel_cache notes, run_manifest.json, resource_profile.csv
"""
import argparse
import csv
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
PX = 0.5066976883697085
WIN_NATIVE = 253          # 128 um / PX, rounded
WIN_MODEL = 128
SEED = 20260920
ALPHA = 1.0               # ridge, Einhaus lambda=1 convention
D2 = "Core_D2"            # sensitivity branch (archive-extra vs paper's 27)
TARGETS = ["CD45", "CD31", "CD68", "CD163", "Ki67", "aSMA"]

# published-30 markers with a local ESM embedding (unambiguous set; CD15/PanCK
# follow the Einhaus mapping convention, flagged in channel_map notes)
VISIBLE = {  # marker -> image channel index
    "CD45": 69, "CD31": 41, "CD68": 53, "CD163": 32, "Ki67": 29, "aSMA": 6,
    "PanCK": 50, "Podoplanin": 9, "GranzymeB": 54, "CD8": 40, "CD11b": 65,
    "CD15": 35,
}
UNIPROT = {
    "CD45": "P08575", "CD31": "P16284", "CD68": "P34810", "CD163": "Q86VB7",
    "Ki67": "P46013", "aSMA": "P62736", "PanCK": "P08727", "Podoplanin": "Q86YL7",
    "GranzymeB": "P10144", "CD8": "P01732", "CD11b": "P11215", "CD15": "P31997",
}


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_windows(regime="A"):
    rows = list(csv.DictReader(open(CAND, encoding="utf-8")))
    out = {}
    for r in rows:
        if r["size_regime"] != regime:
            continue
        out.setdefault(r["core_id"], []).append(
            dict(idx=int(r["window_idx"]), r0=int(r["r0_px"]), c0=int(r["c0_px"]),
                 n_nuclei=int(r["n_nuclei_in_window"])))
    for k in out:
        out[k].sort(key=lambda w: w["idx"])
    return out


def blur3(x):  # (C,H,W) float32 torch, reflect pad, gaussian k=3
    ax = torch.arange(3).float() - 1
    g = torch.exp(-(ax ** 2) / 2.0); g /= g.sum()
    w = (g[:, None] @ g[None, :]).reshape(1, 1, 3, 3).expand(x.shape[0], 1, 3, 3).contiguous()
    return F.conv2d(F.pad(x.unsqueeze(0), (1, 1, 1, 1), mode="reflect"), w, groups=x.shape[0])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = "s4_smoke" if a.smoke else "s4"
    OUT = rf"<project-root>\outputs\e2\{tag}"
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    md = load_marker_embedding_dict(EMB_DIR)
    markers = [m for m in VISIBLE if UNIPROT[m] in md]
    covis = [m for m in markers if m not in TARGETS]
    midx = torch.tensor([md[UNIPROT[m]] for m in markers], dtype=torch.long)
    tpos = {t: markers.index(t) for t in TARGETS}
    print(f"visible set: {len(markers)} markers ({len(covis)} covisible + {len(TARGETS)} targets)")

    wins = load_windows("A")
    cores = sorted(wins.keys())
    print(f"cores with windows: {len(cores)}")
    primary = [c for c in cores if c != D2]

    model = load_model()
    device = "cuda"

    # ---------- pass 1: per-core preprocessing stats (q99, log1p moments) ----------
    core_stats = {}
    t0 = time.time()
    proc = {}   # core -> (z-ready per-window log1p tensors before z), plus stats
    for core in cores:
        with tifffile.TiffFile(os.path.join(DATA, "images", "cores", f"{core}.ome.tif")) as tf:
            arr = np.stack([tf.pages[VISIBLE[m]].asarray().astype(np.float32) for m in markers])
        t = torch.from_numpy(arr)
        t = blur3(t)
        # torch.quantile has a ~16M element cap; use numpy per channel
        q99 = torch.from_numpy(np.percentile(t.numpy().reshape(t.shape[0], -1), 99, axis=1).astype(np.float32))
        tl = torch.log1p(torch.minimum(t, q99[:, None, None]))
        mu = tl.mean(dim=(1, 2)); sd = tl.std(dim=(1, 2)).clamp_min(1e-6)
        core_stats[core] = (mu.numpy(), sd.numpy())
        # window tensors (native 253 -> 128 bilinear, on the blurred+clipped+log1p image)
        ws = []
        for w in wins[core]:
            r0, c0 = w["r0"], w["c0"]
            win = tl[:, r0:r0 + WIN_NATIVE, c0:c0 + WIN_NATIVE].unsqueeze(0)
            win = F.interpolate(win, size=(WIN_MODEL, WIN_MODEL), mode="bilinear", align_corners=False, antialias=True)[0]
            ws.append(win)
        proc[core] = ws
        del arr, t, tl
    t_prep = time.time() - t0
    print(f"pass1 done in {t_prep:.1f}s")

    # ---------- LOPO z moments + arms ----------
    rows = []
    resource = []
    t_fwd_all = 0.0
    all_mus = {c: core_stats[c][0] for c in cores}
    all_sds = {c: core_stats[c][1] for c in cores}

    def lopo_moments(leave_out):
        sel = [c for c in primary if c != leave_out]
        mu = np.mean([all_mus[c] for c in sel], axis=0)
        sd = np.mean([all_sds[c] for c in sel], axis=0)
        return mu, sd

    t0 = time.time()
    for core in cores:
        branch = "primary" if core != D2 else "sensitivity_D2"
        mu, sd = lopo_moments(core)
        mu_t = torch.from_numpy(mu).float(); sd_t = torch.from_numpy(sd).float()
        for widx, win_log in enumerate(proc[core]):
            win_z = ((win_log - mu_t[:, None, None]) / sd_t[:, None, None]).float()
            # ---- VirTues arm ----
            for tname in TARGETS:
                mask = torch.zeros(len(markers), WIN_MODEL // 8, WIN_MODEL // 8, dtype=torch.bool)
                mask[tpos[tname]] = True
                t1 = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    out = model([win_z.to(device)], [midx.to(device)], [mask.to(device)])
                torch.cuda.synchronize()
                dt = time.time() - t1
                t_fwd_all += dt
                rec = out.decoded_multiplex[0].float().cpu()
                pred_z = rec[tpos[tname]]
                pred_log = pred_z * sd_t[tpos[tname]] + mu_t[tpos[tname]]  # to common log1p scale
                meas_log = win_log[tpos[tname]]
                r_pix = float(np.corrcoef(pred_z.numpy().ravel(), win_z[tpos[tname]].numpy().ravel())[0, 1]) \
                    if np.std(pred_z.numpy()) > 0 and np.std(win_z[tpos[tname]].numpy()) > 0 else float("nan")
                rows.append(dict(branch=branch, core_id=core, patient_unit=core if core != D2 else "D2_sensitivity",
                                 window=widx + 1, target=tname, arm="virtues",
                                 meas_mean_log1p=float(meas_log.mean()),
                                 pred_mean_log1p=float(pred_log.mean()),
                                 pixel_r_zspace=r_pix, fwd_s=round(dt, 4)))
    print("virtues forwards done; ridge arm follows")
    t_virt = time.time() - t0

    # ridge: z-space LOPO closed form
    feats_idx = [i for i, m in enumerate(markers) if m not in TARGETS]
    X_all = {}
    for core in cores:
        mu, sd = lopo_moments(core)
        mu_t = torch.from_numpy(mu).float(); sd_t = torch.from_numpy(sd).float()
        Xs = []
        for widx, win_log in enumerate(proc[core]):
            win_z = ((win_log - mu_t[:, None, None]) / sd_t[:, None, None]).float()
            cov = win_z[feats_idx]  # (n_cov, 128,128)
            X = cov.reshape(len(feats_idx), -1).T.numpy()  # (16384, n_cov)
            Xs.append(X)
        X_all[core] = np.concatenate(Xs, axis=0)  # (2*16384, n_cov)
    for core in cores:
        branch = "primary" if core != D2 else "sensitivity_D2"
        mu, sd = lopo_moments(core)
        mu_t = torch.from_numpy(mu).float(); sd_t = torch.from_numpy(sd).float()
        tr_cores = [c for c in primary if c != core]
        Xtr = np.concatenate([X_all[c] for c in tr_cores], axis=0)
        XtX = Xtr.T @ Xtr + ALPHA * np.eye(Xtr.shape[1])
        for widx, win_log in enumerate(proc[core]):
            win_z = ((win_log - mu_t[:, None, None]) / sd_t[:, None, None]).float()
            Xte = win_z[feats_idx].reshape(len(feats_idx), -1).T.numpy()
            for tname in TARGETS:
                ytr = np.concatenate([
                    (((proc[c][j] - torch.from_numpy(lopo_moments(c)[0]).float()[:, None, None]) /
                      torch.from_numpy(lopo_moments(c)[1]).float()[:, None, None])[tpos[tname]]).numpy().ravel()
                    for c in tr_cores for j in range(len(proc[c]))])
                w = np.linalg.solve(XtX, Xtr.T @ ytr)
                pred_z = (Xte @ w).reshape(WIN_MODEL, WIN_MODEL)
                meas_z = win_z[tpos[tname]].numpy()
                pred_log = pred_z * sd[tpos[tname]] + mu[tpos[tname]]
                rows.append(dict(branch=branch, core_id=core, patient_unit=core if core != D2 else "D2_sensitivity",
                                 window=widx + 1, target=tname, arm="ridge_lopo",
                                 meas_mean_log1p=float(win_log[tpos[tname]].mean()),
                                 pred_mean_log1p=float(pred_log.mean()),
                                 pixel_r_zspace=float(np.corrcoef(pred_z.ravel(), meas_z.ravel())[0, 1])
                                 if np.std(pred_z) > 0 else float("nan"),
                                 fwd_s=0.0))
    t_all = time.time() - t0
    out_csv = os.path.join(OUT, "patient_predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    manifest = dict(
        mode=("SMOKE_NOT_FOR_PAPER" if a.smoke else "PRIMARY_RUN"),
        seed=SEED, alpha=ALPHA, window_regime="A", win_native=WIN_NATIVE, win_model=WIN_MODEL,
        px_um=PX, visible_markers=markers, targets=TARGETS, d2_branch=D2,
        cores=cores, n_primary=len(primary),
        recipe="blur3(reflect)->per-core q99 clip->log1p->LOPO z moments(other primary cores); scoring on log1p common scale",
        timing=dict(pass1_s=round(t_prep, 1), virtues_total_s=round(t_fwd_all, 1),
                    total_s=round(t_all, 1), n_forwards=len([r for r in rows if r['arm'] == 'virtues'])),
        candidates_sha16=sha16(CAND),
        weights_sha16=sha16(os.path.join(ROOT, "data", "weights", "virtues-sp32", "model.safetensors")),
    )
    with open(os.path.join(OUT, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print("written:", out_csv)
    print(json.dumps(manifest["timing"], indent=1))
    print("DONE", tag)


if __name__ == "__main__":
    main()
