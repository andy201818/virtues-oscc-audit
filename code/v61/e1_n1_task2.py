#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1-N1 Task 2: fold-standardization re-run (patient-excluded moments).

For each of the 72 cached Task-2 windows: recompute the window with moments
estimated from all 142 ROIs EXCLUDING the left-out patient's ROIs (LOPO),
re-run the per-marker full-channel-masked reconstruction, and compare the
masked-region Pearson r against N0 (original global moments, from cache).

Design anchors (enhancement plan 4.2):
  - q99 per ROI (qdf) unchanged: not patient-dependent
  - moments domain = all 142 ROIs (both cohorts), patient-level exclusion
  - windows/selection unchanged (cache_manifest r0/c0; no re-selection)
  - frozen model, same mask convention as pn2_cache.py

Output: outputs/e1/n1_task2_paired.csv + n1_task2_summary.json
"""
import glob
import json
import os
import sys
import time

ROOT = r"<data-root>\virtual cell and tissue"
sys.path.insert(0, os.path.join(ROOT, "code", "Virtues"))
os.chdir(os.path.join(ROOT, "code", "Virtues"))
sys.path.insert(0, r"<project-root>\scripts")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile

from e0_streaming_adapter import load_model, EMB_DIR
from virtues.utils.utils import load_marker_embedding_dict

E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")
ZOUT = os.path.join(ROOT, "outputs", "einhaus_zeroshot")
PNULL = os.path.join(ROOT, "outputs", "revision_20260901", "pixel_null_v2")
PARQ = r"<cache-root>\output\v5.13_submission\public_repository\outputs\revision_20260901\zscore_sensitivity\roi_channel_stats.parquet"
OUTD = r"<project-root>\outputs\e1"

MDICT = {
 'CD11b':'P11215','CD11c':'P20702','CD14':'P08571','CD15':'P31997','CD16':'P08637',
 'CD163':'Q86VB7','CD20':'P11836','CD206':'P22897','CD209':'Q9NNX6','CD3':'P07766',
 'CD31':'P16284','CD36':'P16671','CD4':'P01730','CD44':'P16070','CD45':'P08575',
 'CD45RA':'P08575','CD56':'P13591','CD68':'P34810','CD86':'P42081','CD8a':'P01732',
 'Collagen':'P02452','E-Cadherin':'P12830','FoxP3':'Q9BZS1','GranzymeB':'P10144',
 'HistoneH3':'P68431','Ki67':'P46013','Pancytokeratin':'P08727','Podoplanin':'Q86YL7',
 'VEGF':'P15692','Vimentin':'P08670','aSMA':'P62736','pCREB':'P16220','pERK':'P28482',
 'pMAPKAPK2':'P49137','pNFkB':'Q04206','pS6':'P62753','pSTAT1':'P42224','pSTAT3':'P40763','pp38':'Q16539'}


def patient_of(tid):
    return tid.rsplit("_", 1)[0]  # OC01_001 -> OC01 ; S01_1 -> S01


_GK = None
def blur(x):
    global _GK
    k = 3
    if _GK is None:
        ax = torch.arange(k).float() - 1
        g = torch.exp(-(ax ** 2) / 2.0); g /= g.sum()
        _GK = g[:, None] @ g[None, :]
    t = torch.from_numpy(x).float().unsqueeze(0)
    w = _GK.reshape(1, 1, k, k).expand(x.shape[0], 1, k, k).contiguous()
    return F.conv2d(F.pad(t, (1, 1, 1, 1), mode='reflect'), w, groups=x.shape[0])[0].numpy()


def main():
    os.makedirs(OUTD, exist_ok=True)
    stats = pd.read_parquet(PARQ)
    stats["patient"] = stats["roi"].map(patient_of)
    n_pat = stats["patient"].nunique()
    print(f"parquet: {stats['roi'].nunique()} ROIs, {n_pat} patients (assert 142/48)")
    assert stats["roi"].nunique() == 142 and n_pat == 48, "ROI/patient counts do not match expectations"

    qdf = pd.read_csv(os.path.join(ZOUT, "standardization_stats.csv"), index_col=0)
    md = load_marker_embedding_dict(EMB_DIR)
    model = load_model()
    manif = json.load(open(os.path.join(PNULL, "cache_manifest.json"), encoding="utf-8"))
    wins = manif["windows"]
    print(f"{len(wins)} windows")

    # moments cache: per left-out patient
    mom_cache = {}
    def lopo_moments(patient, markers):
        if patient in mom_cache:
            return mom_cache[patient]
        sub = stats[stats["patient"] != patient]
        g = sub.groupby("marker").agg(sum=("sum", "sum"), sq=("sq", "sum"), n=("n", "sum"))
        mu = (g["sum"] / g["n"]).to_dict()
        var = (g["sq"] / g["n"] - (g["sum"] / g["n"]) ** 2).clip(lower=0)
        sd = np.sqrt(var).to_dict()
        mom_cache[patient] = (mu, sd)
        return mu, sd

    rows = []
    t_start = time.time()
    n_fwd = 0
    for wi, w in enumerate(wins):
        cohort_dir = "UOPCohort" if w["cohort"] == "UOP" else "STACohort"
        tid, r0, c0 = w["tissue"], w["r0"], w["c0"]
        fi = os.path.join(E, cohort_dir, "Steinbock", "img", tid + ".tiff")
        fm = os.path.join(E, cohort_dir, "Steinbock", "masks", tid + ".tiff")
        raw = tifffile.imread(fi).astype(np.float32)
        msk = tifffile.imread(fm)
        tissue = (msk.sum(axis=0) if msk.ndim == 3 else msk) > 0
        tm = tissue[r0:r0+128, c0:c0+128]
        panel = pd.read_csv(os.path.join(E, cohort_dir, "Steinbock", "panel.csv"))
        names = [n for n in panel["name"].tolist() if n in MDICT]
        keep_idx = [j for j, n in enumerate(panel["name"].tolist()) if n in MDICT]
        xb = blur(raw)
        xs = np.stack([xb[j] for j in keep_idx])
        qs = np.array([qdf.loc[tid, n] for n in names], dtype=np.float32)
        xlog = np.log1p(np.clip(xs, 0, qs[:, None, None]))
        mu, sd = lopo_moments(patient_of(tid), names)
        mu_a = np.array([mu[n] for n in names], dtype=np.float32)
        sd_a = np.array([sd[n] for n in names], dtype=np.float32)
        xn = ((xlog - mu_a[:, None, None]) / sd_a[:, None, None]).astype(np.float32)
        crop = torch.from_numpy(xn[:, r0:r0+128, c0:c0+128]).float().cuda()
        midxs = torch.tensor([md[MDICT[n]] for n in names]).cuda()
        rec1 = np.zeros_like(crop.cpu().numpy())
        for ci in range(len(names)):
            m = torch.zeros(len(names), 16, 16, dtype=torch.bool, device="cuda")
            m[ci] = True
            with torch.no_grad(), torch.amp.autocast("cuda"):
                o = model([crop], [midxs], [m])
            rec1[ci] = o.decoded_multiplex[0].float().cpu().numpy()[ci]
            n_fwd += 1
        # N0 from cache
        z = np.load(os.path.join(PNULL, "cache", w["key"] + ".npz"), allow_pickle=True)
        crop0, rec0, names0 = z["crop"], z["rec"], [str(x) for x in z["names"]]
        for ci, n in enumerate(names):
            a = xn[:, r0:r0+128, c0:c0+128][ci][tm]
            b = rec1[ci][tm]
            r1 = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
            cj = names0.index(n)
            a0 = crop0[cj][tm]; b0 = rec0[cj][tm]
            r0v = float(np.corrcoef(a0, b0)[0, 1]) if np.std(a0) > 0 and np.std(b0) > 0 else float("nan")
            rows.append(dict(window=w["key"], cohort=w["cohort"], patient=patient_of(tid),
                             marker=n, r_n0=r0v, r_n1=r1, d_r=r1 - r0v))
        if (wi + 1) % 12 == 0:
            print(f"[{wi+1}/{len(wins)}] forwards={n_fwd} elapsed={time.time()-t_start:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTD, "n1_task2_paired.csv"), index=False)
    summary = dict(
        n_windows=len(wins), n_forwards=n_fwd, n_rows=len(df),
        elapsed_s=round(time.time() - t_start, 1),
        design="LOPO moments from 142-ROI sufficient stats (patient-level exclusion); qdf/window/model frozen",
        mean_r_n0=round(float(df.r_n0.mean()), 6), mean_r_n1=round(float(df.r_n1.mean()), 6),
        mean_d=round(float(df.d_r.mean()), 6), sd_d=round(float(df.d_r.std()), 6),
        max_abs_d=round(float(df.d_r.abs().max()), 6),
        per_cohort={c: dict(mean_r_n0=round(float(g.r_n0.mean()), 6), mean_r_n1=round(float(g.r_n1.mean()), 6),
                            mean_d=round(float(g.d_r.mean()), 6)) for c, g in df.groupby("cohort")},
        per_marker_mean_d={n: round(float(g.d_r.mean()), 6) for n, g in df.groupby("marker")},
    )
    with open(os.path.join(OUTD, "n1_task2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_marker_mean_d"}, indent=1, ensure_ascii=False))
    print("DONE n1_task2")


if __name__ == "__main__":
    main()
