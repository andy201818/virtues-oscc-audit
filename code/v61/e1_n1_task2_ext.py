#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1-N1 Task 2 (extension): patient-level endpoints under fold standardization.

Audit fix R4: the first N1 run only stored window-marker correlations; this
run additionally stores, for the VirTues arm, per-window measured and
predicted MEANS on the log1p common scale (inverse-z with the respective
moments: N0 = global gms, N1 = fold LOPO moments), and derives patient-level
endpoints (patient = mean over its 2 windows): Spearman(meas, pred),
MAE, and paired N0-vs-N1 differences per marker-cohort.

Window-level r values are re-emitted for cross-check against the first run.
Output: outputs/e1/n1_task2_ext_predictions.csv + n1_task2_patient_endpoints.csv
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
from scipy.stats import spearmanr

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
    return tid.rsplit("_", 1)[0]


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
    stats = pd.read_parquet(PARQ)
    stats["patient"] = stats["roi"].map(patient_of)
    qdf = pd.read_csv(os.path.join(ZOUT, "standardization_stats.csv"), index_col=0)
    gms = pd.read_csv(os.path.join(ZOUT, "global_mean_std.csv"), index_col=0)
    md = load_marker_embedding_dict(EMB_DIR)
    model = load_model()
    manif = json.load(open(os.path.join(PNULL, "cache_manifest.json"), encoding="utf-8"))
    wins = manif["windows"]
    print(f"{len(wins)} windows", flush=True)

    mom_cache = {}
    def lopo_moments(patient):
        if patient not in mom_cache:
            sub = stats[stats["patient"] != patient]
            g = sub.groupby("marker").agg(sum=("sum", "sum"), sq=("sq", "sum"), n=("n", "sum"))
            mu = (g["sum"] / g["n"])
            var = (g["sq"] / g["n"] - mu ** 2).clip(lower=0)
            mom_cache[patient] = (mu.to_dict(), np.sqrt(var).to_dict())
        return mom_cache[patient]

    rows = []
    t0 = time.time()
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
        mu, sd = lopo_moments(patient_of(tid))
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
        z = np.load(os.path.join(PNULL, "cache", w["key"] + ".npz"), allow_pickle=True)
        crop0, rec0, names0 = z["crop"], z["rec"], [str(x) for x in z["names"]]
        for ci, n in enumerate(names):
            # N1 (fold moments) on log1p common scale
            meas_n1 = float(xlog[ci, r0:r0+128, c0:c0+128][tm].mean())
            pred_n1 = float((rec1[ci][tm] * sd_a[ci] + mu_a[ci]).mean())
            r1 = float(np.corrcoef(xn[ci, r0:r0+128, c0:c0+128][tm], rec1[ci][tm])[0, 1])
            # N0 from cache (global moments)
            cj = names0.index(n)
            mu0 = float(gms.loc[n, "mean"]); sd0 = float(gms.loc[n, "std"])
            meas_n0 = float((crop0[cj][tm] * sd0 + mu0).mean())
            pred_n0 = float((rec0[cj][tm] * sd0 + mu0).mean())
            r0v = float(np.corrcoef(crop0[cj][tm], rec0[cj][tm])[0, 1])
            rows.append(dict(window=w["key"], cohort=w["cohort"], patient=patient_of(tid),
                             marker=n, r_n0=r0v, r_n1=r1,
                             meas_log_n0=meas_n0, pred_log_n0=pred_n0,
                             meas_log_n1=meas_n1, pred_log_n1=pred_n1))
        if (wi + 1) % 12 == 0:
            print(f"[{wi+1}/{len(wins)}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTD, "n1_task2_ext_predictions.csv"), index=False, encoding="utf-8")
    print(f"predictions rows: {len(df)}", flush=True)

    # patient-level endpoints: patient = mean over its 2 windows
    agg = df.groupby(["cohort", "patient", "marker"]).agg(
        meas_n0=("meas_log_n0", "mean"), pred_n0=("pred_log_n0", "mean"),
        meas_n1=("meas_log_n1", "mean"), pred_n1=("pred_log_n1", "mean")).reset_index()
    out = []
    for (co, mk), g in agg.groupby(["cohort", "marker"]):
        e0 = (g.meas_n0 - g.pred_n0).abs(); e1 = (g.meas_n1 - g.pred_n1).abs()
        out.append(dict(cohort=co, marker=mk, n_patients=len(g),
                        spearman_n0=round(float(spearmanr(g.meas_n0, g.pred_n0).statistic), 4),
                        spearman_n1=round(float(spearmanr(g.meas_n1, g.pred_n1).statistic), 4),
                        mae_n0=round(float(e0.mean()), 4), mae_n1=round(float(e1.mean()), 4),
                        d_mae=round(float((e1 - e0).mean()), 6),
                        max_abs_d_mae=round(float((e1 - e0).abs().max()), 6),
                        meas_shift=round(float((g.meas_n1 - g.meas_n0).mean()), 6)))
    pe = pd.DataFrame(out)
    pe.to_csv(os.path.join(OUTD, "n1_task2_patient_endpoints.csv"), index=False, encoding="utf-8")
    summary = dict(
        n_marker_cohort=len(pe), n_patients_per_combo=int(pe.n_patients.iloc[0]),
        mean_d_mae=round(float(pe.d_mae.mean()), 6), sd_d_mae=round(float(pe.d_mae.std()), 6),
        max_abs_d_mae=round(float(pe.max_abs_d_mae.max()), 6),
        spearman_change_abs_mean=round(float((pe.spearman_n1 - pe.spearman_n0).abs().mean()), 6),
        design="VirTues arm only (ridge side unchanged by N1 by construction); log1p common scale; tissue-mask pixels; patient = mean of 2 windows",
    )
    with open(os.path.join(OUTD, "n1_task2_patient_endpoints_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print("DONE ext", flush=True)


if __name__ == "__main__":
    main()
