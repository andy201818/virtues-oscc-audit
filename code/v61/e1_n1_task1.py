#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1-N1 Task 1: fold-standardized token re-encoding + probe re-evaluation.

encode (GPU, long): per fold f (frozen task1_patient_folds.json), re-standardize
each of the 142 ROIs with moments estimated ONLY from that fold's training
patients (sufficient stats from roi_channel_stats.parquet), then re-encode cell
tokens via the streaming adapter (bit-identical to library path, memory-safe).
Everything else verbatim from kao1_cell_typing.py (pad 120, blur, per-ROI q99,
log1p, seg argmax, celltypes labels). Output: outputs/e1/tokens_n1/fold{f}/.

probe (CPU): frozen folds, LogisticRegression(max_iter=1000, n_jobs=1) exactly
as c3_statistics.py; N0 = original cached tokens, N1 = fold tokens; paired
per-fold macro-F1 comparison. Fusion/intensity arms unchanged by design
(intensity arm was already fold-correct; reported as context only).

Usage:
  python e1_n1_task1.py encode
  python e1_n1_task1.py probe
"""
import glob
import json
import os
import re
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

E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")
EMB_DIR = os.path.join(ROOT, "data", "einhaus_markers", "embeddings", "esm2_t30_150M_UR50D")
ZOUT = os.path.join(ROOT, "outputs", "einhaus_zeroshot")
KOUT = os.path.join(ROOT, "outputs", "kao1_celltyping")
PARQ = r"<cache-root>\output\v5.13_submission\public_repository\outputs\revision_20260901\zscore_sensitivity\roi_channel_stats.parquet"
FOLDS = r"<cache-root>\output\v5.13_submission\public_repository\outputs\revision_20260901\zscore_sensitivity\task1_patient_folds.json"
OUTD = r"<project-root>\outputs\e1"
PAD = 120

MDICT = {
 'CD11b':'P11215','CD11c':'P20702','CD14':'P08571','CD15':'P31997','CD16':'P08637',
 'CD163':'Q86VB7','CD20':'P11836','CD206':'P22897','CD209':'Q9NNX6','CD3':'P07766',
 'CD31':'P16284','CD36':'P16671','CD4':'P01730','CD44':'P16070','CD45':'P08575',
 'CD45RA':'P08575','CD56':'P13591','CD68':'P34810','CD86':'P42081','CD8a':'P01732',
 'Collagen':'P02452','E-Cadherin':'P12830','FoxP3':'Q9BZS1','GranzymeB':'P10144',
 'HistoneH3':'P68431','Ki67':'P46013','Pancytokeratin':'P08727','Podoplanin':'Q86YL7',
 'VEGF':'P15692','Vimentin':'P08670','aSMA':'P62736','pCREB':'P16220','pERK':'P28482',
 'pMAPKAPK2':'P49137','pNFkB':'Q04206','pS6':'P62753','pSTAT1':'P42224','pSTAT3':'P40763','pp38':'Q16539'}

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


def patient_of(tid):
    return tid.rsplit("_", 1)[0]


def encode():
    from e0_streaming_adapter import load_model, compute_cell_tokens_streaming
    from virtues.utils.utils import load_marker_embedding_dict
    md = load_marker_embedding_dict(EMB_DIR)
    model = load_model()
    qdf = pd.read_csv(os.path.join(ZOUT, "standardization_stats.csv"), index_col=0)
    stats = pd.read_parquet(PARQ)
    stats["patient"] = stats["roi"].map(patient_of)
    folds = json.load(open(FOLDS, encoding="utf-8"))
    print(f"{len(folds)} folds; parquet {stats['roi'].nunique()} ROIs / {stats['patient'].nunique()} patients", flush=True)

    # ROI list = exactly the cached token set (the 142 with annotations)
    rois = []
    for f in sorted(glob.glob(os.path.join(KOUT, "tokens", "*.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]  # e.g. UOPCohort_OC01_001
        cohort, tid = stem.split("_", 1)
        rois.append((cohort, tid))
    print(f"{len(rois)} ROIs to encode x {len(folds)} folds", flush=True)

    t00 = time.time()
    n_done = 0
    for fd in folds:
        f = fd["fold"]
        fdir = os.path.join(OUTD, "tokens_n1", f"fold{f}")
        os.makedirs(fdir, exist_ok=True)
        tr_pats = set(fd["train_patients"])
        sub = stats[stats["patient"].isin(tr_pats)]
        g = sub.groupby("marker").agg(sum=("sum", "sum"), sq=("sq", "sum"), n=("n", "sum"))
        mu_g = (g["sum"] / g["n"]); sd_g = np.sqrt((g["sq"] / g["n"] - mu_g ** 2).clip(lower=0))
        for cohort, tid in rois:
            cache = os.path.join(fdir, f"{cohort}_{tid}.npz")
            if os.path.exists(cache):
                continue
            fi = os.path.join(E, cohort, "Steinbock", "img", tid + ".tiff")
            fm = os.path.join(E, cohort, "Steinbock", "masks", tid + ".tiff")
            panel = pd.read_csv(os.path.join(E, cohort, "Steinbock", "panel.csv"))
            names_all = panel["name"].tolist()
            keep_idx = [j for j, n in enumerate(names_all) if n in MDICT]
            keep_names = [n for n in names_all if n in MDICT]
            midxs = torch.tensor([md[MDICT[n]] for n in keep_names], dtype=torch.long)
            raw = tifffile.imread(fi).astype(np.float32)
            msk = tifffile.imread(fm)
            if msk.ndim == 2:
                msk = msk[None]
            raw = np.pad(raw, ((0, 0), (PAD, PAD), (PAD, PAD)))
            msk = np.pad(msk, ((0, 0), (PAD, PAD), (PAD, PAD)))
            xb = blur(raw)
            xs = np.stack([xb[j] for j in keep_idx])
            qs = np.array([qdf.loc[tid, n] for n in keep_names], dtype=np.float32)
            means = np.array([mu_g.get(n, np.nan) for n in keep_names], dtype=np.float32)
            stds = np.array([sd_g.get(n, np.nan) for n in keep_names], dtype=np.float32)
            assert not np.isnan(means).any() and not np.isnan(stds).any(), f"fold moments missing marker in {tid}"
            xn = ((np.log1p(np.clip(xs, 0, qs[:, None, None])) - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
            seg = torch.from_numpy(msk.astype(np.int32)).argmax(dim=0) if msk.shape[0] > 1 else torch.from_numpy(msk[0].astype(np.int32))
            x_t = torch.from_numpy(xn).float()
            with torch.no_grad():
                cell_ids, cell_tokens, _, _ = compute_cell_tokens_streaming(
                    model, x_t, midxs, seg, device="cuda")
            ann = pd.read_csv(os.path.join(E, cohort, "Dataframes", f"{cohort[:3]}_allcells.csv"),
                              usecols=["sample_id", "ObjectNumber", "celltypes"])
            labels = ann[ann.sample_id == tid].set_index("ObjectNumber")["celltypes"].to_dict()
            cids = cell_ids.numpy()
            lab = np.array([labels.get(int(c), None) for c in cids], dtype=object)
            keep = lab != None
            np.savez_compressed(cache, ids=cids[keep], tokens=cell_tokens.numpy()[keep], labels=lab[keep])
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t00
                print(f"[{time.strftime('%H:%M:%S')}] {n_done} encodings done, {el:.0f}s elapsed, {el/n_done:.1f}s/ROI", flush=True)
    print(f"ENCODE DONE: {n_done} new + skipped-existing; total {time.time()-t00:.0f}s", flush=True)


def load_stack(base):
    X, y, gp = [], [], []
    for f in sorted(glob.glob(os.path.join(base, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        stem = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_")
        m = re.match(r"([A-Z]*\d+)", parts[1])
        patient = m.group(1) if m else parts[1]
        X.append(d["tokens"]); y.append(d["labels"])
        gp += [patient] * len(d["labels"])
    return np.concatenate(X), np.concatenate(y), np.array(gp)


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    folds = json.load(open(FOLDS, encoding="utf-8"))
    X0, y, gp0 = load_stack(os.path.join(KOUT, "tokens"))
    res = []
    for fd in folds:
        f = fd["fold"]
        fdir = os.path.join(OUTD, "tokens_n1", f"fold{f}")
        n_files = len(glob.glob(os.path.join(fdir, "*.npz")))
        if n_files < 142:
            print(f"fold{f}: only {n_files}/142 encoded — run encode first", flush=True)
            continue
        X1, y1, gp1 = load_stack(fdir)
        assert len(y) == len(y1), "cell sets differ between N0 and N1 stacks"
        tr = np.isin(gp0, fd["train_patients"]); te = ~tr
        for tag, X in [("n0", X0), ("n1", X1)]:
            clf = LogisticRegression(max_iter=1000, n_jobs=1)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            res.append(dict(fold=f, arm=f"token_{tag}",
                            macro_f1=round(float(f1_score(y[te], pred, average="macro")), 4),
                            accuracy=round(float(accuracy_score(y[te], pred)), 4),
                            n_test=int(te.sum())))
            print(res[-1], flush=True)
    df = pd.DataFrame(res)
    piv = df.pivot_table(index="fold", columns="arm", values="macro_f1")
    piv["d_f1"] = piv["token_n1"] - piv["token_n0"]
    df.to_csv(os.path.join(OUTD, "n1_task1_folds.csv"), index=False, encoding="utf-8")
    piv.to_csv(os.path.join(OUTD, "n1_task1_paired.csv"), encoding="utf-8")
    summary = dict(
        mean_f1_n0=round(float(piv.token_n0.mean()), 4), sd_f1_n0=round(float(piv.token_n0.std()), 4),
        mean_f1_n1=round(float(piv.token_n1.mean()), 4), sd_f1_n1=round(float(piv.token_n1.std()), 4),
        mean_d=round(float(piv.d_f1.mean()), 6), max_abs_d=round(float(piv.d_f1.abs().max()), 6),
        design="frozen folds; per-fold training-patient moments; LogisticRegression(max_iter=1000) verbatim",
    )
    with open(os.path.join(OUTD, "n1_task1_summary.json"), "w", encoding="utf-8") as fjson:
        json.dump(summary, fjson, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print("DONE probe", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "encode"
    if mode == "encode":
        encode()
    elif mode == "probe":
        probe()
    else:
        raise SystemExit(f"unknown mode {mode}")
