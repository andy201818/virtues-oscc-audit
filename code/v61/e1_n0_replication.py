#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1-N0: replicate original Task-2 reconstructions with the NEW execution path.

For a sample of cached windows (pixel_null_v2/cache), recompute `rec` from the
cached `crop` using the streaming adapter path (same model, same patch-level
full-channel mask convention as pn2_cache.py L96-102), and compare against the
cached `rec` from the original run.

Pass criteria (execution-path equivalence on real inputs): per-window
per-marker max|Δrec| and the downstream metric (masked-region Pearson r on
tissue mask) must agree to float-rounding tolerance; report distribution, no
silent threshold relaxation.

Output: outputs/e1/n0_replication.json + n0_rec_diff.csv
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
import torch

from e0_streaming_adapter import load_model, EMB_DIR
from virtues.utils.utils import load_marker_embedding_dict

CACHE = os.path.join(ROOT, r"outputs\revision_20260901\pixel_null_v2\cache")
MDICT = {  # marker -> uniprot (same table as pn2_cache.py)
 'CD11b':'P11215','CD11c':'P20702','CD14':'P08571','CD15':'P31997','CD16':'P08637',
 'CD163':'Q86VB7','CD20':'P11836','CD206':'P22897','CD209':'Q9NNX6','CD3':'P07766',
 'CD31':'P16284','CD36':'P16671','CD4':'P01730','CD44':'P16070','CD45':'P08575',
 'CD45RA':'P08575','CD56':'P13591','CD68':'P34810','CD86':'P42081','CD8a':'P01732',
 'Collagen':'P02452','E-Cadherin':'P12830','FoxP3':'Q9BZS1','GranzymeB':'P10144',
 'HistoneH3':'P68431','Ki67':'P46013','Pancytokeratin':'P08727','Podoplanin':'Q86YL7',
 'VEGF':'P15692','Vimentin':'P08670','aSMA':'P62736','pCREB':'P16220','pERK':'P28482',
 'pMAPKAPK2':'P49137','pNFkB':'Q04206','pS6':'P62753','pSTAT1':'P42224','pSTAT3':'P40763','pp38':'Q16539'}
OUTJ = r"<project-root>\outputs\e1\n0_replication.json"
OUTC = r"<project-root>\outputs\e1\n0_rec_diff.csv"

SAMPLE = ["UOP_S01_1_w320_128", "UOP_S01_1_w320_640", "STA_S02_2_w320_320",
          "STA_S02_2_w320_384", "UOP_S03_3_w192_128", "STA_S03_3_w192_128"]
MARKERS_SAMPLE = ["CD45", "CD31", "Ki67", "aSMA", "pCREB", "Collagen"]


def masked_r(crop, rec, ci, tm):
    a = crop[ci][tm]; b = rec[ci][tm]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pick_sample():
    m = json.load(open(os.path.join(CACHE, "..", "cache_manifest.json"), encoding="utf-8"))
    wins = m["windows"]
    uop = [w for w in wins if w["cohort"] == "UOP"]
    sta = [w for w in wins if w["cohort"] == "STA"]
    # cover both cohorts and different tissues: first window of each cohort's first two distinct tissues + one middle-index window per cohort
    def two_diff_tissue(ws):
        seen, out = set(), []
        for w in ws:
            if w["tissue"] not in seen:
                seen.add(w["tissue"]); out.append(w["key"])
            if len(out) == 2:
                break
        return out
    sel = two_diff_tissue(uop) + two_diff_tissue(sta) + [uop[len(uop) // 2]["key"], sta[len(sta) // 2]["key"]]
    return sel


def main():
    global SAMPLE
    SAMPLE = pick_sample()
    print("sample:", SAMPLE)
    md = load_marker_embedding_dict(EMB_DIR)
    model = load_model()
    rows = []
    for key in SAMPLE:
        p = os.path.join(CACHE, key + ".npz")
        z = np.load(p, allow_pickle=True)
        crop, rec0, tm, names = z["crop"], z["rec"], z["tm"], [str(n) for n in z["names"]]
        midxs = torch.tensor([md[MDICT[n]] for n in names]).cuda()
        t = torch.from_numpy(crop).float().cuda()
        rec1 = np.zeros_like(rec0)
        t0 = time.time()
        for ci in range(len(names)):
            m = torch.zeros(len(names), 16, 16, dtype=torch.bool, device="cuda")
            m[ci] = True
            with torch.no_grad(), torch.amp.autocast("cuda"):
                o = model([t], [midxs], [m])
            rec1[ci] = o.decoded_multiplex[0].float().cpu().numpy()[ci]
        dt = time.time() - t0
        diffs = np.abs(rec1 - rec0)
        r0s, r1s = [], []
        for ci in range(len(names)):
            r0s.append(masked_r(crop, rec0, ci, tm))
            r1s.append(masked_r(crop, rec1, ci, tm))
        r0a, r1a = np.array(r0s), np.array(r1s)
        dr = np.abs(r1a - r0a)
        rows.append(dict(key=key, n_markers=len(names), fwd_s=round(dt, 1),
                         rec_max_abs=float(diffs.max()), rec_p99=float(np.percentile(diffs, 99)),
                         r_orig_mean=round(float(np.nanmean(r0a)), 6),
                         r_new_mean=round(float(np.nanmean(r1a)), 6),
                         r_max_abs_diff=float(np.nanmax(dr)),
                         marker_sample={MARKERS_SAMPLE[i]:
                             dict(r0=round(float(r0s[names.index(MARKERS_SAMPLE[i])]), 6) if MARKERS_SAMPLE[i] in names else None)
                             for i in range(min(len(MARKERS_SAMPLE), len(names)))}))
        print(rows[-1]["key"], "rec_max_abs=%.2e" % rows[-1]["rec_max_abs"],
              "r_orig=%.6f r_new=%.6f dr_max=%.2e" % (rows[-1]["r_orig_mean"], rows[-1]["r_new_mean"], rows[-1]["r_max_abs_diff"]))
    worst = max(r["rec_max_abs"] for r in rows)
    worst_dr = max(r["r_max_abs_diff"] for r in rows)
    verdict = dict(
        n_windows=len(rows), n_forwards_total=sum(r["n_markers"] for r in rows),
        worst_rec_max_abs=worst, worst_metric_abs_diff=worst_dr,
        interpretation="if both rec and metric differences are at float-rounding level, execution-path equivalence holds on real data",
        pass_rule="worst_rec_max_abs < 1e-4 and worst_metric_abs_diff < 1e-4 (engineering screening start, not relaxed)",
        passed=bool(worst < 1e-4 and worst_dr < 1e-4),
        caveat="crop taken directly from the original cache: verifies the inference execution path only, not the standardized reconstruction (standardization is verified separately in N1)",
        rows=rows,
    )
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=1, ensure_ascii=False)
    print("N0 VERDICT:", "PASS" if verdict["passed"] else "FAIL",
          "| worst rec diff %.2e, worst metric diff %.2e" % (worst, worst_dr))


if __name__ == "__main__":
    main()
