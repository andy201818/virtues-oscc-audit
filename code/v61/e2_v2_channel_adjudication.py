#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G2-4 v2-channel adjudication: which image channel feeds Cell..CD45..Mean?

Question: Core images have both CD45 (idx 69) and CD45v2 (idx 33). The published
per-cell table has a single CD45 column. Which image channel is 'the' CD45?

Method (three discriminators on Core_A5):
  1. centroid-window: per table-cell, mean of each channel in an 11x11 px window
     around the cell centroid (um -> px); Spearman vs Cell..CD45..Mean.
  2. patch-aggregation: 64x64 px grid; channel mean vs table mean per patch.
  3. controls: CD31 (idx 41, no v2) positive; DAPI (idx 0) negative.

Verdict rule: the true channel should clearly beat the other v2 candidate and
the negative control on both discriminators.

Output: outputs/e2/gating/v2_adjudication.csv + console verdict.
"""
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import spearmanr

CORE = "A5"
IMG = rf"<project-root>\external_data\S-BIAD3612\images\cores\Core_{CORE}.ome.tif"
TABLE = r"<project-root>\external_data\S-BIAD3612\analysis_tables\Instanseg_final_high_quality_markers.csv"
PX = 0.5066976883697085
WIN = 5  # half-window -> 11x11
CHANNELS = {"CD45": 69, "CD45v2": 33, "CD31": 41, "DAPI": 0, "PanCK": 50}
# target table column per channel: candidates vs CD45; controls vs own marker
TARGET = {"CD45": "Cell..CD45..Mean", "CD45v2": "Cell..CD45..Mean",
          "CD31": "Cell..CD31..Mean", "PanCK": "Cell..PanCK..Mean",
          "DAPI": "Cell..CD45..Mean"}  # DAPI as negative control vs CD45 table
OUT = r"<project-root>\outputs\e2\gating\v2_adjudication.csv"


def main():
    df = pd.read_csv(TABLE, usecols=lambda c: c in (
        "Image", "Centroid.X.µm", "Centroid.Y.µm", "Cell..CD45..Mean", "Cell..CD31..Mean",
        "Cell..PanCK..Mean"))
    df = df[df["Image"].str.startswith(f"fov0_{CORE}_")].reset_index(drop=True)
    print(f"Core {CORE}: {len(df)} cells from table")
    df["px_x"] = (df["Centroid.X.µm"] / PX).astype(int)
    df["px_y"] = (df["Centroid.Y.µm"] / PX).astype(int)

    imgs = {}
    with tifffile.TiffFile(IMG) as tf:
        h, w = tf.series[0].shape[1:]
        for name, idx in CHANNELS.items():
            imgs[name] = tf.pages[idx].asarray().astype(np.float32)
    print("image:", h, "x", w)

    # clip windows to bounds
    ys = df["px_y"].to_numpy(); xs = df["px_x"].to_numpy()
    ok = (ys >= WIN) & (ys < h - WIN) & (xs >= WIN) & (xs < w - WIN)
    df = df[ok].reset_index(drop=True)
    ys, xs = df["px_y"].to_numpy(), df["px_x"].to_numpy()
    print(f"in-bounds cells: {len(df)}")

    rows = []
    for name in CHANNELS:
        im = imgs[name]
        # 1) per-cell centroid window mean
        win_means = np.array([
            im[y - WIN:y + WIN + 1, x - WIN:x + WIN + 1].mean() for y, x in zip(ys, xs)
        ])
        r1 = spearmanr(win_means, df[TARGET[name]]).statistic
        # 2) patch aggregation (64 px grid)
        P = 64
        gp = h // P, w // P
        im_patch = np.zeros(gp); tab_patch = np.zeros(gp); cnt = np.zeros(gp)
        py, px_ = ys // P, xs // P
        for y, x, v in zip(py, px_, df[TARGET[name]]):
            tab_patch[y, x] += v; cnt[y, x] += 1
        for i in range(gp[0]):
            for j in range(gp[1]):
                im_patch[i, j] = im[i*P:(i+1)*P, j*P:(j+1)*P].mean()
        m = cnt > 20  # patches with enough cells
        r2 = spearmanr(im_patch[m], (tab_patch / np.maximum(cnt, 1))[m]).statistic
        rows.append(dict(channel=name, index=CHANNELS[name], target=TARGET[name],
                         spearman_centroid_window=float(r1),
                         spearman_patch64=float(r2),
                         n_cells=int(len(df)), n_patches=int(m.sum())))
        print(rows[-1])

    # verdict
    by = {r["channel"]: r for r in rows}
    cd45, cd45v2 = by["CD45"], by["CD45v2"]
    cd31, dapi, panck = by["CD31"], by["DAPI"], by["PanCK"]
    win_beats = cd45["spearman_centroid_window"] - cd45v2["spearman_centroid_window"]
    pat_beats = cd45["spearman_patch64"] - cd45v2["spearman_patch64"]
    verdict = (
        f"CD45(idx69) beats CD45v2(idx33) by {win_beats:+.3f} (window) / {pat_beats:+.3f} (patch); "
        f"positive controls CD31/PanCK window rho={cd31['spearman_centroid_window']:.3f}/{panck['spearman_centroid_window']:.3f}; "
        f"negative control DAPI-vs-CD45table rho={dapi['spearman_centroid_window']:.3f}"
    )
    print("VERDICT:", verdict)
    pd.DataFrame(rows).assign(verdict=verdict).to_csv(OUT, index=False, encoding="utf-8")
    print("written:", OUT)


if __name__ == "__main__":
    main()
