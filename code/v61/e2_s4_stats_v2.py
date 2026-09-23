#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 v2 statistics: patient-level paired metrics from CELL-level predictions.

Per audit fix (item 5): bootstrap reported as PERCENTILE unless true BCa is
computed — this implements real BCa (bias-correction z0 + jackknife
acceleration a) with NA+reason when degenerate. All counts printed are read
from the data file at run time (audit lesson: never from memory).

Unit: patient value = equal-weight mean over that core's cells (per window
cells pooled); primary = 27 cores; D2 = appended-core sensitivity (fixed
predictions, relabelled per audit item 12).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

CSV = r"<project-root>\outputs\e2\s4_v2\patient_predictions_cells.csv"
OUT = r"<project-root>\outputs\e2\s4_v2\paired_metrics.csv"
B = 10000
SEED = 20260920
TARGETS = ["CD45", "CD31", "CD68", "CD163", "Ki67", "aSMA"]


def bca_percentile(theta_hat, stat, n, rng):
    """True BCa: z0 from bootstrap, a from jackknife. Returns (lo, hi, method, note)."""
    idx = rng.integers(0, n, size=(B, n))
    boots = np.array([stat(i) for i in idx])
    ok = np.isfinite(boots)
    boots = boots[ok]
    if len(boots) < 0.9 * B:
        return (np.nan, np.nan, "NA", f"only {len(boots)}/{B} finite replicates")
    z0 = norm.ppf(np.clip(np.mean(boots < theta_hat), 1e-6, 1 - 1e-6))
    # jackknife acceleration
    jk = np.array([stat(np.concatenate([np.arange(i), np.arange(i + 1, n)])) for i in range(n)])
    jk = jk[np.isfinite(jk)]
    if len(jk) == n and np.std(jk) > 0:
        jmean = jk.mean()
        a = np.sum((jmean - jk) ** 3) / (6 * (np.sum((jmean - jk) ** 2) + 1e-30) ** 1.5)
        alpha = 0.05
        # standard BCa adjusted percentiles (Efron): p = Phi(z0 + (z0 + z_a) / (1 - a (z0 + z_a)))
        zlo, zhi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
        p_lo = norm.cdf(z0 + (z0 + zlo) / (1 - a * (z0 + zlo)))
        p_hi = norm.cdf(z0 + (z0 + zhi) / (1 - a * (z0 + zhi)))
        lo, hi = np.percentile(boots, [100 * p_lo, 100 * p_hi])
        return (float(lo), float(hi), "BCa", f"z0={z0:.3f}, a={a:.4f}")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(lo), float(hi), "percentile", "jackknife degenerate")


def main():
    df = pd.read_csv(CSV)
    prim = df[df.branch == "primary"]
    rng = np.random.default_rng(SEED)
    counts = {}
    rows = []
    for t in TARGETS:
        sub = prim[prim.target == t]
        agg = sub.groupby(["core_id", "arm"]).agg(meas=("meas", "mean"), pred=("pred", "mean")).reset_index()
        pv = agg[agg.arm == "virtues"].set_index("core_id")
        pr = agg[agg.arm == "ridge_lopo"].set_index("core_id")
        cores = sorted(set(pv.index) & set(pr.index))
        n = len(cores)
        meas = pv.loc[cores, "meas"].to_numpy(); mV = pv.loc[cores, "pred"].to_numpy(); mR = pr.loc[cores, "pred"].to_numpy()
        meas_ok = bool(np.allclose(meas, pr.loc[cores, "meas"].to_numpy()))
        rhoV = spearmanr(meas, mV).statistic; rhoR = spearmanr(meas, mR).statistic
        eV = np.abs(meas - mV); eR = np.abs(meas - mR); d = eV - eR
        def s_rhoV(i): return spearmanr(meas[i], mV[i]).statistic
        def s_rhoR(i): return spearmanr(meas[i], mR[i]).statistic
        def s_drho(i): return s_rhoV(i) - s_rhoR(i)
        def s_d(i): return d[i].mean()
        ciV = bca_percentile(rhoV, s_rhoV, n, rng)
        ciR = bca_percentile(rhoR, s_rhoR, n, rng)
        cid = bca_percentile(rhoV - rhoR, s_drho, n, rng)
        cia = bca_percentile(d.mean(), s_d, n, rng)
        rows.append(dict(target=t, n_patients=n, meas_identical_across_arms=meas_ok,
                         spearman_virtues=round(float(rhoV), 4),
                         ci_virtues=f"[{ciV[0]:.4f},{ciV[1]:.4f}] ({ciV[2]})",
                         spearman_ridge=round(float(rhoR), 4),
                         ci_ridge=f"[{ciR[0]:.4f},{ciR[1]:.4f}] ({ciR[2]})",
                         delta_rho=round(float(rhoV - rhoR), 4),
                         ci_delta_rho=f"[{cid[0]:.4f},{cid[1]:.4f}] ({cid[2]})",
                         mae_virtues=round(float(eV.mean()), 4), mae_ridge=round(float(eR.mean()), 4),
                         paired_dabs_mean=round(float(d.mean()), 4),
                         ci_dabs=f"[{cia[0]:.4f},{cia[1]:.4f}] ({cia[2]})",
                         n_cells_total=int(sub[sub.arm == 'virtues'].shape[0] // 1)))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    # counts computed from file (lesson: no memory)
    w = pd.read_csv(r"<project-root>\outputs\e2\s4_v2\window_predictions_secondary.csv")
    pv2 = w[(w.branch == "primary")]
    spatial_lead_V = 0; mae_lead_R = 0; order_lead_V = 0
    for t in TARGETS:
        r = out[out.target == t].iloc[0]
        if r.spearman_virtues > r.spearman_ridge: order_lead_V += 1
        if r.mae_virtues > r.mae_ridge: mae_lead_R += 1
        sv = pv2[(pv2.target == t) & (pv2.arm == "virtues")].pixel_r_z.mean()
        sr = pv2[(pv2.target == t) & (pv2.arm == "ridge_lopo")].pixel_r_z.mean()
        if sv > sr: spatial_lead_V += 1
    counts = dict(order_lead_virtues=f"{order_lead_V}/6", mae_ridge_smaller=f"{mae_lead_R}/6",
                  spatial_pixelr_lead_virtues=f"{spatial_lead_V}/6 (secondary, window-level)",
                  bootstrap="B=10000, seed=20260920, true BCa (z0+jackknife a), percentile fallback disclosed")
    print(json.dumps(counts, indent=1, ensure_ascii=False), flush=True)
    with open(r"<project-root>\outputs\e2\s4_v2\summary_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=1, ensure_ascii=False)
    # D2 appended sensitivity
    print("--- D2 appended-core sensitivity (fixed predictions, relabelled) ---", flush=True)
    d2 = df[df.branch == "appended_D2"]
    for t in TARGETS:
        sub28 = pd.concat([prim[prim.target == t], d2[d2.target == t]])
        agg = sub28.groupby(["core_id", "arm"]).agg(meas=("meas", "mean"), pred=("pred", "mean")).reset_index()
        pv = agg[agg.arm == "virtues"].set_index("core_id"); pr = agg[agg.arm == "ridge_lopo"].set_index("core_id")
        cores = sorted(set(pv.index) & set(pr.index))
        rhoV = spearmanr(pv.loc[cores, "meas"], pv.loc[cores, "pred"]).statistic
        rhoR = spearmanr(pr.loc[cores, "meas"], pr.loc[cores, "pred"]).statistic
        print(f"{t}: n={len(cores)} rhoV={rhoV:.4f} rhoR={rhoR:.4f} d={rhoV-rhoR:+.4f}", flush=True)
    print("written:", OUT, flush=True)


if __name__ == "__main__":
    main()
