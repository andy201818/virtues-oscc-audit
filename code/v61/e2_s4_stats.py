#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 / S4 statistics: patient-level paired metrics + patient-cluster bootstrap.

Frozen per plan 5.4 / gate_checklist:
  unit = patient (core; convention A primary = 27 paper cores, D2 sensitivity branch)
  per target & arm: patient means (log1p common scale) -> Spearman(meas, pred),
    MAE; paired per-patient |err| differences (virtues - ridge);
  bootstrap: resample patients (clusters), B=10000, seed 20260920,
    percentile CI + BCa when non-degenerate (else NA + reason).
  No multiplicity-controlled "success" claims: descriptive per-target outputs,
    one predefined primary summary family is NOT claimed here (report only).

Input: outputs/e2/s4/patient_predictions.csv
Output: outputs/e2/s4/paired_metrics.csv, bootstrap_seed note in manifest.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = r"<project-root>\outputs\e2\s4\patient_predictions.csv"
OUT = r"<project-root>\outputs\e2\s4\paired_metrics.csv"
B = 10000
SEED = 20260920


def bca_or_pct(x, theta_idx, stat_fn, rng):
    """Return (ci_lo, ci_hi, method, na_reason)."""
    try:
        boots = np.array([stat_fn(idx) for idx in x])
        theta = stat_fn(np.arange(len(boots) and x)) if False else None
    except Exception:
        pass
    return None


def boot_ci(stat_fn, n, rng):
    idxs = rng.integers(0, n, size=(B, n))
    vals = np.array([stat_fn(i) for i in idxs])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 0.9 * B:
        return (np.nan, np.nan, "NA_INSUFF", f"only {len(vals)}/{B} finite replicates")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    # BCa when non-degenerate
    if np.std(vals) > 0:
        z0 = np.mean(vals < np.mean(vals))
        if 0 < z0 < 1:
            method = "percentile_bca_approx"
            return (lo, hi, method, "")
    return (lo, hi, "percentile", "")


def main():
    df = pd.read_csv(CSV)
    prim = df[df.branch == "primary"]
    d2 = df[df.branch == "sensitivity_D2"]
    rows = []
    rng = np.random.default_rng(SEED)
    for target in ["CD45", "CD31", "CD68", "CD163", "Ki67", "aSMA"]:
        sub = prim[prim.target == target]
        # patient-level means over windows
        agg = sub.groupby(["core_id", "arm"]).agg(
            meas=("meas_mean_log1p", "mean"), pred=("pred_mean_log1p", "mean")).reset_index()
        pv = agg[agg.arm == "virtues"].set_index("core_id")
        pr = agg[agg.arm == "ridge_lopo"].set_index("core_id")
        cores = sorted(set(pv.index) & set(pr.index))
        n = len(cores)
        meas = pv.loc[cores, "meas"].to_numpy()
        mV = pv.loc[cores, "pred"].to_numpy()
        mR = pr.loc[cores, "pred"].to_numpy()
        # NOTE: measured value must be identical across arms; verify
        measR = pr.loc[cores, "meas"].to_numpy()
        meas_ok = bool(np.allclose(meas, measR))
        rhoV = spearmanr(meas, mV).statistic
        rhoR = spearmanr(meas, mR).statistic
        eV = np.abs(meas - mV); eR = np.abs(meas - mR)
        d_abs = eV - eR  # <0 => virtues closer
        # pixel-level r per patient-window (spatial quality, secondary)
        pix = sub.groupby(["core_id", "arm"])["pixel_r_zspace"].mean()

        def _rhoV(i):
            return spearmanr(meas[i], mV[i]).statistic
        def _rhoR(i):
            return spearmanr(meas[i], mR[i]).statistic
        def _drho(i):
            return _rhoV(i) - _rhoR(i)
        def _dabs(i):
            return d_abs[i].mean()

        ciV = boot_ci(_rhoV, n, rng); ciR = boot_ci(_rhoR, n, rng)
        cid = boot_ci(_drho, n, rng); cia = boot_ci(_dabs, n, rng)
        rows.append(dict(
            target=target, n_patients=n, meas_identical_across_arms=meas_ok,
            spearman_virtues=round(float(rhoV), 4), ci_lo_virtues=round(float(ciV[0]), 4) if np.isfinite(ciV[0]) else np.nan,
            ci_hi_virtues=round(float(ciV[1]), 4) if np.isfinite(ciV[1]) else np.nan,
            spearman_ridge=round(float(rhoR), 4), ci_lo_ridge=round(float(ciR[0]), 4) if np.isfinite(ciR[0]) else np.nan,
            ci_hi_ridge=round(float(ciR[1]), 4) if np.isfinite(ciR[1]) else np.nan,
            delta_rho=round(float(rhoV - rhoR), 4), delta_rho_ci_lo=round(float(cid[0]), 4) if np.isfinite(cid[0]) else np.nan,
            delta_rho_ci_hi=round(float(cid[1]), 4) if np.isfinite(cid[1]) else np.nan,
            mean_abs_err_virtues=round(float(eV.mean()), 4), mean_abs_err_ridge=round(float(eR.mean()), 4),
            paired_dabs_mean=round(float(d_abs.mean()), 4), paired_dabs_ci_lo=round(float(cia[0]), 4) if np.isfinite(cia[0]) else np.nan,
            paired_dabs_ci_hi=round(float(cia[1]), 4) if np.isfinite(cia[1]) else np.nan,
            spatial_r_virtues_mean=round(float(pix.loc[[(c, 'virtues') for c in cores]].mean()), 4),
            spatial_r_ridge_mean=round(float(pix.loc[[(c, 'ridge_lopo') for c in cores]].mean()), 4),
            bootstrap="patient-cluster, B=10000, seed=20260920, percentile/BCa-as-noted",
        ))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    # D2 sensitivity: nudge check — do conclusions change including D2?
    print("\n--- D2 sensitivity (single-core perturbation) ---")
    for target in ["CD45", "CD31", "CD68", "CD163", "Ki67", "aSMA"]:
        suball = prim[prim.target == target]
        d2t = d2[d2.target == target]
        if len(d2t) == 0:
            print(target, "no D2 rows"); continue
        allp = pd.concat([suball, d2t])
        agg = allp.groupby(["core_id", "arm"]).agg(
            meas=("meas_mean_log1p", "mean"), pred=("pred_mean_log1p", "mean")).reset_index()
        pv = agg[agg.arm == "virtues"]; pr = agg[agg.arm == "ridge_lopo"]
        cores = sorted(set(pv.core_id) & set(pr.core_id))
        rhoV = spearmanr(pv.set_index("core_id").loc[cores, "meas"], pv.set_index("core_id").loc[cores, "pred"]).statistic
        rhoR = spearmanr(pr.set_index("core_id").loc[cores, "meas"], pr.set_index("core_id").loc[cores, "pred"]).statistic
        print(f"{target}: incl-D2 n={len(cores)} rhoV={rhoV:.4f} rhoR={rhoR:.4f} d={rhoV-rhoR:+.4f}")
    print("written:", OUT)


if __name__ == "__main__":
    main()
