# -*- coding: utf-8 -*-
"""修复B10预后的患者ID零填充bug（S01→S1导致S01-S09丢失）并重算outcome_analysis.csv"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")

cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
assign = pd.read_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"))
prof = pd.read_csv(os.path.join(OUT, "cluster_profile_summary.csv"))
cd31_rank = prof.cd31_mean_uop.rank(ascending=False, method="min").astype(int)
cand = prof.cluster[(prof.vessel_pct_uop >= 0.50) & (cd31_rank <= 5) & (prof.patients_uop >= 22)].tolist()
cells["cluster"] = assign.set_index("cell_id").loc[cells.cell_id, "cluster_primary"].values
cells["is_candidate_cluster"] = cells.cluster.isin(cand)

meta = pd.read_excel(os.path.join(E, "STACohort", "Metadata", "STA_metadata.xlsx"))
meta["patient_id"] = meta["ID"].astype(str).str.extract(r"S(\d+)", expand=False).map(
    lambda x: f"S{int(x):02d}" if pd.notna(x) else None)          # <-- 02d零填充修复
rec_col = "Recurrence within 3yrs (0=no, 1=yes)"

sta = cells[cells.cohort == "STA"]
psum = sta.groupby("patient_id").agg(n_cells=("cell_id", "size"),
                                     n_cand=("is_candidate_cluster", "sum")).reset_index()
psum["candidate_cell_fraction"] = psum.n_cand / psum.n_cells
vess = sta[sta.vessel_label]
vsel = (vess.assign(c=vess.is_candidate_cluster).groupby("patient_id").c.mean()
        .rename("vessel_frac_in_candidate").reset_index())

out = meta[["patient_id", rec_col]].merge(psum[["patient_id", "candidate_cell_fraction"]],
                                          on="patient_id", how="left").merge(vsel, on="patient_id", how="left")
print("metadata患者:", len(meta), "| 3年复发非缺失:", out[rec_col].notna().sum(),
      "| 事件数:", out[rec_col].sum())
out = out.dropna(subset=[rec_col, "candidate_cell_fraction"])
out["recur3y"] = out[rec_col].astype(int)

oc_rows = []
for expo, lab in [("candidate_cell_fraction", "candidate-cluster cell fraction (per 1 SD)"),
                  ("vessel_frac_in_candidate", "vessel-cell fraction in candidate clusters (per 1 SD)")]:
    d = out.dropna(subset=[expo]).copy()
    z = (d[expo] - d[expo].mean()) / d[expo].std(ddof=1)
    fit = sm.Logit(d.recur3y, sm.add_constant(z)).fit(disp=0)
    ci = fit.conf_int()
    oc_rows.append({"outcome": "Recurrence within 3yrs (binary)", "n": int(len(d)),
                    "n_events": int(d.recur3y.sum()), "exposure": lab,
                    "OR_per_1SD": float(np.exp(fit.params.iloc[1])),
                    "ci95_lo": float(np.exp(ci.iloc[1, 0])), "ci95_hi": float(np.exp(ci.iloc[1, 1])),
                    "p": float(fit.pvalues.iloc[1]),
                    "interpretation": "exploratory; estimate imprecise/inconclusive"})
pd.DataFrame(oc_rows).to_csv(os.path.join(OUT, "outcome_analysis.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(oc_rows).to_string(index=False))
