# -*- coding: utf-8 -*-
"""修复 cluster_profile_summary.csv 的 n_uop 列（stage2 脚本 stale-sel bug；其余列已正确）"""
import os
import numpy as np
import pandas as pd

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
assign = pd.read_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"))
prof = pd.read_csv(os.path.join(OUT, "cluster_profile_summary.csv"))
cells["cluster"] = assign.set_index("cell_id").loc[cells.cell_id, "cluster_primary"].values

rows = []
for k in range(20):
    gu = cells[(cells.cohort == "UOP") & (cells.cluster == k)]
    gs = cells[(cells.cohort == "STA") & (cells.cluster == k)]
    rows.append({
        "cluster": k,
        "n_uop": len(gu), "patients_uop": gu.patient_id.nunique(), "rois_uop": gu.ROI_id.nunique(),
        "patient_coverage_uop": gu.patient_id.nunique() / 24.0,
        "vessel_pct_uop": float(gu.vessel_label.mean()), "cd31_mean_uop": float(gu.CD31.mean()),
        "ki67_mean_uop": float(gu.Ki67.mean()), "pdpn_mean_uop": float(gu.Podoplanin.mean()),
        "panck_mean_uop": float(gu.panCK.mean()),
        "n_sta": len(gs), "patients_sta": gs.patient_id.nunique(), "rois_sta": gs.ROI_id.nunique(),
        "vessel_pct_sta": float(gs.vessel_label.mean()), "cd31_mean_sta": float(gs.CD31.mean()),
        "ki67_mean_sta": float(gs.Ki67.mean()), "pdpn_mean_sta": np.nan,
    })
new = pd.DataFrame(rows)
# 与旧表逐列核对（n_uop 应当不同, 其余应一致）
for c in new.columns:
    if c == "n_uop":
        continue
    if c in prof.columns:
        a, b = new[c].values, prof[c].values
        if pd.api.types.is_numeric_dtype(new[c]):
            same = np.allclose(a.astype(float), b.astype(float), atol=1e-9, equal_nan=True)
        else:
            same = (a == b).all()
        print(f"{c}: {'一致' if same else '不一致!!'}")
new.to_csv(os.path.join(OUT, "cluster_profile_summary.csv"), index=False, encoding="utf-8-sig")
print("n_uop 合计(应=273408):", new.n_uop.sum(), "| n_sta 合计(应=224830):", new.n_sta.sum())
print(new[["cluster","n_uop","patients_uop","vessel_pct_uop","cd31_mean_uop","n_sta","patients_sta","vessel_pct_sta"]].round(4).to_string(index=False))
