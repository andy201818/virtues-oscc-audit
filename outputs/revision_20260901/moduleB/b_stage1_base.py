# -*- coding: utf-8 -*-
"""
模块B 阶段1（分析规范B3）：构建Task-3细胞级底表
- 数据源: outputs/kao1_celltyping/tokens/*.npz（冻结VirTues cell summary tokens, 512维, 已校验权重产出）
          Einhaus UOP/STA allcells.csv（marker+临床+位置摘要）
          Steinbock/regionprops/*.csv（细胞质心坐标; x=centroid-1, y=centroid-0）
- 缺失显式标记(NA/NaN)，不以0代替；STA无Podoplanin
- 输出: cells_base.parquet（暂无cluster列，B4后回填）, marker_availability_by_cohort.csv, sanity_report.json
锁定规范口径:
- marker值 = Einhaus作者发布值（其管线已做变换），本分析作为连续变量使用，不再二次变换
- tumour_border_distance = 作者字段 celldistance（带符号距离, 原样保留）
"""
import os, glob, json
import numpy as np
import pandas as pd

ROOT = r"<project-root>"
E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")
TOK = os.path.join(ROOT, "outputs", "kao1_celltyping", "tokens")
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
RUN_ID = "task3-v2-20260901"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 读token与注释 ----------
rows, tok_path = [], sorted(glob.glob(os.path.join(TOK, "*.npz")))
token_files = []
for fp in tok_path:
    base = os.path.basename(fp)[:-4]           # e.g. STACohort_S01_1 / UOPCohort_OC01_001
    cohort_long, sid = base.split("_", 1)
    cohort = "UOP" if "UOP" in cohort_long else "STA"
    z = np.load(fp, allow_pickle=True)
    n = len(z["ids"])
    token_files.append((cohort, sid, n))
    rows.append(pd.DataFrame({
        "cohort": cohort, "ROI_id": sid, "ObjectNumber": z["ids"].astype(np.int64),
        "source_cell_label": z["labels"].astype(str),
        "_token_file": os.path.basename(fp),
    }))
cells = pd.concat(rows, ignore_index=True)
cells["vessel_label"] = (cells["source_cell_label"] == "Vessel")
cells["patient_id"] = np.where(cells["cohort"] == "UOP",
                               cells["ROI_id"].str.extract(r"^(OC\d+)", expand=False),
                               cells["ROI_id"].str.extract(r"^(S\d+)", expand=False))
cells["cell_id"] = cells["ROI_id"] + ":" + cells["ObjectNumber"].astype(str)
print("token汇总:", len(cells), "细胞 | UOP", (cells.cohort == "UOP").sum(), "| STA", (cells.cohort == "STA").sum())

# ---------- 2. 合并marker/临床（作者发布值, 原样） ----------
mk_cols = ["sample_id", "ObjectNumber", "patient_id", "grade", "location", "celldistance",
           "Ki67", "CD31", "Podoplanin", "Pancytokeratin"]
parts = []
for cohort, pref in [("UOP", "UOP"), ("STA", "STA")]:
    dfm = pd.read_csv(os.path.join(E, f"{cohort}Cohort", "Dataframes", f"{pref}_allcells.csv"))
    have = [c for c in mk_cols if c in dfm.columns]
    dfm = dfm[have]
    if "Podoplanin" not in have:
        dfm["Podoplanin"] = np.nan          # STA面板无PDPN → 显式NA
    parts.append(dfm)
mk = pd.concat(parts, ignore_index=True).drop_duplicates(["sample_id", "ObjectNumber"])
mk = mk.rename(columns={"celldistance": "tumour_border_distance", "Pancytokeratin": "panCK"})
# ROI_id ↔ sample_id 对齐: token文件的sid即allcells的sample_id
cells = cells.merge(mk, left_on=["ROI_id", "ObjectNumber"], right_on=["sample_id", "ObjectNumber"],
                    how="left", suffixes=("", "_mk"))
for c in ["Ki67", "CD31", "Podoplanin", "panCK", "tumour_border_distance"]:
    cells[c] = pd.to_numeric(cells[c], errors="coerce")   # 真缺失→NaN，不填0

# ---------- 3. 合并质心坐标 ----------
crd = []
for cohort_long, pref in [("UOPCohort", "UOP"), ("STACohort", "STA")]:
    d = os.path.join(E, cohort_long, "Steinbock", "regionprops")
    for fp in sorted(glob.glob(os.path.join(d, "*.csv"))):
        sid = os.path.basename(fp)[:-4]
        r = pd.read_csv(fp)[["Object", "centroid-0", "centroid-1"]]
        r = r.rename(columns={"Object": "ObjectNumber", "centroid-0": "y", "centroid-1": "x"})
        r["ROI_id"] = sid
        crd.append(r)
coords = pd.concat(crd, ignore_index=True).drop_duplicates(["ROI_id", "ObjectNumber"])
cells = cells.merge(coords, on=["ROI_id", "ObjectNumber"], how="left")

# ---------- 4. 字段整理与顺序 ----------
cells["analysis_run_id"] = RUN_ID
cells = cells[["cell_id", "patient_id", "cohort", "ROI_id", "x", "y",
               "source_cell_label", "vessel_label", "Ki67", "CD31", "Podoplanin", "panCK",
               "tumour_border_distance", "location", "grade", "_token_file", "ObjectNumber",
               "analysis_run_id"]]
cells.to_parquet(os.path.join(OUT, "cells_base.parquet"), index=False)

# ---------- 5. marker可得性 ----------
avail = []
for c in ["Ki67", "CD31", "Podoplanin", "panCK"]:
    for coh in ["UOP", "STA"]:
        s = cells.loc[cells.cohort == coh, c]
        avail.append({"marker": c, "cohort": coh, "n_cells": len(s),
                      "n_nonmissing": int(s.notna().sum()), "fraction_nonmissing": float(s.notna().mean()),
                      "availability": "measured" if s.notna().any() else "absent-from-panel"})
pd.DataFrame(avail).to_csv(os.path.join(OUT, "marker_availability_by_cohort.csv"),
                           index=False, encoding="utf-8-sig")

# ---------- 6. sanity ----------
san = {}
san["n_cells"] = len(cells)
san["by_cohort"] = cells.cohort.value_counts().to_dict()
san["patients_per_cohort"] = cells.groupby("cohort").patient_id.nunique().to_dict()
san["roi_per_cohort"] = cells.groupby("cohort").ROI_id.nunique().to_dict()
san["label_domain"] = cells.source_cell_label.value_counts().to_dict()
san["missing_rate"] = {c: float(cells[c].isna().mean()) for c in
                       ["x", "y", "Ki67", "CD31", "Podoplanin", "panCK", "tumour_border_distance", "grade"]}
san["patient_id_null"] = int(cells.patient_id.isna().sum())
# 标签一致性: npz labels vs allcells celltypes（同键应一致; 这里检查merge完备性）
mm = cells.merge(
    pd.concat(parts, ignore_index=True)[["sample_id", "ObjectNumber"]].assign(in_mk=True),
    left_on=["ROI_id", "ObjectNumber"], right_on=["sample_id", "ObjectNumber"], how="left")
san["marker_merge_hit_rate"] = float(mm.in_mk.notna().mean())
# celldistance符号与location的一致性（描述性, 不做解释性使用）
sub = cells.dropna(subset=["tumour_border_distance"])
san["celldistance_sign_by_location"] = {
    loc: {"frac_negative": float((g < 0).mean()), "median": float(g.median()), "n": len(g)}
    for loc, g in sub.groupby("location").tumour_border_distance}
# Ki67值域（作者发布值, 供阈值定义参考）
san["ki67_quantiles_pooled"] = {q: float(cells.Ki67.quantile(float(q))) for q in
                                ["0.25", "0.5", "0.75", "0.8", "0.9", "0.95"]}
san["ki67_quantiles_by_cohort"] = {coh: {q: float(g.quantile(float(q))) for q in ["0.5", "0.75", "0.9"]}
                                   for coh, g in cells.groupby("cohort").Ki67}
json.dump(san, open(os.path.join(OUT, "sanity_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(san, ensure_ascii=False, indent=1)[:1800])
print("stage1 done ->", OUT)
