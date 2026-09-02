# -*- coding: utf-8 -*-
"""模块B 阶段4：组装最终交付物
cell_level_task3.parquet / cluster_assignment_primary.parquet / run_manifest.json
"""
import os, json, hashlib, datetime
import numpy as np
import pandas as pd

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")

cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
assign = pd.read_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"))
prof = pd.read_csv(os.path.join(OUT, "cluster_profile_summary.csv"))
cd31_rank = prof.cd31_mean_uop.rank(ascending=False, method="min").astype(int)
cand = prof.cluster[(prof.vessel_pct_uop >= 0.50) & (cd31_rank <= 5) & (prof.patients_uop >= 22)].tolist()

cells = cells.merge(assign[["cell_id", "cluster_primary"]], on="cell_id", how="left")
cells = cells.rename(columns={"cluster_primary": "token_cluster"})
cells["token_cluster"] = cells.token_cluster.astype("int32")
cells["is_candidate_cluster"] = cells.token_cluster.isin(cand)
final_cols = ["cell_id", "patient_id", "cohort", "ROI_id", "ObjectNumber", "x", "y", "token_cluster",
              "is_candidate_cluster", "source_cell_label", "vessel_label",
              "Ki67", "CD31", "Podoplanin", "panCK", "tumour_border_distance",
              "location", "grade", "analysis_run_id"]
cells[final_cols].to_parquet(os.path.join(OUT, "cell_level_task3.parquet"), index=False)

pa = assign[["cell_id", "cohort", "cluster_primary"]].rename(columns={"cluster_primary": "token_cluster"})
pa["analysis_run_id"] = "task3-v2-20260901"
pa.to_parquet(os.path.join(OUT, "cluster_assignment_primary.parquet"), index=False)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()

files = ["cells_base.parquet", "cell_level_task3.parquet", "cluster_assignment_primary.parquet",
         "cluster_assignments_all_seeds.parquet", "centroids_primary.npy",
         "cluster_stability.csv", "cluster_evaluation.csv", "cluster_enrichment.csv",
         "cluster_profile_summary.csv", "candidate_selection_locked.csv",
         "endothelial_cell_analysis.csv", "ki67_patient_paired.csv", "pdpn_patient_paired_uop.csv",
         "mixed_model_sensitivity.csv", "threshold_sensitivity.csv", "b9_recomputed_numbers.json",
         "patient_level_cluster_summary.csv", "patient_level_cluster_summary.parquet",
         "outcome_analysis.csv", "marker_availability_by_cohort.csv", "sanity_report.json",
         "b8_decision.json", "stage2_run_config.json"]

manifest = {
    "run_id": "task3-v2-20260901",
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "task": "模块B Task-3细胞级重分析（分析规范B1-B10）",
    "inputs": {
        "tokens": "outputs/kao1_celltyping/tokens/*.npz（142 ROI, 498,238细胞, 512维, 冻结virtues-sp32产出; 权重SHA-256见moduleA/checkpoint_sha256.txt）",
        "markers_clinical": "Einhaus UOP/STA allcells.csv（作者发布值, 原样）",
        "coordinates": "Steinbock/regionprops/*.csv（x=centroid-1, y=centroid-0; 0%缺失）",
        "outcome": "STACohort/Metadata/STA_metadata.xlsx（3年复发二分类; 无时间/删失信息）",
    },
    "protocol_pre_registered": {
        "clustering": "MiniBatchKMeans K=20 n_init=10 batch=8192; 发现=UOP全量(273,408); 主run=seed0; 种子0-9",
        "clustering_inputs_strictly_tokens_only": True,
        "stability_threshold_ari": 0.70,
        "sta_assignment": "最近质心(frozen UOP centroids, KMeans.predict)",
        "candidate_criteria": "UOP only: vessel_pct>=0.50 AND CD31均值排名<=5 AND 患者覆盖>=22/24; Ki-67不参与",
        "ki67_primary": "患者级配对差值(候选cluster内vessel细胞中位Ki67 - 其他cluster内vessel细胞中位Ki67); Wilcoxon; 患者bootstrap B=1000",
        "decision_rule": "分析规范B8（脚本b_stage3_endothelial_outcome.py内置）",
    },
    "stop_conditions_encountered": [
        "cluster跨种子不稳定: ARI vs primary 0.381-0.428(中位0.407) < 锁定规范阈值0.70 → 触发分析规范停止条件; 生物学rediscovery结论已停止, 不自行补救"
    ],
    "b8_decision": json.load(open(os.path.join(OUT, "b8_decision.json"), encoding="utf-8"))["decision"],
    "deliverables": {f: {"sha256": sha(os.path.join(OUT, f)),
                         "size_bytes": os.path.getsize(os.path.join(OUT, f))} for f in files
                     if os.path.exists(os.path.join(OUT, f))},
    "scripts": {
        "b_stage1_base.py": "底表构建(B3)",
        "b_stage2_cluster_eval.py": "聚类+稳定性+外部评价(B4/B5/B6); 注: cluster_profile_summary.csv的n_uop列由b_fix_profile.py修复(stale变量bug, 其余列已校验一致)",
        "b_fix_profile.py": "profile表修复与校验",
        "b_stage3_endothelial_outcome.py": "内皮+阈值+预后(B7/B8/B9/B10); 注: 预后患者ID零填充bug由b_fix_outcome.py修复重算(n=23,9事件)",
        "b_fix_outcome.py": "B10预后修复重算",
        "b_stage3b_diagnostics.py": "补充方法学诊断(非锁定规范, 不改B8决策)",
    },
}
json.dump(manifest, open(os.path.join(OUT, "run_manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("候选clusters:", cand)
print("cell_level_task3.parquet:", len(cells), "行")
print("run_manifest.json 完成")
