# -*- coding: utf-8 -*-
"""
模块B 阶段3b：补充方法学诊断（非锁定规范主分析，不改B8决策，仅供作者解读不稳定的来源）
诊断1: vessel轴跨种子一致性 —— 每个MiniBatchKMeans种子下是否存在vessel主导cluster
        (指标: max vessel_pct / vessel_pct>=0.5的cluster数 / 这些cluster覆盖的UOP vessel细胞比例)
诊断2: 全量KMeans(非MiniBatch)敏感性 —— 同K=20, seeds 0-4, 成对ARI
        区分[数据无稳定K=20结构] vs [MiniBatch小批量随机性]
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

ROOT = r"<project-root>"
TOK = os.path.join(ROOT, "outputs", "kao1_celltyping", "tokens")
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")

cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
assign = pd.read_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"))
vessel_uop = cells.loc[cells.cohort == "UOP", "vessel_label"].values.astype(bool)

# ---- 诊断1: vessel轴跨种子(MiniBatchKMeans, 10种子已有assignments) ----
diag1 = []
n_vessel_uop = int(vessel_uop.sum())
for s in range(10):
    lab = assign[f"seed_{s}"].values
    lab_uop = lab[cells.cohort.values == "UOP"]
    dfv = pd.DataFrame({"cl": lab_uop, "v": vessel_uop})
    vp = dfv.groupby("cl").v.mean()
    big = vp[vp >= 0.5]
    cover = float(dfv.loc[dfv.cl.isin(big.index) & dfv.v, "v"].sum() / n_vessel_uop) if len(big) else 0.0
    diag1.append({"seed": s, "max_vessel_pct": float(vp.max()),
                  "n_clusters_vessel_pct_ge_0.5": int(len(big)),
                  "clusters_vessel_dominated": sorted(int(i) for i in big.index),
                  "frac_uop_vessel_cells_in_those_clusters": cover})
d1 = pd.DataFrame(diag1)
d1.to_csv(os.path.join(OUT, "diag_vessel_axis_seed_consistency.csv"), index=False, encoding="utf-8-sig")

# ---- 诊断2: 全量KMeans敏感性 ----
feats, order = [], []
for fp in sorted(glob.glob(os.path.join(TOK, "*.npz"))):
    z = np.load(fp, allow_pickle=True)
    sid = os.path.basename(fp)[:-4].split("_", 1)[1]
    feats.append(z["tokens"])
    order.append(pd.DataFrame({"ROI_id": sid, "ObjectNumber": z["ids"].astype(np.int64)}))
X_all = np.concatenate(feats).astype(np.float32)
oo = pd.concat(order, ignore_index=True)
m = cells.merge(oo.assign(ord2=np.arange(len(oo))), on=["ROI_id", "ObjectNumber"], how="left")
X = X_all[m.ord2.values.astype(np.int64)]
uop_mask = m.cohort.values == "UOP"
Xu = X[uop_mask]

diag2 = []
labs = {}
for s in range(5):
    km = KMeans(n_clusters=20, random_state=s, n_init=10, algorithm="lloyd")
    km.fit(Xu)
    labs[s] = km.labels_
    dfv = pd.DataFrame({"cl": km.labels_, "v": vessel_uop})
    vp = dfv.groupby("cl").v.mean()
    big = vp[vp >= 0.5]
    diag2.append({"seed": s, "inertia": float(km.inertia_),
                  "max_vessel_pct": float(vp.max()), "n_clusters_vessel_pct_ge_0.5": int(len(big)),
                  "clusters_vessel_dominated": sorted(int(i) for i in big.index)})
    print("KMeans seed", s, "done")
for s in range(1, 5):
    diag2.append({"seed_pair": f"{s}_vs_0",
                  "ari": float(adjusted_rand_score(labs[0], labs[s]))})
d2 = pd.DataFrame(diag2)
d2.to_csv(os.path.join(OUT, "diag_full_kmeans_sensitivity.csv"), index=False, encoding="utf-8-sig")

summary = {
    "note": "补充诊断, 非锁定规范主分析; B8决策以cluster_stability.csv(锁定规范MiniBatchKMeans)为准",
    "diag1_vessel_axis_consistent": bool(d1["n_clusters_vessel_pct_ge_0.5"].min() >= 1),
    "diag1_min_max_vessel_pct": float(d1.max_vessel_pct.min()),
    "diag1_vessel_dominated_cluster_counts": d1["n_clusters_vessel_pct_ge_0.5"].tolist(),
    "diag2_full_kmeans_pairwise_ari_vs_seed0": [float(adjusted_rand_score(labs[0], labs[s])) for s in range(1, 5)],
    "diag2_interpretation_hint": "若全量KMeans的ARI仍低→token空间在K=20下确无稳定划分; 若显著升高→MiniBatch随机性主导",
}
json.dump(summary, open(os.path.join(OUT, "diag_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print(d1.to_string(index=False))
print(d2.to_string(index=False))
