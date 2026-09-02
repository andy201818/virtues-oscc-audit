# -*- coding: utf-8 -*-
"""
模块B 阶段2（分析规范B4/B5/B6）：
B4 聚类只用冻结VirTues tokens（不含label/vessel/任何marker/预后）。
    算法 MiniBatchKMeans（与原管线同族）; K=20（先验沿用原协议）; n_init=10; batch=8192; seeds=0..9
    发现队列=UOP（273,408细胞, 全量拟合）; 主run=seed0
B5 聚类后外部评价: ARI/AMI/NMI/加权与均衡purity + 患者级bootstrap CI(B=500)
    + cluster×cell-type富集 + 每cluster患者/ROI覆盖
B6 STA验证: 固定UOP centroids, 最近质心分配(=KMeans.predict); STA标签仅用于事后评价
    pooled两队列合并结果仅作 exploratory 对照, 不得称独立验证
输出: cluster_assignments_all_seeds.parquet / centroids_primary.npy / cluster_stability.csv
      cluster_evaluation.csv / cluster_enrichment.csv / cluster_profile_summary.csv
"""
import os, glob, json, time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom

ROOT = r"<project-root>"
TOK = os.path.join(ROOT, "outputs", "kao1_celltyping", "tokens")
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
RUN_ID = "task3-v2-20260901"
K, SEEDS, N_INIT, BATCH, N_BOOT = 20, list(range(10)), 10, 8192, 500

# ---------- 载入 ----------
cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
feats, ids = [], []
for fp in sorted(glob.glob(os.path.join(TOK, "*.npz"))):
    z = np.load(fp, allow_pickle=True)
    base = os.path.basename(fp)[:-4]
    sid = base.split("_", 1)[1]
    feats.append(z["tokens"])
    ids.append(pd.DataFrame({"ROI_id": sid, "ObjectNumber": z["ids"].astype(np.int64)}))
tok_order = pd.concat(ids, ignore_index=True)
X_all = np.concatenate(feats).astype(np.float32)
assert len(tok_order) == len(cells)
# 对齐顺序（cells_base即按同一路径顺序构建, 校验键完全一致）
key1 = cells["_token_file"] if "_token_file" in cells else None
m = cells.merge(tok_order.assign(ord2=np.arange(len(tok_order))),
                on=["ROI_id", "ObjectNumber"], how="left")
assert m.ord2.notna().all()
ordidx = m.ord2.values.astype(np.int64)
X = X_all[ordidx]                      # 与cells行序对齐
labels = cells.source_cell_label.values
cohort = cells.cohort.values
patient = cells.patient_id.values
roi = cells.ROI_id.values
print("cells:", X.shape)

uop = cohort == "UOP"
sta = ~uop

# ---------- B4: UOP发现聚类, 多种子 ----------
t0 = time.time()
assign = {}
for s in SEEDS:
    km = MiniBatchKMeans(n_clusters=K, random_state=s, batch_size=BATCH, n_init=N_INIT)
    km.fit(X[uop])
    assign[s] = {"uop": km.labels_, "sta": km.predict(X[sta]), "centers": km.cluster_centers_}
    print(f"seed {s} fitted ({time.time()-t0:.0f}s)")
PRIMARY = 0
cen = assign[PRIMARY]["centers"]
np.save(os.path.join(OUT, "centroids_primary.npy"), cen)

prim_uop = assign[PRIMARY]["uop"]
prim_sta = assign[PRIMARY]["sta"]
clu_primary = np.empty(len(cells), dtype=np.int32)
clu_primary[uop] = prim_uop
clu_primary[sta] = prim_sta

# 全种子assignments表（每种子UOP+STA完整列）
seed_cols = {}
for s in SEEDS:
    full = np.empty(len(cells), dtype=np.int32)
    full[uop] = assign[s]["uop"]
    full[sta] = assign[s]["sta"]
    seed_cols[f"seed_{s}"] = full
assign_df = pd.DataFrame({"cell_id": cells.cell_id, "cohort": cohort, **seed_cols})
assign_df["cluster_primary"] = clu_primary
assign_df.to_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"), index=False)

# ---------- B4: 稳定性 ----------
def hungarian_map(a, b, K=K):
    """返回b->a的最优匹配及整体一致率"""
    C = np.zeros((K, K), dtype=np.int64)
    for i, j in zip(b, a):
        C[i, j] += 1
    r, c = linear_sum_assignment(-C)
    mapping = {int(r[t]): int(c[t]) for t in range(len(r))}
    mapped = np.array([mapping.get(int(v), -1) for v in b])
    agree = float((mapped == a).mean())
    return mapping, agree

stab_rows = []
for s in SEEDS:
    b = assign[s]["uop"]
    if s == PRIMARY:
        mapping, agree = {i: i for i in range(K)}, 1.0
        ari = 1.0
    else:
        mapping, agree = hungarian_map(prim_uop, b)
        ari = float(adjusted_rand_score(prim_uop, b))
    # 匹配后每primary-cluster的患者覆盖与单一患者/ROI占比（UOP）
    mapped = np.array([mapping.get(int(v), -1) for v in b])
    covers, maxpat, maxroi = [], [], []
    for k in range(K):
        sel = mapped == k
        if sel.sum() == 0:
            covers.append(0.0); maxpat.append(np.nan); maxroi.append(np.nan); continue
        pats = pd.Series(patient[uop][sel]).value_counts()
        rois = pd.Series(roi[uop][sel]).value_counts()
        covers.append(pats.size / 24.0)
        maxpat.append(pats.iloc[0] / sel.sum())
        maxroi.append(rois.iloc[0] / sel.sum())
    stab_rows.append({"seed": s, "ari_vs_primary": ari, "hungarian_agreement": agree,
                      "mean_patient_coverage_per_cluster": float(np.mean(covers)),
                      "max_single_patient_share_median": float(np.nanmedian(maxpat)),
                      "max_single_roi_share_median": float(np.nanmedian(maxroi))})
stab = pd.DataFrame(stab_rows)
stab.to_csv(os.path.join(OUT, "cluster_stability.csv"), index=False, encoding="utf-8-sig")

# 主run的cluster级构成（发现队列UOP; 用于B7候选选择——只用vessel/CD31/覆盖度）
prof = []
for k in range(K):
    g = cells.loc[uop].iloc[np.where(prim_uop == k)[0]]
    prof.append({
        "cluster": k, "n_uop": int(sel.sum()),
        "patients_uop": g.patient_id.nunique(), "rois_uop": g.ROI_id.nunique(),
        "patient_coverage_uop": g.patient_id.nunique() / 24.0,
        "vessel_pct_uop": float(g.vessel_label.mean()),
        "cd31_mean_uop": float(g.CD31.mean()),
        "ki67_mean_uop": float(g.Ki67.mean()),           # 记录但【不得】用于候选选择
        "pdpn_mean_uop": float(g.Podoplanin.mean()),
        "panck_mean_uop": float(g.panCK.mean()),
    })
# STA侧（验证队列, 事后）
for row in prof:
    k = row["cluster"]
    g = cells.loc[sta].iloc[np.where(prim_sta == k)[0]]
    row.update({"n_sta": len(g), "patients_sta": g.patient_id.nunique(), "rois_sta": g.ROI_id.nunique(),
                "vessel_pct_sta": float(g.vessel_label.mean()), "cd31_mean_sta": float(g.CD31.mean()),
                "ki67_mean_sta": float(g.Ki67.mean()), "pdpn_mean_sta": np.nan})
prof = pd.DataFrame(prof)
prof.to_csv(os.path.join(OUT, "cluster_profile_summary.csv"), index=False, encoding="utf-8-sig")

# ---------- B5: 外部评价 ----------
def purities(true, pred, K=K):
    df = pd.crosstab(pred, true)
    pk = df.max(axis=1) / df.sum(axis=1)
    weighted = float((df.max(axis=1)).sum() / df.values.sum())
    balanced = float(pk.mean())
    return weighted, balanced

def metrics(true, pred):
    ari = adjusted_rand_score(true, pred)
    ami = adjusted_mutual_info_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    w, b = purities(true, pred)
    return {"ARI": ari, "AMI": ami, "NMI": nmi, "purity_weighted": w, "purity_balanced": b}

eval_rows = []
for setting, mask in [("UOP_discovery", uop), ("STA_validation", sta),
                      ("pooled_exploratory", np.ones(len(cells), bool))]:
    t, p = labels[mask], clu_primary[mask]
    m0 = metrics(t, p)
    # 患者级bootstrap
    pats = np.unique(patient[mask])
    idx_by_pat = {pp: np.where(patient[mask] == pp)[0] for pp in pats}
    rng = np.random.RandomState(20260901)
    boots = {k: [] for k in m0}
    for _ in range(N_BOOT):
        pick = rng.choice(pats, size=len(pats), replace=True)
        idx = np.concatenate([idx_by_pat[pp] for pp in pick])
        mb = metrics(t[idx], p[idx])
        for k in boots:
            boots[k].append(mb[k])
    row = {"setting": setting, "n_cells": int(mask.sum()), "n_patients": len(pats), **m0}
    for k, v in boots.items():
        row[k + "_boot_lo"] = float(np.percentile(v, 2.5))
        row[k + "_boot_hi"] = float(np.percentile(v, 97.5))
    eval_rows.append(row)
    print(setting, {k: round(v, 4) for k, v in m0.items()}, f"({time.time()-t0:.0f}s)")
ev = pd.DataFrame(eval_rows)
ev["bootstrap"] = f"patient-level, B={N_BOOT}"
ev.to_csv(os.path.join(OUT, "cluster_evaluation.csv"), index=False, encoding="utf-8-sig")

# ---------- B5: cluster×cell-type富集（各队列, 描述性+超几何P注明非独立） ----------
enr_rows = []
for coh_name, mask in [("UOP", uop), ("STA", sta)]:
    t, p = labels[mask], clu_primary[mask]
    N = len(t)
    df = pd.crosstab(p, t)
    for ctype in df.columns:
        Nc = int(df[ctype].sum())
        for k in df.index:
            n_k = int(df.loc[k].sum()); n_ck = int(df.loc[k, ctype])
            exp = n_k * Nc / N
            pv = hypergeom.sf(n_ck - 1, N, Nc, n_k)
            enr_rows.append({"cohort": coh_name, "cluster": int(k), "cell_type": ctype,
                             "n_cells": n_ck, "n_cluster": n_k, "n_type": Nc,
                             "expected": exp, "ratio_obs_over_exp": n_ck / exp if exp else np.nan,
                             "log2_enrichment": float(np.log2(n_ck / exp)) if exp else np.nan,
                             "hypergeom_p": pv,
                             "note": "P值为描述性参考: 细胞非独立(患者/ROI聚类), 推断以患者级bootstrap为准"})
pd.DataFrame(enr_rows).to_csv(os.path.join(OUT, "cluster_enrichment.csv"),
                              index=False, encoding="utf-8-sig")

json.dump({"run_id": RUN_ID, "K": K, "algorithm": "MiniBatchKMeans", "seeds": SEEDS,
           "n_init": N_INIT, "batch_size": BATCH, "primary_seed": PRIMARY,
           "discovery": "UOP (all 273,408 cells, full fit)",
           "validation": "STA nearest-centroid assignment (KMeans.predict on frozen UOP centroids)",
           "n_boot": N_BOOT,
           "stability_ari_vs_primary": dict(zip(stab.seed, stab.ari_vs_primary.round(4)))},
          open(os.path.join(OUT, "stage2_run_config.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("stage2 done", f"({time.time()-t0:.0f}s)")
print(stab.to_string(index=False))
print(ev.to_string(index=False))
