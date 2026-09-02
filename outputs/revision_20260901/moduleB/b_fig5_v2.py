# -*- coding: utf-8 -*-
"""
阶段3a：Figure 5 v2 —— Unsupervised token-space phenotype recovery（方法学版, B11规格+Task3降级决策）
A1/A2: UMAP（UOP 3万细胞子样）按token_cluster与source label双着色
B: cluster × cell-type log2富集热图（UOP发现 / STA验证两栏）
C: UOP发现 vs STA验证外部评价指标（ARI/AMI/NMI/purity, 患者bootstrap 95%CI）
D: Ki-67患者级配对斜率图（候选cluster vessel细胞 vs 其他cluster vessel细胞, UOP/STA分面）
E: CD31–Ki-67单细胞联合分布（vessel细胞, 候选/其他着色, 两队列）
F1/F2: 代表性原始图像（CD31/Ki67/panCK合成）与细胞轮廓（按候选/其他vessel/其他细胞）
标注: 患者数/ROI数/细胞数/bootstrap单位/标准化方式/UOP与STA分开
禁止项: 预设增殖标签打分 / normalized-to-max / 与source data不一致的旧比例（详见分析规范B11）
产出: fig5_v2.pdf, fig5_v2.png, figure5_source_data.csv, fig5_source/ 各面板数据
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

ROOT = r"<project-root>"
MB = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
TOK = os.path.join(ROOT, "outputs", "kao1_celltyping", "tokens")
E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")
FIG = os.path.join(MB, "fig5_source")
os.makedirs(FIG, exist_ok=True)

RUN = "task3-v2-20260901"
cells = pd.read_parquet(os.path.join(MB, "cell_level_task3.parquet"))
assign = pd.read_parquet(os.path.join(MB, "cluster_assignments_all_seeds.parquet"))
prof = pd.read_csv(os.path.join(MB, "cluster_profile_summary.csv"))
evalv = pd.read_csv(os.path.join(MB, "cluster_evaluation.csv"))
enr = pd.read_csv(os.path.join(MB, "cluster_enrichment.csv"))
ki_pair = pd.read_csv(os.path.join(MB, "ki67_patient_paired.csv"))
endo = pd.read_csv(os.path.join(MB, "endothelial_cell_analysis.csv"))
cd31_rank = prof.cd31_mean_uop.rank(ascending=False, method="min").astype(int)
cand_clusters = prof.cluster[(prof.vessel_pct_uop >= 0.50) & (cd31_rank <= 5) & (prof.patients_uop >= 22)].tolist()
print("候选:", cand_clusters)

labels8 = ["Tumor", "Fibroblasts", "CD4 T cells", "Myeloid", "CD8 T cells", "Vessel", "other", "B cells"]
CAND_COLOR = "#d62728"

# ---------------- Panel A: UMAP（UOP子样, seed固定；有缓存则复用） ----------------
if os.path.exists(os.path.join(FIG, "panelA_umap.csv")):
    umap_df = pd.read_csv(os.path.join(FIG, "panelA_umap.csv"))
    print("Panel A: 复用缓存UMAP")
else:
    print("Panel A: UMAP ...")
    rng = np.random.RandomState(20260901)
    uop_cells = cells[cells.cohort == "UOP"]
    n_s = min(30000, len(uop_cells))
    sub = uop_cells.sample(n_s, random_state=20260901)
    tok_rows = []
    for fp in sorted(glob.glob(os.path.join(TOK, "*.npz"))):
        base = os.path.basename(fp)[:-4]; sid = base.split("_", 1)[1]
        z = np.load(fp, allow_pickle=True)
        tok_rows.append(pd.DataFrame({"ROI_id": sid, "ObjectNumber": z["ids"].astype(np.int64),
                                      "token_pos": np.arange(len(z["ids"]))}))
    tok_map = pd.concat(tok_rows, ignore_index=True)
    sub = sub.merge(tok_map, on=["ROI_id", "ObjectNumber"], how="left")
    assert sub.token_pos.notna().all()
    feats, order = [], []
    for fp in sorted(glob.glob(os.path.join(TOK, "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        sid = os.path.basename(fp)[:-4].split("_", 1)[1]
        feats.append(z["tokens"]); order.append((sid, len(z["ids"])))
    X_all = np.concatenate(feats)
    pos = []
    i0 = 0
    for sid, n in order:
        pos.append(pd.DataFrame({"ROI_id": sid, "token_pos": np.arange(n), "g": np.arange(i0, i0 + n)}))
        i0 += n
    gmap = pd.concat(pos, ignore_index=True)
    sub = sub.merge(gmap, on=["ROI_id", "token_pos"], how="left")
    Xs = X_all[sub.g.values.astype(np.int64)]
    from umap import UMAP
    emb = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=20260901, low_memory=True).fit_transform(Xs)
    umap_df = pd.DataFrame({"umap1": emb[:, 0], "umap2": emb[:, 1],
                            "cluster": sub.token_cluster.values, "label": sub.source_cell_label.values})
    umap_df.to_csv(os.path.join(FIG, "panelA_umap.csv"), index=False, encoding="utf-8-sig")

# ---------------- Panel B: 富集热图数据 ----------------
enr["is_cand"] = enr.cluster.isin(cand_clusters)
enr.to_csv(os.path.join(FIG, "panelB_enrichment.csv"), index=False, encoding="utf-8-sig")

# ---------------- Panel C: 指标 ----------------
evalv.to_csv(os.path.join(FIG, "panelC_metrics.csv"), index=False, encoding="utf-8-sig")

# ---------------- Panel D: 配对斜率 ----------------
ki_pair.to_csv(os.path.join(FIG, "panelD_ki67_paired.csv"), index=False, encoding="utf-8-sig")

# ---------------- Panel E: CD31-Ki67联合分布 ----------------
endo_s = endo[["cohort", "patient_id", "arm", "Ki67", "CD31"]].copy()
endo_s.to_csv(os.path.join(FIG, "panelE_cd31_ki67.csv"), index=False, encoding="utf-8-sig")

# ---------------- Panel F: 代表性ROI（UOP, 候选vessel细胞最多的ROI） ----------------
vess_cand = cells[(cells.vessel_label) & (cells.is_candidate_cluster) & (cells.cohort == "UOP")]
roi_pick = vess_cand.ROI_id.value_counts().idxmax()
cohort_long = "UOPCohort"
img = None
try:
    import tifffile
    img = tifffile.imread(os.path.join(E, cohort_long, "Steinbock", "img", roi_pick + ".tiff"))
    msk = tifffile.imread(os.path.join(E, cohort_long, "Steinbock", "masks", roi_pick + ".tiff"))
    panel_csv = pd.read_csv(os.path.join(E, cohort_long, "Steinbock", "panel.csv"))
    pn = {n: i for i, n in enumerate(panel_csv.name.tolist())}
    fcells = cells[cells.ROI_id == roi_pick]
    np.savez_compressed(os.path.join(FIG, "panelF_roi.npz"), img=img.astype(np.float32),
                        msk=msk, ObjectNumber=fcells.ObjectNumber.values,
                        cluster=fcells.token_cluster.values, label=fcells.source_cell_label.values,
                        is_cand=fcells.is_candidate_cluster.values, vessel=fcells.vessel_label.values)
    json.dump({"roi": roi_pick, "panel_channels": {k: int(pn.get(k, -1)) for k in ["CD31", "Ki67", "Pancytokeratin"]}},
              open(os.path.join(FIG, "panelF_roi.json"), "w"))
except Exception as ex:
    print("Panel F 数据失败:", ex)

# ================= 绘图 =================
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
fig = plt.figure(figsize=(14.5, 11))
gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.28)

# A1/A2
ax = fig.add_subplot(gs[0, 0])
sc = ax.scatter(umap_df.umap1, umap_df.umap2, c=umap_df.cluster, cmap="tab20", s=1, alpha=0.5, linewidths=0)
ax.set_title("A1 Token clusters (UOP discovery, K=20)"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax = fig.add_subplot(gs[0, 1])
lmap = {l: i for i, l in enumerate(labels8)}
lc = [lmap.get(l, 7) for l in umap_df.label]
cmap8 = plt.get_cmap("tab10")
sc = ax.scatter(umap_df.umap1, umap_df.umap2, c=[cmap8(i) for i in lc], s=1, alpha=0.5, linewidths=0)
ax.set_title("A2 Published cell labels (same cells)"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
handles = [Line2D([], [], marker='o', ls='', color=cmap8(lmap[l]), markersize=4, label=l) for l in labels8]
ax.legend(handles=handles, loc='upper right', fontsize=5.5, frameon=False, ncol=2, handletextpad=0.1)

# B heatmap (log2 enrichment, UOP & STA)
axb = fig.add_subplot(gs[0, 2])
piv_u = enr[enr.cohort == "UOP"].pivot(index="cluster", columns="cell_type", values="log2_enrichment").reindex(columns=labels8)
piv_s = enr[enr.cohort == "STA"].pivot(index="cluster", columns="cell_type", values="log2_enrichment").reindex(columns=labels8)
M = np.hstack([piv_u.values, piv_s.values])
im = axb.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
axb.set_yticks(range(20)); axb.set_yticklabels([f"{c}{'*' if c in cand_clusters else ''}" for c in piv_u.index], fontsize=5.5)
axb.set_xticks(range(len(labels8) * 2))
axb.set_xticklabels(list(labels8) + [l + "\n(STA)" for l in labels8], rotation=90, fontsize=5)
axb.axvline(len(labels8) - 0.5, color="k", lw=1)
axb.set_title("B Cluster × cell-type enrichment\nlog2 obs/exp (*=candidate; left UOP, right STA)", fontsize=8)
plt.colorbar(im, ax=axb, fraction=0.04, pad=0.02)

# C metrics
axc = fig.add_subplot(gs[1, 0])
mets = ["ARI", "AMI", "NMI", "purity_weighted"]
width = 0.35
xs = np.arange(len(mets))
for j, (setting, color) in enumerate([("UOP_discovery", "#1f77b4"), ("STA_validation", "#ff7f0e")]):
    r = evalv[evalv.setting == setting].iloc[0]
    vals = [r[m] for m in mets]
    lo = [r[m] - r[f"{m}_boot_lo"] for m in mets]
    hi = [r[f"{m}_boot_hi"] - r[m] for m in mets]
    axc.bar(xs + (j - 0.5) * width, vals, width, yerr=[lo, hi], capsize=2,
            color=color, alpha=0.85, label=setting.replace("_", " "))
axc.set_xticks(xs); axc.set_xticklabels(["ARI", "AMI", "NMI", "Purity\n(weighted)"])
axc.set_ylim(0, 0.8); axc.legend(fontsize=6, frameon=False)
axc.set_title("C External recovery of published labels\n(patient-level bootstrap 95% CI, B=500)")

# D paired slopes
axd = fig.add_subplot(gs[1, 1])
for j, coh in enumerate(["UOP", "STA"]):
    d = ki_pair[ki_pair.cohort == coh]
    axd.scatter([j - 0.06] * len(d), d.med_oth, s=10, color="#7f7f7f", alpha=0.7, zorder=2)
    axd.scatter([j + 0.06] * len(d), d.med_cand, s=10, color=CAND_COLOR, alpha=0.7, zorder=2)
    for _, r in d.iterrows():
        axd.plot([j - 0.06, j + 0.06], [r.med_oth, r.med_cand], color="k", alpha=0.12, lw=0.6, zorder=1)
    md = d["diff"].median()
    axd.plot([j - 0.2, j + 0.2], [d.med_oth.median() + md / 2, d.med_cand.median() + md / 2], ls="--", color=CAND_COLOR, lw=1)
axd.set_xticks([0, 1]); axd.set_xticklabels(["UOP\n(n=24)", "STA\n(n=24)"])
axd.set_ylabel("Per-patient median Ki-67\n(vessel-labelled cells, as-published)")
axd.set_title("D Ki-67: candidate-cluster vessel cells vs\nother-cluster vessel cells (paired patients)")
axd.legend(handles=[Line2D([], [], marker='o', ls='', color="#7f7f7f", label="other clusters"),
                    Line2D([], [], marker='o', ls='', color=CAND_COLOR, label="candidate cluster")],
           fontsize=6, frameon=False, loc="upper left")

# E joint distribution
axe = fig.add_subplot(gs[1, 2])
for arm, color, mk in [("other_clusters", "#7f7f7f", "."), ("candidate_cluster", CAND_COLOR, ".")]:
    for coh, z in [("UOP", 1), ("STA", 2)]:
        d = endo_s[(endo_s.arm == arm) & (endo_s.cohort == coh)]
        axe.scatter(d.CD31, d.Ki67, s=1.5, c=color, alpha=0.25 if z == 1 else 0.25, marker=mk,
                    edgecolors="none", label=None)
axe.set_xlabel("CD31 (as-published)"); axe.set_ylabel("Ki-67 (as-published)")
axe.set_title("E Single-cell CD31–Ki-67 joint distribution\n(vessel-labelled cells; grey=other, red=candidate)")
axe.legend(handles=[Line2D([], [], marker='o', ls='', color="#7f7f7f", label=f"other clusters (UOP+STA, n={int((endo_s.arm=='other_clusters').sum()):,})"),
                    Line2D([], [], marker='o', ls='', color=CAND_COLOR, label=f"candidate cluster (n={int((endo_s.arm=='candidate_cluster').sum()):,})")],
           fontsize=6, frameon=False)

# F
axf1 = fig.add_subplot(gs[2, 0]); axf2 = fig.add_subplot(gs[2, 1])
try:
    z = np.load(os.path.join(FIG, "panelF_roi.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(FIG, "panelF_roi.json")))
    img, msk = z["img"], z["msk"]
    ch = meta["panel_channels"]
    def norm99(a):
        a = a.astype(np.float32); q = np.quantile(a, 0.99)
        return np.clip(a / (q + 1e-6), 0, 1)
    rgb = np.dstack([norm99(img[ch["CD31"]]), norm99(img[ch["Ki67"]]), norm99(img[ch["Pancytokeratin"]])]) ** 0.7
    axf1.imshow(rgb); axf1.set_title(f"F1 ROI {roi_pick}\nR=CD31 G=Ki-67 B=panCK (raw, 0.99-norm)")
    # 轮廓: 候选vessel红 / 其他vessel灰 / 其余细胞淡蓝
    from matplotlib.colors import BoundaryNorm
    outl = np.zeros(msk.shape + (4,))
    seg = msk if msk.ndim == 2 else msk.sum(axis=0)
    obj_cand = set(z["ObjectNumber"][z["is_cand"] & z["vessel"]].tolist())
    obj_vess = set(z["ObjectNumber"][z["vessel"]].tolist())
    import numpy as _np
    from skimage import measure as _measure
    ovl = np.zeros(seg.shape + (4,), dtype=float)
    for o in _np.unique(seg):
        if o == 0: continue
        m = seg == o
        per = _measure.find_contours(m.astype(float), 0.5)
        col = (0.84, 0.15, 0.15, 1.0) if o in obj_cand else ((0.4, 0.4, 0.4, 0.9) if o in obj_vess else (0.3, 0.5, 0.8, 0.25))
        for p in per:
            axf2.plot(p[:, 1], p[:, 0], color=col, lw=0.6 if o in obj_vess or o in obj_cand else 0.2)
    axf2.set_xlim(0, seg.shape[1]); axf2.set_ylim(seg.shape[0], 0)
    axf2.set_aspect("equal"); axf2.axis("off")
    axf2.set_title("F2 Cell outlines\nred=candidate vessel; grey=other vessel; blue=others")
    axf1.axis("off")
except Exception as ex:
    axf1.text(0.5, 0.5, f"Panel F unavailable: {ex}", ha="center"); axf1.axis("off"); axf2.axis("off")

# 注释框（第3行第3列）
axn = fig.add_subplot(gs[2, 2]); axn.axis("off")
n_uop, n_sta = (cells.cohort == "UOP").sum(), (cells.cohort == "STA").sum()
txt = (
    "Run: %s\n"
    "Clustering: MiniBatchKMeans K=20, UOP only\n(273,408 cells, full fit), frozen VirTues tokens only\n"
    "STA: nearest-centroid assignment (224,830)\n"
    "Multi-seed stability: ARI 0.38–0.43 vs primary\n(median 0.407; K=20 partition seed-unstable)\n"
    "Vessel axis robust: vessel-dominated clusters\npresent in every seed and both cohorts\n"
    "Candidate cluster %s (criteria frozen in\ndated reanalysis spec:\nvessel%%≥0.50 ∧ CD31 top-5 ∧ ≥22/24 patients)\n"
    "Ki-67 paired (candidate − other vessel cells):\n"
    "UOP −0.001 (p=0.41); STA −0.012 (p=8e-6)\n→ no proliferation enrichment (direction negative)\n"
    "Markers as-published (Einhaus values); bootstrap unit\n= patient; UOP and STA analysed separately\n"
    "Lymphatic identity not evaluable (no PROX1/LYVE1 in panel)" % (RUN, ",".join(map(str, cand_clusters)))
)
axn.text(0.0, 0.98, txt, va="top", fontsize=6.8, family="monospace",
         bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#999999"))

fig.suptitle("Figure 5 | Unsupervised token-space phenotype recovery: seed-unstable partition, robust vessel axis, no proliferation enrichment",
             fontsize=11, y=0.995)
fig.savefig(os.path.join(MB, "fig5_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(MB, "fig5_v2.png"), dpi=300, bbox_inches="tight")
print("Fig5 saved")

# 合并source data（长格式）
parts = []
for f, panel in [("panelA_umap.csv", "A"), ("panelB_enrichment.csv", "B"), ("panelC_metrics.csv", "C"),
                 ("panelD_ki67_paired.csv", "D"), ("panelE_cd31_ki67.csv", "E")]:
    d = pd.read_csv(os.path.join(FIG, f))
    d.insert(0, "panel", panel)
    parts.append(d)
pd.concat(parts, ignore_index=True).to_csv(os.path.join(MB, "figure5_source_data.csv"),
                                            index=False, encoding="utf-8-sig")
print("source data saved")
