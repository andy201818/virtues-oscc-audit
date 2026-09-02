# -*- coding: utf-8 -*-
"""
阶段3b：Fig1/Fig2/Fig4 重制（对齐 A4/A6/B8 新定位）
Fig1 v2: 流程图——audit pipeline 框架（frozen checkpoint / SCCHN-cohort control / 语境5队列全in-corpus /
         Task3=token-space phenotype recovery+不稳定 / 描述性差异 / provenance锁定; 142 ROIs）
Fig2 v2: UMAP(标注着色)+逐类F1; 基准线改为cell-level macro-F1(0.756, 同协议口径); patient级移入文字注
Fig4 v2: SCCHN对照 vs OSCC——IMMUcan两样本逐点显示 + OSCC窗级bootstrap 95%CI; 标题改描述性口径
产出: figs/fig1_v2.* fig2_v2.* fig4_v2.* + fig2_v2_source_data.csv fig4_v2_source_data.csv
     (fig1为示意流程图, 无数据面板, 不含source data)
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "figs")
os.makedirs(OUT, exist_ok=True)

# ================= Fig 1 v2 =================
fig, ax = plt.subplots(figsize=(13.6, 8.6))
ax.set_xlim(0, 13.4); ax.set_ylim(0, 10.6); ax.axis("off")

def box(x, y, w, h, text, fc="#EAF2FA", ec="#4C72B0", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color="#555555", lw=1.2))

box(0.3, 8.3, 3.6, 1.3, "PRIMARY (frozen checkpoint)\nEinhaus 2023 OSCC\n48 patients / 142 ROIs / 39 markers",
    fc="#FDECEA", ec="#C44E52", bold=True)
box(0.3, 6.5, 3.6, 1.3, "SCCHN-COHORT CONTROL\nIMMUcan, 2 samples\n(cohort in pretraining corpus)\n38 channels",
    fc="#EAF7EE", ec="#55A868")
box(0.3, 4.7, 3.6, 1.5, "CROSS-CANCER CONTEXT\n5 IMC cohorts, all listed in\nthe 32-cohort corpus\n(descriptive only, no\ncorpus-membership contrast)",
    fc="#F3EEF9", ec="#8172B2")
box(4.9, 5.6, 3.0, 4.2, "AUDIT PROTOCOL\n\nESM-2 marker embeddings\n(39 markers, cos=1.00\nvs released)\n\nuq0.99 clip + log1p\n+ z-score\n\n128×128 windows, top-2 density\nper selected ROI (Task 2:\n36/142 ROI deterministic\nsubsample)\n\nprovenance locked\n(SHA-256, 3-way match)",
    fs=8.5)
box(8.8, 7.6, 4.0, 1.6, "TASK 1  Cell phenotyping\nfrozen-feature linear probe\npatient-level GroupKFold(5)\nmacro-F1 = 0.713 vs\nintensity baseline 0.783", fc="#EAF2FA")
box(8.8, 5.5, 4.0, 1.6, "TASK 2  Masked-channel\nreconstruction\nfull-channel masking ×39\nwindow-exchange permutation\n+ ridge/template baselines", fc="#EAF2FA")
box(8.8, 3.4, 4.0, 1.6, "TASK 3  Token-space\nphenotype recovery\nUOP discovery / STA validation\nmulti-seed stability audit\n(seed-unstable at K=20)", fc="#EAF2FA")
box(4.9, 2.6, 3.0, 1.9, "TWO EVALUATION PITFALLS\n\nlabel-derivation\ncircularity (probe vs\nintensity baseline)\n\ncluster seed-instability\n(ARI 0.38–0.43)", fc="#FFF6E5", ec="#DD8452")
box(8.8, 1.0, 4.0, 1.6, "OUTPUT\ncapability map + audit checklist\n+ full evidence package\n(all numbers traceable)", fc="#E8F5E9", ec="#55A868", bold=True)
arrow(3.9, 8.95, 4.9, 8.3); arrow(3.9, 7.15, 4.9, 7.3); arrow(3.9, 5.45, 4.9, 6.3)
arrow(7.9, 8.2, 8.8, 8.4); arrow(7.9, 7.4, 8.8, 6.3); arrow(7.9, 6.6, 8.8, 4.2)
arrow(7.9, 5.8, 7.4, 4.5); arrow(6.4, 2.6, 6.4, 2.3)
ax.text(0.3, 10.1, "Study cohorts", fontsize=12, fontweight="bold")
ax.text(4.9, 10.1, "Audit protocol", fontsize=12, fontweight="bold")
ax.text(8.8, 9.6, "Frozen-checkpoint tasks & outputs", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ================= Fig 2 v2 =================
X, y = [], []
for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "kao1_celltyping", "tokens", "*.npz"))):
    d = np.load(f, allow_pickle=True)
    X.append(d["tokens"]); y.append(d["labels"])
X = np.concatenate(X); y = np.concatenate(y)
rng = np.random.RandomState(0)
idx = []
for c in np.unique(y):
    ci = np.where(y == c)[0]
    take = min(len(ci), max(2000, int(30000 * len(ci) / len(y))))
    idx.append(rng.choice(ci, take, replace=False))
idx = np.concatenate(idx)
Xs, ys = X[idx], y[idx]
import umap
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, low_memory=True).fit_transform(Xs)

rep = open(os.path.join(ROOT, "outputs", "kao1_celltyping", "kao1_report.txt"), encoding="utf-8").read()
f1s = {}
for ln in rep.splitlines():
    p = ln.split()
    # v5.1修复: 从右侧解析(support,f1,recall,precision), 类名含空格不再丢失
    if len(p) >= 5 and p[-1].replace(".", "").isdigit() and p[-2].replace(".", "").isdigit() \
            and p[-3].replace(".", "").isdigit() and p[-4].replace(".", "").isdigit():
        name = " ".join(p[:-4])
        if name not in ("accuracy", "macro avg", "weighted avg"):
            f1s[name] = float(p[-2])
macro_cell = float(np.mean([f1s[c] for c in f1s]))
assert len(f1s) == 8, f1s

order = ["Tumor", "Vessel", "CD8 T cells", "B cells", "Fibroblasts", "Myeloid", "CD4 T cells", "other"]
cmap = plt.get_cmap("tab10")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [1.6, 1]})
for i, c in enumerate(order):
    m = ys == c
    ax1.scatter(emb[m, 0], emb[m, 1], s=1.5, color=cmap(i), alpha=0.5, label=c, rasterized=True)
ax1.set_title("Frozen VirTues cell tokens (n=%d cells shown)" % len(ys))
ax1.legend(markerscale=8, fontsize=9, loc="best", frameon=False)
ax1.set_xticks([]); ax1.set_yticks([])
vals = [f1s.get(c, np.nan) for c in order]
ax2.barh(np.arange(len(order))[::-1], vals, color=[cmap(i) for i in range(len(order))])
for yy, v in zip(np.arange(len(order))[::-1], vals):
    ax2.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=10)
ax2.set_yticks(np.arange(len(order))[::-1]); ax2.set_yticklabels(order, fontsize=10)
ax2.set_xlabel("per-class F1 (cell-level protocol)")
ax2.set_xlim(0, 1.12)
ax2.axvline(macro_cell, color="k", ls="--", lw=1)
ax2.text(0.02, 0.02, f"cell-level macro-F1 = {macro_cell:.3f}\n(patient-level 0.713 reported\n separately in Table 2)",
         fontsize=7, va="bottom", transform=ax2.transAxes,
         bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
ax2.set_title("Frozen-feature phenotyping (Task 1)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
pd.DataFrame({"cell_type": order, "f1_celllevel": [f1s.get(c) for c in order],
              "macro_f1_celllevel": macro_cell}).to_csv(
    os.path.join(OUT, "fig2_v2_source_data.csv"), index=False, encoding="utf-8-sig")

# ================= Fig 4 v2 =================
immu = json.load(open(os.path.join(ROOT, "outputs", "immucan_control", "incorpus_control_results.json")))
rm = pd.read_csv(os.path.join(ROOT, "outputs", "kao2_marker_matrix", "r_matrix.csv"))
markers = ["Histone H3", "Ki-67", "CD4", "CD68", "SMA", "CD8a"]   # 六个配对标记
mk_key = {"Histone H3": "HistoneH3", "Ki-67": "Ki67", "CD4": "CD4", "CD68": "CD68",
          "SMA": "aSMA", "CD8a": "CD8a"}
rows = []
rng4 = np.random.RandomState(20260901)                      # v5.1: 固定种子
tissues = rm["tissue"].unique()                             # v5.1: ROI(组织)级bootstrap, 2窗成组
for m in markers:
    k = mk_key[m]                                            # r_matrix列名
    m_immu = {"Histone H3": "Histone H3", "Ki-67": "Ki-67", "SMA": "SMA"}.get(m, m)  # immu键名
    dcol = rm[k]
    boot = []
    for _ in range(2000):
        pick = rng4.choice(tissues, size=len(tissues), replace=True)
        vals = np.concatenate([dcol[rm["tissue"] == t].dropna().values for t in pick])
        boot.append(np.mean(vals))
    osc = dcol.dropna().values
    for sid, v in immu.items():                              # IMMUcan两样本逐点
        if m_immu in v["full_channel"]:
            rows.append({"marker": m, "arm": "SCCHN-cohort control", "sample": sid,
                         "r": v["full_channel"][m_immu], "oscc_mean": np.mean(osc),
                         "osci_lo": np.percentile(boot, 2.5), "oscc_hi": np.percentile(boot, 97.5)})
d4 = pd.DataFrame(rows)
d4.to_csv(os.path.join(OUT, "fig4_v2_source_data.csv"), index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8.6, 5.4))
x = np.arange(len(markers))
w = 0.38
osc_mean = [d4[d4.marker == m].oscc_mean.iloc[0] for m in markers]
osc_lo = [d4[d4.marker == m].oscc_mean.iloc[0] - d4[d4.marker == m].osci_lo.iloc[0] for m in markers]
osc_hi = [d4[d4.marker == m].oscc_hi.iloc[0] - d4[d4.marker == m].oscc_mean.iloc[0] for m in markers]
ax.bar(x - w / 2, osc_mean, w, yerr=[osc_lo, osc_hi], capsize=3, color="#C44E52", alpha=0.85,
       label="OSCC Einhaus (mean; ROI-level bootstrap 95% CI, 36 ROIs, windows resampled together)")
pts = {m: d4[(d4.marker == m) & (d4.arm == "SCCHN-cohort control")].r.values for m in markers}
for i, m in enumerate(markers):
    ax.scatter(np.full(len(pts[m]), i + w / 2), pts[m], color="#55A868", zorder=3, s=42,
               edgecolor="k", linewidth=0.5, label="IMMUcan SCCHN samples (individual)" if i == 0 else None)
    ax.plot([i + w / 2 - 0.06, i + w / 2 + 0.06], [np.mean(pts[m])] * 2, color="#2c6e49", lw=1.6)
ax.set_xticks(x); ax.set_xticklabels(markers)
ax.set_ylabel("Full-channel inpainting r (identical pipeline)")
ax.set_title("Marker-dependent difference between the SCCHN-cohort control and OSCC\n(descriptive; corpus-membership attribution not tested)")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig1/2/4 v2 saved ->", OUT)
