# -*- coding: utf-8 -*-
"""
阶段1b+阶段2（CPU）：像素null v2 置换检验 + 轻量重建基线（LOWO模板 / LOWO ridge）
锁定规范（运行前固定，作者已批准）：
- shift bank: 每窗×标记 K=40 个循环平移（dr,dc∈[8,120)），RNG seed=20260901；shift级r全部落盘
- 置换: 交换单位=window；每次置换每窗独立抽1个shift → 窗均值统计量T；B=9999
- P值: add-one (b+1)/(B+1)；单侧为主（重建优于空间失配配对），双侧敏感性同表
- 多重检验: BH-FDR × 39标记（单侧P为基准）
- 基线（必做，同72窗同指标同tissue mask）:
    * LOWO template: 目标标记在其他71窗（该标记存在的窗）组织像素上的均值图像
    * LOWO ridge: 共享目标用其余36个共享通道为特征；UOP独有目标(Podoplanin/CD86)用全部37个共享通道、Gram仅UOP窗累计并留一窗（λ=1.0）
产出: shift_level.parquet / permutation_results.csv / baselines_r.csv /
      fig3_source_data.csv / fig3_v2.pdf/png / pn2_manifest.json
"""
import os, glob, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "pixel_null_v2")
CACHE = os.path.join(OUT, "cache")
K_SHIFT, B, SEED, LAMBDA = 40, 9999, 20260901, 1.0

files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
assert len(files) == 72, f"缓存窗数={len(files)}≠72"
wins = []
for f in files:
    z = np.load(f, allow_pickle=True)
    wins.append({"key": os.path.basename(f)[:-4], "crop": z["crop"].astype(np.float32),
                 "rec": z["rec"].astype(np.float32), "tm": z["tm"].astype(bool),
                 "names": [str(x) for x in z["names"]]})
all_markers = sorted(set().union(*[set(w["names"]) for w in wins]), key=lambda m: -1)
print("windows:", len(wins), "| UOP:", sum('UOP' in w['key'] for w in wins), "| STA:", sum('STA' in w['key'] for w in wins))

def pear(a, b, m):
    a = a[m].astype(np.float64); b = b[m].astype(np.float64)
    if len(a) < 100:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

# ---------- shift bank ----------
rng = np.random.RandomState(SEED)
shift_rows = []
t0 = time.time()
for w in wins:
    for ci, m in enumerate(w["names"]):
        gt = w["crop"][ci]; rr = w["rec"][ci]; tm = w["tm"]
        obs = pear(gt, rr, tm)
        shifts = rng.randint(8, 120, size=(K_SHIFT, 2))
        for si, (dr, dc) in enumerate(shifts):
            r_s = pear(gt, np.roll(rr, (dr, dc), axis=(0, 1)), tm)
            shift_rows.append({"key": w["key"], "cohort": "UOP" if "UOP" in w["key"] else "STA",
                               "marker": m, "shift_idx": si, "r_obs": obs, "r_shift": r_s})
print(f"shift bank done ({time.time()-t0:.0f}s, {len(shift_rows)} rows)")
sdf = pd.DataFrame(shift_rows)
sdf.to_parquet(os.path.join(OUT, "shift_level.parquet"), index=False)

# ---------- 置换检验 ----------
rng2 = np.random.RandomState(SEED + 1)
perm_rows = []
for m, gm in sdf.groupby("marker", sort=True):
    keys = gm.key.unique()
    n_w = len(keys)
    obs_by = gm.groupby("key").r_obs.first().values
    sh_mat = np.stack([gm[gm.key == kk].r_shift.values for kk in keys])   # (n_w, K)
    assert sh_mat.shape[1] == K_SHIFT
    T_obs = np.nanmean(obs_by)
    draws = rng2.randint(0, K_SHIFT, size=(B, n_w))
    vals = sh_mat[np.arange(n_w)[None, :], draws]      # (B, n_w): 每窗独立抽1个shift
    T_b = np.nanmean(vals, axis=1)
    p_one = (1 + np.sum(T_b >= T_obs)) / (B + 1)
    cen = np.nanmean(T_b)
    p_two = (1 + np.sum(np.abs(T_b - cen) >= np.abs(T_obs - cen))) / (B + 1)
    perm_rows.append({"marker": m, "n_windows": int(n_w), "T_obs_mean_r": T_obs,
                      "null_mean": cen, "null_sd": np.nanstd(T_b),
                      "p_one_sided_addone": p_one, "p_two_sided_addone": p_two,
                      "perm_B": B, "exchange_unit": "window"})
perm = pd.DataFrame(perm_rows)
# BH-FDR（单侧为基准）
from statsmodels.stats.multitest import multipletests
rej, q, _, _ = multipletests(perm.p_one_sided_addone.values, alpha=0.05, method="fdr_bh")
perm["q_BH_one_sided"] = q
perm["significant_FDR05"] = rej
perm = perm.sort_values("T_obs_mean_r", ascending=False)
perm.to_csv(os.path.join(OUT, "permutation_results.csv"), index=False, encoding="utf-8-sig")
print(perm.head(8).round(4).to_string(index=False))
print("FDR<0.05:", int(rej.sum()), "/", len(rej))

# ---------- 轻量基线 ----------
# LOWO template: 目标标记在其余含该标记窗的标准化图像逐像素平均（留一窗; 弱变化模板）
tmpl_sum, tmpl_cnt = {}, {}
for w in wins:
    for ci, m in enumerate(w["names"]):
        tmpl_sum[m] = tmpl_sum.get(m, 0) + w["crop"][ci].astype(np.float64)
        tmpl_cnt[m] = tmpl_cnt.get(m, 0) + 1
# LOWO ridge: 特征=37个共享标记（两队列panel交集）中除目标外的通道, Gram留一窗
shared_names = sorted(set.intersection(*[set(w["names"]) for w in wins]))
assert len(shared_names) == 37
s_index = {n: i for i, n in enumerate(shared_names)}
G_list = []
for w in wins:
    rows = [s_index[n] for n in shared_names if n in w["names"]]
    assert len(rows) == 37
    X = w["crop"][[w["names"].index(n) for n in shared_names]][:, w["tm"]].T.astype(np.float64)
    G_list.append(X.T @ X)
G_arr = np.stack(G_list)
G_tot = G_arr.sum(axis=0)
G_tot_uop = np.sum([G_arr[i] for i, w in enumerate(wins) if "UOP" in w["key"]], axis=0)  # v5.1修复: 非共享目标仅UOP Gram
# 非共享目标(CD86/Podoplanin)与37共享通道的交叉矩累计（仅UOP窗有值）
C_tot_uop = {}
for w in wins:
    if "UOP" not in w["key"]:
        continue
    wname_pos = {n: i for i, n in enumerate(w["names"])}
    Xs = w["crop"][[wname_pos[n] for n in shared_names]][:, w["tm"]].T.astype(np.float64)
    for m in set(w["names"]) - set(shared_names):
        y = w["crop"][wname_pos[m]][w["tm"]].astype(np.float64)
        C_tot_uop[m] = C_tot_uop.get(m, 0) + Xs.T @ y

base_rows = []
for wi, w in enumerate(wins):
    wname_pos = {n: i for i, n in enumerate(w["names"])}
    Xt_shared = w["crop"][[wname_pos[n] for n in shared_names]][:, w["tm"]].T.astype(np.float64)
    for m in w["names"]:
        ci = wname_pos[m]
        tm = w["tm"]; gt = w["crop"][ci]; rr = w["rec"][ci]
        r_virt = pear(gt, rr, tm)
        # LOWO template（留一窗逐像素平均模板）
        tmpl = ((tmpl_sum[m] - w["crop"][ci].astype(np.float64)) / (tmpl_cnt[m] - 1)).astype(np.float32)
        r_tmpl = pear(gt, tmpl, tm)
        # LOWO ridge:
        #   共享目标: 特征=其余36共享通道, Gram/G_total留一窗
        #   非共享目标(CD86/Podoplanin): 特征=全部37共享通道, 交叉矩仅UOP窗累计, 留一窗
        G_lo = G_tot - G_arr[wi]
        try:
            if m in s_index:
                mi = s_index[m]
                feats = [j for j in range(len(shared_names)) if j != mi]
                Goo = G_lo[np.ix_(feats, feats)] + LAMBDA * np.eye(len(feats))
                boo = G_lo[np.ix_(feats, [mi])].ravel()
            else:
                c_lo = (C_tot_uop[m] - Xt_shared.T @ gt[tm].astype(np.float64))
                feats = list(range(len(shared_names)))
                G_lo_ns = G_tot_uop - G_arr[wi]          # v5.1修复: 留一窗(仅UOP窗)
                Goo = G_lo_ns[np.ix_(feats, feats)] + LAMBDA * np.eye(len(feats))
                boo = c_lo.ravel()
            beta = np.linalg.solve(Goo, boo)
            Xo = w["crop"][[wname_pos[shared_names[j]] for j in feats]][:, tm].T.astype(np.float64)
            pred = Xo @ beta                       # (npix_tissue,)
            gtm = gt[tm].astype(np.float64)
            r_ridge = float(np.corrcoef(gtm, pred)[0, 1]) if len(gtm) >= 100 else np.nan
        except np.linalg.LinAlgError:
            r_ridge = np.nan
        base_rows.append({"key": w["key"], "cohort": "UOP" if "UOP" in w["key"] else "STA",
                          "marker": m, "r_virtues": r_virt, "r_ridge_lowo": r_ridge,
                          "r_template_lowo": r_tmpl})
bd = pd.DataFrame(base_rows)
bd.to_parquet(os.path.join(OUT, "baselines_celllevel.parquet"), index=False)
bsum = bd.groupby("marker").agg(n=("r_virtues", "size"), r_virtues=("r_virtues", "mean"),
                                r_ridge=("r_ridge_lowo", "mean"),
                                r_template=("r_template_lowo", "mean")).reset_index()
bsum["virtues_minus_ridge"] = bsum.r_virtues - bsum.r_ridge
bsum["virtues_minus_template"] = bsum.r_virtues - bsum.r_template
bsum = bsum.sort_values("r_virtues", ascending=False)
bsum.to_csv(os.path.join(OUT, "baselines_r.csv"), index=False, encoding="utf-8-sig")
print(bsum.head(8).round(4).to_string(index=False))

# ---------- Fig 3 v2（先落盘source data, 绘图从CSV读取 → 图可由source独立重绘） ----------
plt.rcParams.update({"font.size": 8})
d = perm.merge(bsum[["marker", "r_ridge"]], on="marker", how="left")
d["sig"] = np.where(d.q_BH_one_sided <= 0.05, "FDR<0.05", "n.s.")
d["n_w"] = d.n_windows.map(lambda x: f"n={x}")
d.to_csv(os.path.join(OUT, "fig3_source_data.csv"), index=False, encoding="utf-8-sig")
d = pd.read_csv(os.path.join(OUT, "fig3_source_data.csv"))
fig, ax = plt.subplots(figsize=(9.5, 6.2))
colors = np.where(d.q_BH_one_sided <= 0.05, "#1f77b4", "#bbbbbb")
ax.bar(range(len(d)), d.T_obs_mean_r, color=colors, edgecolor="none")
# v5.2修复: null envelope 以 null_mean 为中心(报告 M-02), 不再画在 observed 周围
for i, (_, r) in enumerate(d.iterrows()):
    ax.plot([i, i], [r.null_mean - 1.96 * r.null_sd, r.null_mean + 1.96 * r.null_sd],
            color="#444444", lw=0.6)
    ax.plot([i], [r.null_mean], "_", color="#444444", markersize=4)
ax.plot(range(len(d)), d.r_ridge, "_", color="#d62728", markersize=6, label="ridge baseline (LOWO, leave-one-window-out)")
ax.set_xticks(range(len(d)))
ax.set_xticklabels([f"{m}\n({n})" for m, n in zip(d.marker, d.n_w)], rotation=90, fontsize=5.5)
ax.set_ylabel("Mean masked-region Pearson r")
ax.set_title("Masked-channel reconstruction vs window-exchange null\n(bar=observed; whisker=null mean ±1.96·SD of permutation null; blue=FDR<0.05, B=9999, add-one P; red=ridge baseline)")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_v2.png"), dpi=300, bbox_inches="tight")
print("fig3_v2 saved")

json.dump({"K_shift": K_SHIFT, "B": B, "seed": SEED, "lambda_ridge": LAMBDA,
           "exchange_unit": "window", "p_definition": "add-one (b+1)/(B+1)",
           "fdr": "BH on one-sided P, 39 markers",
           "n_windows": 72, "template_baseline": "LOWO逐像素平均模板(留一窗); 主基线=LOWO ridge(λ=1.0; 共享目标特征=其余36共享通道, UOP独有目标=37共享通道/仅UOP窗Gram留一窗)",
           "generated_by": "pn2_analysis.py"},
          open(os.path.join(OUT, "pn2_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("PN2_ANALYSIS_DONE")
