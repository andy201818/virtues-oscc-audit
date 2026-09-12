# -*- coding: utf-8 -*-
"""
模块B 阶段3（分析规范B7/B8/B9/B10）：内皮相关分析 + 命名决策 + 预后处置
B7 候选cluster选择（锁定规范标准, 仅UOP, 不含Ki-67）:
    vessel_pct_uop >= 0.50 且 CD31均值排名<=5(top quartile of 20) 且 患者覆盖>=22/24
  比较对象: 候选cluster中的vessel-labelled cells vs 其他cluster中的vessel-labelled cells
  Ki-67: 患者级配对差值(候选vessel中位 - 其他vessel中位), Wilcoxon + 患者bootstrap CI;
        混合模型敏感性: Ki67 ~ candidate + cohort + (1|patient) [与 (1|patient:ROI)]
  阈值敏感性: 队列内全细胞Ki67分位{75,80,90,95}% + 2x中位
  PDPN: 仅UOP探索性; STA面板无PDPN→不分析(NA)
B8 决策规则(脚本内固定): 见 b8_decision.json
B9 从底表重算候选集合的vessel_pct/Ki67/CD31/PDPN等(对照旧审计值仅作历史参照)
B10 STA 3年复发(二分类, 无时间/删失→不做Cox): 单变量logistic, 暴露=候选cluster细胞占比(每1SD),
    报告OR/95%CI/P; 表述=estimate imprecise/inconclusive; 不拟合多变量
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import statsmodels.api as sm

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "moduleB")
B = 1000
MIN_CELLS_PER_ARM = 5          # 患者纳入阈值: 两臂各>=5个vessel细胞
RNG = np.random.RandomState(20260901)

cells = pd.read_parquet(os.path.join(OUT, "cells_base.parquet"))
assign = pd.read_parquet(os.path.join(OUT, "cluster_assignments_all_seeds.parquet"))
prof = pd.read_csv(os.path.join(OUT, "cluster_profile_summary.csv"))
stab = pd.read_csv(os.path.join(OUT, "cluster_stability.csv"))
cells["cluster"] = assign.set_index("cell_id").loc[cells.cell_id, "cluster_primary"].values

# ---------- B7a: 候选选择(锁定规范, 仅UOP, 无Ki-67) ----------
cd31_rank = prof.cd31_mean_uop.rank(ascending=False, method="min").astype(int)
sel_crit = pd.DataFrame({
    "cluster": prof.cluster,
    "vessel_pct_uop": prof.vessel_pct_uop,
    "vessel_pct_pass": prof.vessel_pct_uop >= 0.50,
    "cd31_mean_uop": prof.cd31_mean_uop,
    "cd31_rank": cd31_rank,
    "cd31_pass": cd31_rank <= 5,
    "patient_coverage_uop": prof.patient_coverage_uop,
    "coverage_pass": prof.patients_uop >= 22,
    "ki67_mean_uop_not_used": prof.ki67_mean_uop,   # 记录, 不参与选择
})
sel_crit["candidate"] = sel_crit.vessel_pct_pass & sel_crit.cd31_pass & sel_crit.coverage_pass
sel_crit.to_csv(os.path.join(OUT, "candidate_selection_locked.csv"), index=False, encoding="utf-8-sig")
cand_clusters = sorted(sel_crit.loc[sel_crit.candidate, "cluster"].tolist())
print("候选clusters:", cand_clusters)
print(sel_crit.to_string(index=False))

cells["is_candidate_cluster"] = cells.cluster.isin(cand_clusters)
vess = cells[cells.vessel_label].copy()
vess["arm"] = np.where(vess.is_candidate_cluster, "candidate_cluster", "other_clusters")
vess_out = vess[["cell_id", "cohort", "patient_id", "ROI_id", "cluster", "is_candidate_cluster", "arm",
                 "Ki67", "CD31", "Podoplanin", "panCK", "tumour_border_distance", "location"]]
vess_out.to_csv(os.path.join(OUT, "endothelial_cell_analysis.csv"), index=False, encoding="utf-8-sig")
print("vessel细胞:", len(vess), vess.cohort.value_counts().to_dict())

# ---------- B7b: 患者级配对(Ki67; PDPN仅UOP) ----------
def paired_patient(df, marker, label):
    rows = []
    for coh in ["UOP", "STA"]:
        d = df[(df.cohort == coh) & df[marker].notna()]
        for pat, g in d.groupby("patient_id"):
            a = g.loc[g.arm == "candidate_cluster", marker]
            b = g.loc[g.arm == "other_clusters", marker]
            if len(a) >= MIN_CELLS_PER_ARM and len(b) >= MIN_CELLS_PER_ARM:
                rows.append({"cohort": coh, "patient_id": pat,
                             "n_cand": len(a), "n_oth": len(b),
                             "med_cand": float(a.median()), "med_oth": float(b.median()),
                             "diff": float(a.median() - b.median())})
    pr = pd.DataFrame(rows)
    res = {"patient_table": pr, "label": label, "by_cohort": {}}
    for scope, d in [("UOP", pr[pr.cohort == "UOP"]), ("STA", pr[pr.cohort == "STA"]),
                     ("pooled", pr)]:
        if len(d) < 5:
            continue
        diffs = d["diff"].values
        med = float(np.median(diffs)); mean = float(np.mean(diffs))
        try:
            w_p = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
        except ValueError:
            w_p = np.nan
        boots = []
        for _ in range(B):
            bs = d.sample(len(d), replace=True, random_state=RNG.randint(1 << 30))["diff"].values
            boots.append(np.median(bs))
        res["by_cohort"][scope] = {
            "n_patients": len(d), "median_diff": med, "mean_diff": mean,
            "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "wilcoxon_p": w_p, "n_diff_positive": int((diffs > 0).sum()),
        }
    return res

ki67_res = paired_patient(vess, "Ki67", "Ki-67 (as-published continuous)")
pdpn_res = paired_patient(vess[vess.cohort == "UOP"], "Podoplanin", "Podoplanin (UOP only, exploratory)")
print(json.dumps(ki67_res["by_cohort"], ensure_ascii=False, indent=1))
ki67_res["patient_table"].to_csv(os.path.join(OUT, "ki67_patient_paired.csv"), index=False, encoding="utf-8-sig")
pdpn_res["patient_table"].to_csv(os.path.join(OUT, "pdpn_patient_paired_uop.csv"), index=False, encoding="utf-8-sig")

# ---------- B7c: 混合模型敏感性(细胞级, 患者随机截距) ----------
mm_rows = []
vm = vess.dropna(subset=["Ki67"]).copy()
for groups_col, tag in [("patient_id", "random_patient"), ("ROI_id", "random_patientROI")]:
    try:
        if groups_col == "ROI_id":
            vm["patientROI"] = vm.patient_id + ":" + vm.ROI_id
            groups_col = "patientROI"
        md = smf.mixedlm("Ki67 ~ is_candidate_cluster + cohort", vm, groups=vm[groups_col])
        # C4 修复 2026-09-12：lbfgs 曾边界坍缩（Group Var=0、cohort CI ±1.4e4/±1.7e5、Hessian 非 PD）。
        # 换 powell；守卫拒绝退化拟合：converged ∧ Group Var 有限且>0 ∧ 目标行齐全 ∧ 参数/CI/P 有限
        # ∧ CI 下界<上界 ∧ max CI 宽度<1（NaN 经有限性与 <1 双通道拒绝，防 pandas.max 跳 NaN 漏检）。
        # 注：statsmodels 对健康拟合也恒发 "MLE may be on the boundary" 谨慎警告，无区分力，不作拒绝条件。
        fit = md.fit(method="powell", maxiter=1000)
        _gv = float(fit.cov_re.iloc[0, 0])
        _ci = fit.conf_int()
        _need = [t for t in ("is_candidate_cluster[T.True]", "cohort[T.UOP]") if t in fit.params.index]
        _w = (_ci[1] - _ci[0]).astype(float)
        _ok = (fit.converged and np.isfinite(_gv) and _gv > 0 and len(_need) == 2
               and all(np.isfinite(float(fit.params[t])) and np.isfinite(float(fit.pvalues[t])) for t in _need)
               and bool(np.isfinite(_w[_w.index.isin(_need)].values).all())
               and all(float(_ci.loc[t, 0]) < float(_ci.loc[t, 1]) for t in _need)
               and bool(np.isfinite(_w).all()) and float(_w.max()) < 1)
        assert _ok, \
            f"degenerate mixed-model fit ({tag}): gv={_gv}, terms={len(_need)}, max_w={_w.max()}, finite={bool(np.isfinite(_w).all())}"
        for name in ["is_candidate_cluster[T.True]", "cohort[T.UOP]"]:
            if name in fit.params.index:
                ci = fit.conf_int().loc[name]
                mm_rows.append({"model": tag, "term": name, "coef": float(fit.params[name]),
                                "ci_lo": float(ci.iloc[0]), "ci_hi": float(ci.iloc[1]),
                                "p": float(fit.pvalues[name]), "n_cells": len(vm),
                                "n_groups": vm[groups_col].nunique()})
    except AssertionError:
        raise  # 守卫失败必须致命，不得落成 ERROR 行静默通过
    except Exception as e:
        mm_rows.append({"model": tag, "term": "ERROR", "coef": np.nan, "ci_lo": np.nan,
                        "ci_hi": np.nan, "p": np.nan, "n_cells": len(vm), "n_groups": np.nan})
pd.DataFrame(mm_rows).to_csv(os.path.join(OUT, "mixed_model_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ---------- B7d: 阈值敏感性 ----------
thr_rows = []
for coh in ["UOP", "STA"]:
    pool = cells[(cells.cohort == coh) & cells.Ki67.notna()].Ki67
    thrs = {f"q{int(q*100)}": float(pool.quantile(q)) for q in [0.75, 0.80, 0.90, 0.95]}
    thrs["2x_median"] = float(2 * pool.median())
    dv = vess[vess.cohort == coh]
    for tname, t in thrs.items():
        rows = []
        for pat, g in dv.groupby("patient_id"):
            a = g.loc[g.arm == "candidate_cluster", "Ki67"]
            b = g.loc[g.arm == "other_clusters", "Ki67"]
            if len(a) >= MIN_CELLS_PER_ARM and len(b) >= MIN_CELLS_PER_ARM:
                rows.append(a.gt(t).mean() - b.gt(t).mean())
        rows = np.array(rows)
        if len(rows) >= 5:
            boots = [np.median(RNG.choice(rows, len(rows), replace=True)) for _ in range(B)]  # v5.2修复: 固定RNG
            try:
                w_p = float(wilcoxon(rows).pvalue) if np.any(rows != 0) else 1.0
            except ValueError:
                w_p = np.nan
            thr_rows.append({"cohort": coh, "threshold_name": tname, "threshold_value": t,
                             "n_patients": len(rows), "median_diff_frac_ki67pos": float(np.median(rows)),
                             "ci95_lo": float(np.percentile(boots, 2.5)),
                             "ci95_hi": float(np.percentile(boots, 97.5)),
                             "wilcoxon_p": w_p})
pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ---------- B9: 从底表重算(候选集合 vs 其余; UOP/STA/pooled) ----------
b9 = {}
for scope, d in [("UOP", cells[cells.cohort == "UOP"]), ("STA", cells[cells.cohort == "STA"]),
                 ("pooled", cells)]:
    cand = d[d.is_candidate_cluster]; oth = d[~d.is_candidate_cluster]
    b9[scope] = {
        "candidate_cell_fraction": float(len(cand) / len(d)) if len(cand) else np.nan,
        "candidate_vessel_pct": float(cand.vessel_label.mean()) if len(cand) else np.nan,
        "other_vessel_pct": float(oth.vessel_label.mean()),
        "vessel_cells_in_candidate_mean_ki67": float(cand.loc[cand.vessel_label, "Ki67"].mean()) if len(cand) else np.nan,
        "vessel_cells_in_other_mean_ki67": float(oth.loc[oth.vessel_label, "Ki67"].mean()),
        "vessel_cells_in_candidate_mean_cd31": float(cand.loc[cand.vessel_label, "CD31"].mean()) if len(cand) else np.nan,
        "vessel_cells_in_other_mean_cd31": float(oth.loc[oth.vessel_label, "CD31"].mean()),
    }
    if scope == "UOP":
        b9[scope]["vessel_cells_in_candidate_mean_pdpn"] = float(cand.loc[cand.vessel_label, "Podoplanin"].mean())
        b9[scope]["vessel_cells_in_other_mean_pdpn"] = float(oth.loc[oth.vessel_label, "Podoplanin"].mean())
json.dump(b9, open(os.path.join(OUT, "b9_recomputed_numbers.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------- 患者级cluster汇总 ----------
pl = (cells.groupby(["cohort", "patient_id", "cluster"])
      .agg(n_cells=("cell_id", "size"), n_vessel=("vessel_label", "sum")).reset_index())
pl["cell_fraction"] = pl.n_cells / pl.groupby(["cohort", "patient_id"]).n_cells.transform("sum")
pl["is_candidate_cluster"] = pl.cluster.isin(cand_clusters)
psum = (cells.groupby(["cohort", "patient_id"])
        .agg(n_cells_total=("cell_id", "size"), n_vessel_total=("vessel_label", "sum"),
             n_cells_candidate=("is_candidate_cluster", "sum")).reset_index())
psum["candidate_cell_fraction"] = psum.n_cells_candidate / psum.n_cells_total
pl = pl.merge(psum, on=["cohort", "patient_id"], how="left")
pl.to_parquet(os.path.join(OUT, "patient_level_cluster_summary.parquet"), index=False)
pl.to_csv(os.path.join(OUT, "patient_level_cluster_summary.csv"), index=False, encoding="utf-8-sig")

# ---------- B10: 预后(单变量logistic, 每暴露1SD) ----------
E = os.path.join(ROOT, "data", "einhaus2023", "extracted", "OSCC-IMC Einhaus et al. 2023")
meta = pd.read_excel(os.path.join(E, "STACohort", "Metadata", "STA_metadata.xlsx"))
meta["patient_id"] = meta["ID"].astype(str).str.extract(r"S(\d+)", expand=False).map(
    lambda x: f"S{int(x)}" if pd.notna(x) else None)
rec_col = "Recurrence within 3yrs (0=no, 1=yes)"
sta_ps = psum[psum.cohort == "STA"][["patient_id", "candidate_cell_fraction"]]
vsel = vess[vess.cohort == "STA"].groupby("patient_id").apply(
    lambda g: g.loc[g.arm == "candidate_cluster"].shape[0] / max(g.shape[0], 1), include_groups=False
).rename("vessel_frac_in_candidate").reset_index()
out = meta.merge(sta_ps, on="patient_id", how="left").merge(vsel, on="patient_id", how="left")
out = out.dropna(subset=[rec_col, "candidate_cell_fraction"])
out["recur3y"] = out[rec_col].astype(int)
oc_rows = []
for expo, lab in [("candidate_cell_fraction", "candidate-cluster cell fraction (per 1 SD)"),
                  ("vessel_frac_in_candidate", "vessel-cell fraction in candidate clusters (per 1 SD)")]:
    d = out.dropna(subset=[expo]).copy()
    z = (d[expo] - d[expo].mean()) / d[expo].std(ddof=1)
    try:
        fit = sm.Logit(d.recur3y, sm.add_constant(z)).fit(disp=0)
        ci = fit.conf_int()
        oc_rows.append({"outcome": "Recurrence within 3yrs (binary)", "n": int(len(d)),
                        "n_events": int(d.recur3y.sum()), "exposure": lab,
                        "OR_per_1SD": float(np.exp(fit.params.iloc[1])),
                        "ci95_lo": float(np.exp(ci.iloc[1, 0])), "ci95_hi": float(np.exp(ci.iloc[1, 1])),
                        "p": float(fit.pvalues.iloc[1]),
                        "interpretation": "exploratory; estimate imprecise/inconclusive"})
    except Exception as e:
        oc_rows.append({"outcome": "Recurrence within 3yrs (binary)", "exposure": lab,
                        "interpretation": f"fit failed: {str(e)[:80]}"})
pd.DataFrame(oc_rows).to_csv(os.path.join(OUT, "outcome_analysis.csv"), index=False, encoding="utf-8-sig")

# ---------- B8: 决策 ----------
ari = stab[stab.seed != 0].ari_vs_primary
median_ari = float(ari.median())
k67 = ki67_res["by_cohort"]
stable_ki67 = (
    len(cand_clusters) > 0
    and "pooled" in k67 and "UOP" in k67 and "STA" in k67
    and k67["pooled"]["ci95_lo"] > 0
    and k67["UOP"]["median_diff"] > 0 and k67["STA"]["median_diff"] > 0
    and min(k67["UOP"]["wilcoxon_p"], k67["STA"]["wilcoxon_p"]) < 0.05
)
if median_ari < 0.70:
    decision = "situation_4_clustering_unstable"
elif len(cand_clusters) == 0:
    decision = "no_vessel_enriched_clusters"
elif stable_ki67:
    decision = "situation_2_proliferative_endothelial_like"
else:
    decision = "situation_1_cd31_high_vessel_enriched"
b8 = {
    "median_ari_vs_primary_across_seeds": median_ari,
    "min_ari_vs_primary": float(ari.min()),
    "candidate_clusters": cand_clusters,
    "candidate_selection_criteria": "UOP only; vessel_pct>=0.50 AND CD31 mean rank<=5/20 AND patients>=22/24; Ki-67 not used",
    "ki67_paired_results": k67,
    "pdpn_uop_exploratory": pdpn_res["by_cohort"],
    "decision_rule_applied": "locked-specification decision rule (specification B8)",
    "decision": decision,
    "naming": {
        "situation_1": "CD31-high vessel-enriched token clusters",
        "situation_2": "proliferative endothelial-like token clusters",
        "situation_3": "The vessel-enriched token clusters did not show reproducible proliferation enrichment.",
        "situation_4": "delete Task-3 biological rediscovery claims; keep methodological negative result only",
    }.get(decision.split("_")[0] + "_" + decision.split("_")[1] if decision.startswith("situation") else decision, None),
}
if decision == "situation_1_cd31_high_vessel_enriched" and "pooled" in k67 and not (k67["pooled"]["ci95_lo"] > 0):
    b8["naming"] = "CD31-high vessel-enriched token clusters; " + \
        "The vessel-enriched token clusters did not show reproducible proliferation enrichment."
json.dump(b8, open(os.path.join(OUT, "b8_decision.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("B8 decision:", decision)
print(json.dumps(b8, ensure_ascii=False, indent=1)[:2000])
