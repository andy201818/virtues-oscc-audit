# -*- coding: utf-8 -*-
"""Task 2 ROI 选择披露(v5.2, 报告M-04):
- 从 cache_manifest.json 导出全部72窗/36ROI的确定性选择清单
- 与未入选的106个ROI做代表性比较(每ROI细胞数、vessel细胞比例、队列)
- 幂等: 纯读取, 输出CSV/JSON
"""
import os, json
import numpy as np
import pandas as pd

ROOT = r"<project-root>"
OUT = os.path.join(ROOT, "outputs", "revision_20260901", "pixel_null_v2")
B = 2000
RNG = np.random.RandomState(20260902)

m = json.load(open(os.path.join(OUT, "cache_manifest.json"), encoding="utf-8"))
assert m["n_windows"] == 72 and len(m["windows"]) == 72
sel = pd.DataFrame(m["windows"])
# ROI = tissue(每队列sorted顺序k%4==0 → 每入选组织恰好2窗)
roi = (sel.groupby(["cohort", "tissue"])
       .agg(n_windows=("key", "size"), mean_density=("density", "mean")).reset_index())
roi["patient"] = roi.tissue.str.extract(r"(S\d+|OC\d+)", expand=False)
roi = roi.sort_values(["cohort", "tissue"]).reset_index(drop=True)
sel_out = sel.merge(roi[["cohort", "tissue", "patient"]], on=["cohort", "tissue"], how="left")
sel_out.to_csv(os.path.join(OUT, "task2_window_selection_72.csv"), index=False, encoding="utf-8-sig")
roi.to_csv(os.path.join(OUT, "task2_roi_selection_36.csv"), index=False, encoding="utf-8-sig")

# ---------- 代表性比较: 入选36 ROI vs 其余106 ROI ----------
cells = pd.read_parquet(os.path.join(ROOT, "outputs", "revision_20260901", "moduleB", "cells_base.parquet"))
per_roi = (cells.groupby(["cohort", "ROI_id"])
           .agg(n_cells=("cell_id", "size"), n_vessel=("vessel_label", "sum")).reset_index())
per_roi["vessel_frac"] = per_roi.n_vessel / per_roi.n_cells
sel_key = set(zip(roi.cohort, roi.tissue))
# ROI_id 命名对齐: cells_base 的 ROI_id 需能匹配 tissue id
cand_keys = per_roi[["cohort", "ROI_id"]].apply(tuple, axis=1)
per_roi["selected"] = [k in sel_key for k in cand_keys]
n_match = int(per_roi.selected.sum())
if n_match == 0:  # 命名不一致时按字符串包含匹配
    tmap = {}
    for _, r in per_roi.iterrows():
        hit = [t for c, t in sel_key if t in str(r.ROI_id) or str(r.ROI_id) in t]
        tmap[(r.cohort, r.ROI_id)] = bool(hit)
    per_roi["selected"] = [tmap[(r.cohort, r.ROI_id)] for _, r in per_roi.iterrows()]
    n_match = int(per_roi.selected.sum())
print("matched selected ROIs:", n_match)

rep = {}
for metric in ["n_cells", "vessel_frac"]:
    a = per_roi.loc[per_roi.selected, metric].dropna().values
    b = per_roi.loc[~per_roi.selected, metric].dropna().values
    boots = [np.median(RNG.choice(a, len(a), replace=True)) - np.median(RNG.choice(b, len(b), replace=True))
             for _ in range(B)]
    from scipy.stats import mannwhitneyu
    rep[metric] = {
        "selected_median": float(np.median(a)), "nonselected_median": float(np.median(b)),
        "median_diff": float(np.median(a) - np.median(b)),
        "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
        "mannwhitney_p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue),
        "n_selected": int(len(a)), "n_nonselected": int(len(b)),
    }
n_by_cohort = roi.groupby("cohort").patient.nunique().to_dict()
summary = {
    "selection_rule": "per cohort, tissues in sorted file order with index k % 4 == 0 (every fourth ROI), then the two highest tissue-density 128x128 windows per selected ROI",
    "n_patients_selected": int(roi.dropna(subset=['patient']).groupby(['cohort','patient']).ngroups),
    "n_patients_selected_by_cohort": {k: int(v) for k, v in n_by_cohort.items()},
    "n_roi_selected": int(len(roi)), "n_windows": 72,
    "n_roi_total": int(len(per_roi)), "n_roi_excluded": int((~per_roi.selected).sum()),
    "patients_all_48_note": "one ROI per selected patient; 36 of the 48 patients entered Task 2 (12 excluded by the k%4==0 rule)",
    "representativeness": rep,
    "note": "deterministic, post-hoc disclosed subsample set at cache-construction time (v5.1); full-142-ROI extension not run (compute); representativeness comparison computed in this script",
}
assert roi.patient.notna().all(), "patient extraction left empty values"
json.dump(summary, open(os.path.join(OUT, "task2_roi_disclosure.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
per_roi.to_csv(os.path.join(OUT, "task2_roi_representativeness.csv"), index=False, encoding="utf-8-sig")
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("TASK2_DISCLOSURE_DONE")
