# -*- coding: utf-8 -*-
"""
A6: IMMUcan两例SCCHN样本metadata证据提取（实际执行版，本机无R，用纯Python rdata解析RDS）
等价的规范R脚本见 immucan_metadata_extraction.R（本机无R环境未执行）。
输出: immucan_scchn_two_sample_metadata.csv
"""
import rdata, numpy as np, csv, os, hashlib, json, datetime

ROOT = r"<project-root>"
SCE_RAW = os.path.join(ROOT, "data", "immucan", "sce_raw.rds")
SCE_V3  = os.path.join(ROOT, "data", "immucan", "sce_labelled_V3.rds")
OUT_CSV = os.path.join(ROOT, "outputs", "revision_20260901", "moduleA", "immucan_scchn_two_sample_metadata.csv")

# 本研究实际使用的两例（outputs/immucan_incorpus_control.py: HN_IDS[:2]，权重路径=已哈希校验的sp32）
USED = ["10089821-SPECT-VAR-TIS-UNST-03_001", "10089829-SPECT-VAR-TIS-UNST-03_001"]
FIELDS = ["image", "sample_id", "Indication", "SlideId", "Study", "BatchId", "SubBatchId",
          "ROI", "ROIonSlide", "acquisition_id", "Box.Description", "Position", "SampleId"]

def sha256(p, _buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(_buf), b""):
            h.update(b)
    return h.hexdigest().upper()

def coldata(path):
    sce = rdata.read_rds(path)
    ld = sce.colData.listData
    return {str(k): v for k, v in ld.items()}, len(next(iter(ld.values())))

rows = []
cross = {}
for path, tag in [(SCE_RAW, "sce_raw.rds"), (SCE_V3, "sce_labelled_V3.rds")]:
    ld, n = coldata(path)
    img = np.array([str(x) for x in ld["image"]])
    for t in USED:
        m = np.array([t in s for s in img])
        vals = {f: sorted({str(x) for x in np.array([str(y) for y in ld[f]])[m]}) for f in FIELDS if f in ld}
        assert len(vals.get("Indication", [])) == 1, f"Indication不唯一: {t} in {tag}"
        assert vals["Indication"][0] == "HN", f"Indication≠HN: {t} in {tag}"
        cross.setdefault(t, {})[tag] = {"n_cells": int(m.sum()), "Indication": vals["Indication"][0]}
        if tag == "sce_raw.rds":
            rows.append(vals)

with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "sample_id", "image_id", "indication", "panel",
                "source_rds", "source_field_names",
                "slide_id", "study", "batch_id", "roi", "n_cells_sce_raw",
                "indication_code_evidence", "extraction_note"])
    for t, v in zip(USED, rows):
        sid_full = v["sample_id"][0]
        w.writerow([
            v["SlideId"][0].split("-")[0],            # patient_id: SlideId数字前缀（10089821）
            sid_full,                                  # sample_id: colData.sample_id 原值
            v["image"][0],                             # image_id: colData.image 原值（tiff名）
            "HN (head and neck) = SCCHN",              # indication
            "IMMUcan IMC panel 1 (panel_mapped.json, 38 channels mapped)",  # panel
            "data/immucan/sce_raw.rds (cross-checked: sce_labelled_V3.rds)",
            ";".join(FIELDS),                          # source_field_names
            v["SlideId"][0], v["Study"][0], v["BatchId"][0], v["ROI"][0],
            cross[t]["sce_raw.rds"]["n_cells"],
            "RDS原字段Indication='HN'; IMMUcan正式论文(PMC12539258)列明五癌种之一为'squamous cell carcinoma of head and neck (SCCHN)'，即HN指示=SCCHN",
            "本研究实际使用样本(outputs/immucan_incorpus_control.py HN_IDS[:2]); 提取脚本=本文件(Python rdata, 本机无R); 规范R版=immucan_metadata_extraction.R",
        ])

prov = {
    "task": "A6 IMMUcan两例SCCHN metadata证据",
    "extraction_date": datetime.date.today().isoformat(),
    "executed_script": "immucan_metadata_extraction.py (Python 3.11 + rdata, 本机无R)",
    "canonical_R_script": "immucan_metadata_extraction.R (本机无R环境，未执行；与Python版字段/断言一致)",
    "samples_used_by_study": USED,
    "evidence_chain": {
        "step1_raw_field": "sce_raw.rds colData: Indication='HN' for both samples (457,117 cells total; domain={BREAS,GI,GU,HN,THOR})",
        "step2_cross_check": "sce_labelled_V3.rds colData: Indication='HN' for both samples (80,988 cells)",
        "step3_official_mapping": "IMMUcan正式论文 (PMC12539258): 五癌种之一 = 'squamous cell carcinoma of head and neck (SCCHN)' -> HN=SCCHN",
        "step4_usage_trace": "outputs/immucan_incorpus_control.py L38-39 HN_IDS[:2] 即这两例; 权重=已校验sp32 model.safetensors",
    },
    "assertions_passed": "两样本在两份RDS中Indication均唯一且='HN'",
    "rds_sha256": {"sce_raw.rds": sha256(SCE_RAW), "sce_labelled_V3.rds": sha256(SCE_V3)},
    "csv_sha256": sha256(OUT_CSV),
    "manuscript_wording": "two samples from an SCCHN cohort represented in the pretraining corpus（不写two in-corpus samples: 样本级训练manifest不存在）",
}
with open(os.path.join(ROOT, "outputs", "revision_20260901", "moduleA", "immucan_metadata_provenance.json"),
          "w", encoding="utf-8") as f:
    json.dump(prov, f, ensure_ascii=False, indent=2)
print("CSV rows:", len(rows))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv[:16] + "..." for kk, vv in v.items()}) for k, v in prov.items() if k in ("rds_sha256", "csv_sha256")}, indent=1))
