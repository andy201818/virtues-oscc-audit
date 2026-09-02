# A6: IMMUcan两例SCCHN样本metadata证据提取（规范R版）
# 用途：在有R环境的机器上复现 immucan_scchn_two_sample_metadata.csv
# 本机（Windows, 分析工作机）无R，实际执行的是等价Python版 immucan_metadata_extraction.py
# （rdata包解析S4 SingleCellExperiment；两版字段、样本、断言一致；结果见provenance JSON）
#
# 依赖: R >= 4.2; install.packages("SingleCellExperiment") 不必需——readRDS原生解析即可
# 运行: Rscript immucan_metadata_extraction.R

ROOT <- "<project-root>"
SCE_RAW <- file.path(ROOT, "data/immucan/sce_raw.rds")
SCE_V3  <- file.path(ROOT, "data/immucan/sce_labelled_V3.rds")
OUT_CSV <- file.path(ROOT, "outputs/revision_20260901/moduleA/immucan_scchn_two_sample_metadata.csv_rerun")

# 本研究实际使用的两例（outputs/immucan_incorpus_control.py: HN_IDS[:2]）
USED <- c("10089821-SPECT-VAR-TIS-UNST-03_001", "10089829-SPECT-VAR-TIS-UNST-03_001")

FIELDS <- c("image", "sample_id", "Indication", "SlideId", "Study", "BatchId", "SubBatchId",
            "ROI", "ROIonSlide", "acquisition_id", "Box.Description", "Position", "SampleId")

extract <- function(path, tag) {
  sce <- readRDS(path)
  cd <- as.data.frame(colData(sce))          # SummarizedExperiment::colData
  for (t in USED) {
    m <- grepl(t, cd$image, fixed = TRUE)
    stopifnot(sum(m) > 0)
    ind <- unique(as.character(cd$Indication[m]))
    stopifnot(length(ind) == 1, ind == "HN")  # 原字段必须唯一且为HN
    if (tag == "sce_raw.rds") {
      out <<- rbind(out, data.frame(
        patient_id  = strsplit(as.character(cd$SlideId[m])[1], "-")[[1]][1],
        sample_id   = as.character(cd$sample_id[m])[1],
        image_id    = as.character(cd$image[m])[1],
        indication  = "HN (head and neck) = SCCHN",
        panel       = "IMMUcan IMC panel 1 (panel_mapped.json, 38 channels mapped)",
        source_rds  = "data/immucan/sce_raw.rds (cross-checked: sce_labelled_V3.rds)",
        source_field_names = paste(FIELDS, collapse = ";"),
        slide_id    = as.character(cd$SlideId[m])[1],
        study       = as.character(cd$Study[m])[1],
        batch_id    = as.character(cd$BatchId[m])[1],
        roi         = as.character(cd$ROI[m])[1],
        n_cells_sce_raw = sum(m),
        indication_code_evidence = paste("RDS原字段Indication='HN'; IMMUcan正式论文(PMC12539258)",
                                         "列明'squamous cell carcinoma of head and neck (SCCHN)', 即HN=SCCHN"),
        extraction_note = "R版复现; 首次提取由Python rdata版执行(本机无R)",
        stringsAsFactors = FALSE))
    }
    cat(sprintf("[%s] %s: Indication=%s, n_cells=%d\n", tag, t, ind, sum(m)))
  }
}

out <- data.frame()
extract(SCE_RAW, "sce_raw.rds")
extract(SCE_V3,  "sce_labelled_V3.rds")      # 交叉验证（只断言，不重复写行）
write.csv(out, OUT_CSV, row.names = FALSE, fileEncoding = "UTF-8")
cat("written:", OUT_CSV, "\n")
