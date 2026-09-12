# Retrospective consolidation of the OSCC reanalysis specification

Document prepared: 5 September 2026. Earlier reanalysis records referenced: 1 September 2026. This consolidated document was not created on 1 September and is not evidence of prospective registration. It describes the analysis configuration and distinguishes revision-stage additions from earlier computations. Where documentation and code disagree, the discrepancy must be resolved and reported, not silently overwritten.

## Model and input scope

The evaluated backbone is frozen virtues-sp32. Its model.safetensors SHA-256 is 974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7. Logistic probes, ridge baselines and clustering estimators are fitted downstream; frozen weights does not mean no fitted parameters anywhere in the workflow.

The primary dataset comprises UOP and STA, each with 24 patients and 71 ROIs. Phenotyping uses 498,238 cells and 37 shared intensity features. The image analysis covers 39 markers, including two UOP-only markers. Gaussian blur, per-ROI upper-quantile clipping, log1p and global image normalization precede inference. Global moments use the 142-ROI dataset and are transductive.

## Task 1

Frozen 512-dimensional tokens are evaluated by logistic regression. Patient GroupKFold is primary; ROI-grouped and cell-level splits are secondary. Intensity and fused features provide comparators. Probe scaling is fitted within each fold, but this does not undo the preceding global image normalization. The published labels were derived from intensity features, limiting interpretation of comparisons. Fold differences are descriptive; five cross-validation folds do not constitute five independent new cohorts.

## Task 2

Each target channel is fully masked in 128 by 128 windows. Pearson correlation is measured within the tissue mask. The deterministic every-fourth-ROI selection was chosen for computational reasons and reported retrospectively: 72 windows, 36 ROIs and 36 patients, with two windows per patient. UOP-only markers use 36 windows. Spatially shifted correlations are aggregated at window level using 9,999 draws and add-one P values, with BH correction across markers. LOWO template and ridge baselines use the same windows and evaluation mask. Bootstrap intervals in the original capability table resample ROIs.

## Task 3

MiniBatchKMeans with K=20 and n_init=10 is fitted on UOP tokens; STA assignment uses frozen UOP centroids. Seeds 0 through 9 quantify partition stability. The recorded candidate criteria are vessel-label fraction at least 0.50, CD31 mean among the top five clusters and coverage of at least 22 of 24 UOP patients. These criteria select the follow-up set; they do not establish lymphatic identity. Ki-67 follow-up compares vessel-labelled cells inside and outside the candidate cluster within patients. The panel lacks PROX1 and LYVE1. Recurrence analysis is exploratory and univariable, using 23 STA patients with 9 events after patient identifier harmonization.

## Additions during the September 2026 revision

1. LOPO baseline sensitivity excludes both windows from the held-out patient when fitting ridge/template baselines. Frozen VirTues predictions and global image normalization remain unchanged.
2. Input normalization sensitivity compares moments from auxiliary equal-patient partitions and from the actual cell-count-balanced Task 1 partitions. These are separate scenarios. No GPU prediction sensitivity was computed.
3. The claim-support map adds cohort-stratified paired patient bootstrap intervals on fixed LOWO prediction differences (10,000 resamples, seed 20260905). These are pointwise descriptive intervals, without multiplicity correction or refitting uncertainty.
4. The reusable protocol is a retrospective methodological synthesis, not a newly registered prospective analysis plan.

## Interpretation constraints

Spatial-null significance alone does not establish useful imputation. Positive mean differences alone do not establish statistical superiority. Patient grouping with global normalization is not wholly inductive validation. Unstable partitions and negative Ki-67 findings do not support biological subtype recovery. Outcome intervals crossing the null remain inconclusive. Cohort-level provenance does not establish sample-level absence from pretraining.
