# External evaluation of a frozen VirTues checkpoint in OSCC

This directory contains the revised companion analysis materials. Versioned archives of this repository are deposited on Zenodo under concept DOI [10.5281/zenodo.22271392](https://doi.org/10.5281/zenodo.22271392) (each release creates a new version; the concept DOI resolves to the latest version).

Analysis code and result tables for: **"External evaluation of a frozen VirTues checkpoint in oral squamous cell carcinoma"** (Wan, Ban & Liang).

An external evaluation of the frozen, publicly released [VirTues](https://huggingface.co/bunnelab/virtues) `virtues-sp32` checkpoint on public OSCC imaging mass cytometry data. It links interpretations to label provenance, spatial nulls, simple reconstruction baselines, paired uncertainty and clustering stability. The statistical components are established methods; their combination does not establish a new model, a universal validation standard or clinical utility. The implementation covers two cohorts from one study.

## Repository layout

```
code/Virtues/                       # VirTues official inference code (upstream snapshot)
outputs/revision_20260901/
  moduleA/                          # model & corpus provenance (SHA-256, environment, corpus table)
  moduleB/                          # Task 3 cell-level analysis (clustering, stability, paired tests)
  pixel_null_v2/                    # Task 2 permutation null + LOWO baselines + Figure 3
  figs/                             # figure generation scripts and outputs (Figures 1-6)
```

Result tables too large for the repository tree (per-cell tables, ~500k cells) are attached to the release **"Evidence package v5.4 (result tables)"**.

The current manuscript figures are the six numbered figures in `outputs/revision_20260901/figs/publication/`; older named figure files are retained for provenance and are not the current publication layouts. `render_revision_figures.py` regenerates Figures 1–4 and 6 from fixed inputs; `render_figure5.py` regenerates Figure 5. These plotting commands additionally require Matplotlib and scikit-image. No generative image synthesis is used.

## Reproduce summaries without a GPU

Python 3.11 or later with NumPy, pandas, SciPy and PyArrow is sufficient for these entry points. From this directory:

The tested CPU environment is recorded in `environment-summary.json`; the separate upstream GPU environment remains in `moduleA/environment.txt` under the revision outputs directory.

```bash
python reproduce_key_tables.py
python reproduce_normalization_moments.py
python build_claim_support_map.py
```

The first checks and summarizes fixed outputs for Tables 2–3 and selected supplementary results. It does not rerun token extraction, cross-validation, pixel reconstruction or the full spatial permutation analysis. The second recomputes normalization-moment summaries from ROI sufficient statistics and recorded patient folds; it does not assess prediction sensitivity. The third regenerates the post-hoc paired-difference table from cached predictions, with 10,000 cohort-stratified patient bootstrap resamples and seed 20260905. Its pointwise intervals are not multiplicity-adjusted and exclude refitting uncertainty.

`AUDIT_PROTOCOL.md` documents the reusable workflow; `REANALYSIS_SPECIFICATION.md` records timing and retrospective additions; `CLAIM_EVIDENCE_MAP.csv` links interpretations to controls, units and limitations. Under `outputs/revision_20260901/`, `task1/` contains run summaries, `moduleB/lopo_vs_lowo_by_marker.csv` contains patient-held-out baseline sensitivity, `zscore_sensitivity/` contains ROI moments and patient-fold maps, and `claim_support/` contains paired marker differences.

## Raw-input recomputation

Legacy analysis scripts record the original pipeline and require local data locations and, for inference, a compatible GPU environment. They are not all portable one-command entry points. Original image datasets and model weights are not redistributed here.

1. Clone this repository and note the local path (`<project-root>`).
2. In each analysis script, replace the `ROOT` placeholder (`<project-root>`) with your local repository path.
3. Environment: Python 3.12, PyTorch 2.11 (CUDA 12.8); package list in `outputs/revision_20260901/moduleA/environment.txt`.
4. Model weights: download `virtues-sp32/model.safetensors` from Hugging Face (academic-use licence CC BY-NC 4.0). Expected SHA-256:
   `974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7`
5. Datasets: all public, each cited by Zenodo DOI in the manuscript (Einhaus 2023 OSCC IMC; IMMUcan panel 1; Hoch 2022; Danenberg 2022; Allam 2022; Salié 2025; Ehret 2025). Not redistributed here.
6. Task 2 pixel cache (~180 MB) is rebuilt deterministically: run `pixel_null_v2/pn2_cache.py` then `pn2_analysis.py`.
7. Bootstrap and permutation procedures use recorded seeds (20260901/20260902; the revision-stage paired bootstrap uses 20260905).

Full-dataset image normalization remains transductive. LOPO sensitivity refits the baseline only. Negative Ki-67 findings, unstable partitions and the inconclusive recurrence association are retained. Neither cohort-list provenance nor this evaluation establishes sample-level exclusion from pretraining.

## Licence

Original analysis code: MIT. VirTues upstream code, model weights and datasets retain their upstream licences. Source-derived tables retain applicable attribution and reuse conditions; public release must respect those conditions.

## v6.1 additions (patient-level protocol transfer)

This version adds the execution/preprocessing sensitivity checks and the external laryngeal SCC CODEX transfer, matching the revised manuscript *"A patient-level evaluation protocol for frozen spatial proteomics models: a two-study VirTues analysis"*.

- `code/v61/` — cleaned analysis scripts. Machine-specific absolute paths are replaced by placeholders (`<project-root>` for this repository, `<data-root>` for local dataset/weight locations, `<cache-root>` for pre-existing cached predictions):
  - `e0_*` — streaming-adapter self-test (seeded synthetic data) and operator-level attention parity against a manually computed FP64 reference;
  - `e1_*` — N0 cached-forward replication and N1 fold-restricted moments rerun (Task-1 probe, Task-2 window-level and patient-level endpoints);
  - `e2_*` — external CODEX transfer execution under the audit-corrected (v2) protocol, admission-gating artefacts (channel adjudication, geometry check, window candidates) and the masked-forward runnability probe;
  - `s5_fig_ext.py`, `s5c_fig1_protocol.py`, `make_figure1_round5.py` — figure generation for the external-comparison and protocol-schematic figures.
- `outputs/e0/`, `outputs/e1/`, `outputs/e2/s4_v2/`, `outputs/e2/gating/`, `outputs/s5/` — 21 fixed result files (SHA-256 in `MANIFEST_SHA256.json`): external per-cell / per-window / per-patient predictions for both arms, paired metrics, the run manifest (seeds, window geometry, visible-marker list), admission-gating evidence and sensitivity summaries.

The external transfer cohort (S-BIAD3612, EBI BioStudies) is cited in the manuscript and not redistributed here. The earlier `outputs/revision_20260901/` tree is unchanged in this version. Recorded seeds, the five-condition admission checklist, the withdrawn first external implementation and the four audit corrections are documented in the manuscript and its supplement.

## Citation and archived version

Versioned archives: concept DOI [10.5281/zenodo.22271392](https://doi.org/10.5281/zenodo.22271392) (resolves to the latest released version; the per-version DOI is listed on the Zenodo record). The earlier [v5.5 source archive](https://doi.org/10.5281/zenodo.22271393) does not include the revision-stage additions. The accompanying file manifest identifies this local bundle; it is not proof of public deposition.
