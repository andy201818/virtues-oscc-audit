# VirTues–OSCC external audit

Analysis code and result tables for: **"External evaluation of a frozen VirTues checkpoint in oral squamous cell carcinoma"** (Wan, Ban & Liang).

An independent, third-party audit of the frozen, publicly released [VirTues](https://huggingface.co/bunnelab/virtues) `virtues-sp32` checkpoint on public oral squamous cell carcinoma (OSCC) imaging mass cytometry data, across three axes: frozen-feature linear probing, masked-channel reconstruction, and unsupervised token-space analysis. The paper reports a per-marker capability map, two evaluation pitfalls (label-derivation advantage in probing; clustering seed-instability), and a reusable audit checklist.

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

## Reproduction

1. Clone this repository and note the local path (`<project-root>`).
2. In each analysis script, replace the `ROOT` placeholder (`<project-root>`) with your local repository path.
3. Environment: Python 3.12, PyTorch 2.11 (CUDA 12.8); package list in `outputs/revision_20260901/moduleA/environment.txt`.
4. Model weights: download `virtues-sp32/model.safetensors` from Hugging Face (academic-use licence CC BY-NC 4.0). Expected SHA-256:
   `974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7`
5. Datasets: all public, each cited by Zenodo DOI in the manuscript (Einhaus 2023 OSCC IMC; IMMUcan panel 1; Hoch 2022; Danenberg 2022; Allam 2022; Salié 2025; Ehret 2025). Not redistributed here.
6. Task 2 pixel cache (~180 MB) is rebuilt deterministically: run `pixel_null_v2/pn2_cache.py` then `pn2_analysis.py`.
7. All bootstrap and permutation procedures use recorded seeds (20260901/20260902).

## Licence

Code: MIT. The VirTues upstream code retains its upstream licence; model weights remain under CC BY-NC 4.0; datasets under their upstream licences.
