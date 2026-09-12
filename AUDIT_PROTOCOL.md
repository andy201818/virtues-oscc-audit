# External evaluation protocol for a frozen spatial proteomics checkpoint

Version 1.1, 5 September 2026. Companion to Wan, Ban and Liang, External evaluation of a frozen VirTues checkpoint in oral squamous cell carcinoma.

## Purpose and scope

This workflow links a proposed interpretation to the evidence needed to support it. It combines established controls; it is not a new statistical test, a universal validation standard or a clinical certification. Its implementation has been examined in one public OSCC study. A model-agnostic description does not establish performance in other diseases or models.

## 1. Record the evaluated object

Record the model source, version, licence and full checkpoint SHA-256. Distinguish frozen foundation-model weights from fitted downstream probes, baselines and clustering models. Verify channel identities and marker embeddings. State training-corpus overlap at the granularity actually documented; absence from a cohort list does not establish absence of particular patients or diseases.

## 2. Record inputs and timing

Record patient, ROI, window and cell identifiers; label provenance; missing channels; selection and exclusion rules; and the exact preprocessing sequence. Identify whether each fitted transform uses only training data. Explicitly label global normalization as transductive. Input-moment sensitivity alone does not demonstrate prediction invariance, even when two methods share preprocessing. Record separately which choices preceded analysis, followed exploration, or were added during revision. Non-significant selected-versus-excluded tests do not prove representative sampling.

## 3. Match claims to evidence

| Proposed claim | Relevant evaluation | What the result cannot establish alone |
|---|---|---|
| Reconstruction preserves spatial pairing | Spatially shifted null, valid exchange unit, add-one P and multiplicity control | Incremental performance or clinical utility |
| Reconstruction adds over a simple method | Paired comparison with a fitted linear imputation baseline and a template control | Panel removal, prospective transfer or biological validity |
| Prediction transfers to held-out patients | Patient-disjoint splitting, training-only fitting of preprocessing and downstream estimators | Independent disease-cohort generalization or unbiased reference labels |
| A cluster represents a reproducible structure | Multi-seed stability, held-out-cohort assignment, appropriate identity markers and patient-level follow-up | A biological subtype based on visualization alone |

These are complementary dimensions, not an automatic ladder from significance to utility.

## 4. Implement the comparisons

For cell phenotyping, compare token, intensity and fused features on matched partitions and record how labels were created. For masked-channel reconstruction, report each marker's sample count, observed score, spatial null and incremental score over simple baselines. If several windows belong to one patient, retain that grouping in resampling and assess whether baseline fits share patient information. Distinguish LOWO from LOPO and disclose any global preprocessing retained. For token clustering, record seeds, fitting cohort, assignment procedure and candidate selection separately from follow-up measurements.

## 5. Report uncertainty and unsupported interpretations

Report differences as well as individual scores. Distinguish pointwise intervals from multiplicity-adjusted inference. The companion post-hoc patient bootstrap treats already fitted predictions as fixed; it does not include uncertainty from model or baseline refitting. Do not equate a positive mean difference with a significant or clinically meaningful benefit. Retain unstable partitions, null associations and reversed marker directions. Label exploratory outcomes as exploratory and avoid multivariable claims unsupported by the event count.

## 6. Reproducible delivery checklist

- Exact model identity, source data identifiers and licences recorded.
- Patient and ROI counts, exclusions, missing channels and reference labels documented.
- Transform fitting scope and patient split map released.
- Per-window/per-patient outputs linked to claims and tables.
- Spatial null, simple comparator, paired uncertainty and stability checks distinguished.
- Retrospective additions identified; no implied prospective registration.
- Code that summarizes fixed outputs distinguished from raw-input recomputation.
- Negative and inconclusive findings retained in the main interpretation.
- Archive manifest, environment and runnable commands supplied.

The accompanying manuscript demonstrates the workflow on a frozen checkpoint. It does not demonstrate that every component is novel or that this protocol outperforms all existing evaluation workflows.
