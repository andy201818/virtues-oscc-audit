#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5: main external figure + comparison table for the paper.

Figure (Figure 6 candidate): per-target patient-level Spearman for VirTues vs
LOPO ridge with bootstrap CIs (n=27 patients), plus paired |err| means.
Table: dataset/input/budget comparison between OSCC (original) and laryngeal
CODEX (external transfer).

Inputs: outputs/e2/s4/paired_metrics.csv, s4 run_manifest, gating artifacts.
Outputs: outputs/s5/fig_external_paired_v2.{png,pdf}, table_transfer_comparison.csv
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN = r"<project-root>\outputs\e2\s4_v2\paired_metrics.csv"
MAN = r"<project-root>\outputs\e2\s4_v2\run_manifest.json"
OUTD = r"<project-root>\outputs\s5"
os.makedirs(OUTD, exist_ok=True)

df = pd.read_csv(IN)
targets = ["Ki67", "CD45", "CD68", "CD163", "CD31", "aSMA"]
x = np.arange(len(targets))
w = 0.38


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 4.2))
targets = df.target.tolist()  # labels from data, never hardcoded
x = np.arange(len(targets))

def parse_ci(v):
    m = __import__('re').match(r'\[(-?[0-9.]+),(-?[0-9.]+)\]', v)
    return float(m.group(1)), float(m.group(2))

# A: Spearman both arms with CI
loV, hiV = zip(*[parse_ci(v) for v in df.ci_virtues])
loR, hiR = zip(*[parse_ci(v) for v in df.ci_ridge])
w = 0.38
ax1.errorbar(x - w/2, df.spearman_virtues, yerr=[df.spearman_virtues - loV, np.array(hiV) - df.spearman_virtues],
             fmt='o', capsize=3, label='frozen VirTues', color='#1f77b4')
ax1.errorbar(x + w/2, df.spearman_ridge, yerr=[df.spearman_ridge - loR, np.array(hiR) - df.spearman_ridge],
             fmt='s', capsize=3, label='LOPO ridge', color='#d62728')
ax1.axhline(0, color='gray', lw=0.8, ls=':')
ax1.set_xticks(x); ax1.set_xticklabels(targets)
ax1.set_ylabel('patient-level Spearman rho (n=27)')
ax1.set_title('A  Ordering fidelity', fontsize=10)
ax1.legend(frameon=False, fontsize=8)
ax1.spines[['top','right']].set_visible(False)

# B: delta rho with CI
loD, hiD = zip(*[parse_ci(v) for v in df.ci_delta_rho])
colD = ['#d62728' if l > 0 or h < 0 else '#7f7f7f' for l, h in zip(loD, hiD)]
ax2.errorbar(x, df.delta_rho, yerr=[df.delta_rho - loD, np.array(hiD) - df.delta_rho],
             fmt='D', capsize=4, color='black', ecolor='gray')
for xi, (d, l, h) in enumerate(zip(df.delta_rho, loD, hiD)):
    ax2.plot(xi, d, 'D', color=colD[xi], markersize=6)
ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.set_xticks(x); ax2.set_xticklabels(targets)
ax2.set_ylabel('delta rho (VirTues - ridge)')
ax2.set_title('B  Ranking contrast', fontsize=10)
ax2.spines[['top','right']].set_visible(False)

# C: paired |err| difference with CI
loP, hiP = zip(*[parse_ci(v) for v in df.ci_dabs])
colP = ['#1f77b4' if l > 0 or h < 0 else '#7f7f7f' for l, h in zip(loP, hiP)]
ax3.errorbar(x, df.paired_dabs_mean, yerr=[df.paired_dabs_mean - loP, np.array(hiP) - df.paired_dabs_mean],
             fmt='o', capsize=4, color='black', ecolor='gray')
for xi, (d, l, h) in enumerate(zip(df.paired_dabs_mean, loP, hiP)):
    ax3.plot(xi, d, 'o', color=colP[xi], markersize=6)
ax3.axhline(0, color='gray', lw=0.8, ls=':')
ax3.set_xticks(x); ax3.set_xticklabels(targets, rotation=30, ha='right')
ax3.set_ylabel('paired |err| difference (log1p)')
ax3.set_title('C  Paired error difference', fontsize=10)
ax3.spines[['top','right']].set_visible(False)

fig.suptitle('Protocol transfer: independent laryngeal SCC CODEX cohort (27 presumed patients)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTD, 'fig_external_paired_v2.png'), dpi=300)
fig.savefig(os.path.join(OUTD, 'fig_external_paired_v2.pdf'))
plt.close(fig)
print('3-panel figure written')
# comparison table
man = json.load(open(MAN, encoding="utf-8"))
rows = [
    dict(item="Disease / site", original="oral squamous cell carcinoma", external="laryngeal SCC"),
    dict(item="Platform", original="imaging mass cytometry", external="CODEX multiplex IF"),
    dict(item="Study", original="Einhaus et al. 2023", external="Simkin et al. 2026 (S-BIAD3612)"),
    dict(item="Patients (analysis)", original="36 (Task 2) / 48 (Task 1 pool)", external="27 (27 cores=presumed patients; D2 appended-core sensitivity)"),
    dict(item="ROIs / cores", original="142 ROIs", external="27 cores (+D2)"),
    dict(item="Evaluation windows", original="72 (top-2 density per ROI)", external="54 (top-2 density per core)"),
    dict(item="Physical window", original="128 µm (128 px @ 1 µm/px)", external="128 µm (~253 px @ 0.5067 µm/px, resampled)"),
    dict(item="Visible channels", original="39 / 37 per cohort", external="12 (published markers with local ESM embeddings; limitation)"),
    dict(item="Targets", original="all 39 markers (per-cohort)", external="6 prespecified (CD45/CD31/CD68/CD163/Ki67/aSMA)"),
    dict(item="Comparator", original="LOWO ridge (window-exchange)", external="LOPO ridge (patient-exchange)"),
    dict(item="Forwards executed", original="2,736 (N0/N1 each)", external="336 (0.116 s each, measured)"),
    dict(item="GPU", original="RTX 5060 Ti 16 GB", external="same, peak 2.25 GB"),
]
pd.DataFrame(rows).to_csv(os.path.join(OUTD, "table_transfer_comparison.csv"), index=False, encoding="utf-8")
print("written:", OUTD)
