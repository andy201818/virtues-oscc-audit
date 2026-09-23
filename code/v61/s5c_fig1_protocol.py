#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 v2: protocol-centric two-study schematic (replaces r9 flow figure).

Third-review item 4: old Figure 1 showed neither the CODEX branch nor the
five protocol elements that are now the paper's primary contribution.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.2, 7.4))
ax.set_xlim(0, 102); ax.set_ylim(0, 74); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3fb", ec="#31527a", fs=8, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x1, y1, x2, y2, color="#5a6b7f"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=1.1))

# ---- top: five protocol elements ----
ax.text(51, 72.2, "Patient-level evaluation protocol — five elements", ha="center",
        fontsize=11, fontweight="bold")
els = [
    "(1) Input matching\nsame visible channels\n& masking, both arms",
    "(2) Patient-level\nsplits\nGroupKFold · LOPO\npatient = unit",
    "(3) Statistical support\ndomain\ncell objects, exclusion\nrules declared",
    "(4) Paired patient\nmetrics\nordering rho, MAE,\npaired differences",
    "(5) Reporting rules\nNA rules, audit-correction\ndisclosure, descriptive\nBCa intervals",
]
ew, gap = 17.4, 1.8
x0 = (102 - 5 * ew - 4 * gap) / 2
for i, t in enumerate(els):
    box(x0 + i * (ew + gap), 59.4, ew, 9.8, t, fc="#e8eef8", fs=6.8)
arrow(51, 58.6, 51, 56.4)

# ---- middle: two study instantiations ----
box(4, 33, 44, 22.5, "", fc="#ffffff", ec="#31527a", lw=1.4)
ax.text(26, 52.6, "Study 1 — OSCC IMC (Einhaus et al. 2023)", ha="center", fontsize=9, fontweight="bold")
box(6.5, 44.2, 39, 7.0, "48 patients · 142 ROIs · 498,238 cells\nTask 2: 36 patients / 72 windows / 39+37 markers\ncomparator: LOWO ridge (window-exchange)", fc="#f6f8fc", fs=7.2)
box(6.5, 34.6, 39, 8.4, "patient-level endpoints: 76 marker-cohort\ncombinations, ordering vs mean-intensity error\n(separated) — primary evidence", fc="#f6f8fc", fs=7.2)

box(54, 33, 44, 22.5, "", fc="#ffffff", ec="#7a3131", lw=1.4)
ax.text(76, 52.6, "Study 2 — laryngeal SCC CODEX (Simkin et al. 2026)", ha="center", fontsize=9, fontweight="bold", color="#7a3131")
box(56.5, 44.2, 39, 7.0, "27 cores = presumed patients (admission: \nfive checklist conditions) · 12 visible markers\ncomparator: LOPO ridge, same 11 covisibles/target", fc="#fbf3f3", fs=7.2)
box(56.5, 34.6, 39, 8.4, "six prespecified targets, cell-level aggregation\n(Not_cells excluded), 336 forwards\nsame paired endpoints, BCa bootstrap", fc="#fbf3f3", fs=7.2)

arrow(34, 58.6, 26, 56.4); arrow(68, 58.6, 76, 56.4)
ax.text(51, 57.4, "common principles instantiated per study\n(study-specific adaptations declared)", ha="center", fontsize=7.4, style="italic", color="#5a6b7f")

# ---- bottom: shared finding + sensitivity chain ----
arrow(26, 32.6, 40, 29.4); arrow(76, 32.6, 62, 29.4)
box(20, 21.5, 62, 7.6,
    "Shared finding: spatial-alignment quality and patient-level value fidelity separate\n"
    "(informative ordering coexists with baseline-level or larger mean-intensity error)",
    fc="#eef8ee", ec="#317a4e", fs=8.0, bold=False)
box(14, 8.5, 74, 9.6,
    "Sensitivity chain (all rerun, not disclosed only): execution-path bit-identity (E0) · attention-fallback parity ·\n"
    "fold-restricted standardization — window r (max 9.6e-3), combination-mean MAE (max 6.8e-3; largest\n"
    "single-patient 4.8e-2), Task 1 probe macro-F1 (max 1.4e-3) — conclusions unchanged within verified scope",
    fc="#f7f4ee", ec="#7a5e31", fs=7.2)
arrow(51, 21.1, 51, 18.6)

fig.tight_layout()
import os
os.makedirs(r"<project-root>\outputs\s5", exist_ok=True)
fig.savefig(r"<project-root>\outputs\s5\fig1_protocol_v2.png", dpi=300)
fig.savefig(r"<project-root>\outputs\s5\fig1_protocol_v2.pdf")
print("fig1_protocol_v2 written")
