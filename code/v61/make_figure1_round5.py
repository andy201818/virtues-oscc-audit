# -*- coding: utf-8 -*-
"""
Round-5 Figure 1 redraw (gate ①). Replaces the graphic whose element (1) still read
"Input matching — same visible channels & masking, both arms" and whose sensitivity
strip merged the four checks as "all rerun".

Content contract = the round-5 caption in CMPB_manuscript.md:
  five audit elements (documented input conditions / patient-aware splitting /
  declared aggregation domains / separate ordering and error endpoints / explicit
  reporting rules), two study panels with recorded input conditions
  (VirTues 38 vs ridge 36/37 for UOP; 11 covisible matched in CODEX),
  LOPO in both studies + LOWO as the original spatial baseline, and the four
  sensitivity checks reported as distinct scopes.

Numbers are taken from pn2_analysis.py (37 shared markers; shared targets -> 36
features; UOP-only CD86/Podoplanin -> 37) and the manuscript's established counts
(48/142/498,238; 36/72; 39/37; 76; 27; 12/11; 336; 228). No new data; schematic only.

Self-QA: after rendering, the script extracts the PDF text layer and asserts the
required strings are present and the banned strings are absent; it also asserts a
minimum font size. Exit code is non-zero on any failure.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(7.2, 8.6))
fig.subplots_adjust(left=.02, right=.98, bottom=.01, top=.99)
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")

C_HDR = "#DCE8F4"; C_EL = "#ECF2F8"; C_IN = "#FBF3E4"; C_EP = "#EAF3EC"; C_SENS = "#F3ECF5"

def box(x, y, w, h, s, fill=C_EL, bold=False, fs=6.4, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fill, ec="#61768C", lw=.8))
    ax.text(x + w / 2, y + h / 2, s, ha="center", va=va, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)

# ---- title ----
ax.text(5, 12.65, "Patient-level audit design across two studies", ha="center",
        va="center", fontsize=11, fontweight="bold")

# ---- five audit elements ----
els = [
    ("1  Documented input\nconditions",
     "per-comparison visible-\nchannel accounting;\ninputs differ for UOP\ntargets"),
    ("2  Patient-aware\nsplitting",
     "patient = split unit;\nLOPO / GroupKFold"),
    ("3  Declared\naggregation domains",
     "cells, windows, cohorts;\nexclusion rules stated"),
    ("4  Separate ordering\nand error endpoints",
     "Spearman ordering vs\npatient-mean MAE,\nreported apart"),
    ("5  Explicit\nreporting rules",
     "NA rules, audit-correction\ndisclosure, descriptive\nBCa intervals"),
]
bw, gap, x0, ytop = 1.84, 0.12, 0.20, 12.25
for i, (head, body) in enumerate(els):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, ytop - 1.45), bw, 1.45, boxstyle="round,pad=0.06",
                                fc=C_EL, ec="#61768C", lw=.8))
    ax.text(x + bw / 2, ytop - 0.38, head, ha="center", va="center",
            fontsize=6.8, fontweight="bold", linespacing=1.25)
    ax.text(x + bw / 2, ytop - 1.02, body, ha="center", va="center",
            fontsize=5.9, linespacing=1.3, color=".15")
ax.text(5, 12.44, "five audit elements — instantiated per study, adaptations declared",
        ha="center", va="center", fontsize=6.2, style="italic", color=".3")
for i in range(5):
    ax.annotate("", xy=(x0 + i * (bw + gap) + bw / 2, 10.42), xytext=(5, ytop - 1.5),
                arrowprops=dict(arrowstyle="-", color=".6", lw=.5))

# ---- Study 1 (top -> bottom: header bar, cohort lines, Input conditions, Endpoints) ----
box(0.2, 9.85, 9.6, 0.45, "Study 1 — OSCC IMC (Einhaus et al. 2023): original analysis, retrospective audit",
    fill=C_HDR, bold=True, fs=7.2)
ax.text(0.45, 9.58, "Phenotyping: 48 patients · 142 ROIs · 498,238 annotated cells",
        fontsize=6.3, va="center")
ax.text(0.45, 9.36, "Patient-level reconstruction: 36 patients · 72 windows · panel union 39 markers (STA subset 37)",
        fontsize=6.3, va="center")
box(0.35, 8.20, 9.3, 1.00,
    "Input conditions (documented, per comparison)\n"
    "VirTues: 38 visible channels in UOP; 36 in STA, after target masking\n"
    "Ridge: 36 shared predictors for shared targets; 37 for UOP-only targets (CD86, Podoplanin)\n"
    "Inputs are matched in STA, but differ in UOP; UOP-only markers never imputed in STA",
    fill=C_IN, fs=6.0, va="center")
box(0.35, 6.35, 9.3, 1.05,
    "Endpoints\n"
    "Patient endpoint (primary): LOPO ridge comparator — leave-one-patient-out fitting of the ridge arm\n"
    "Spatial endpoint (original analysis): LOWO ridge baseline — leave-one-window-out\n"
    "76 marker–cohort combinations; ordering and error endpoints reported separately",
    fill=C_EP, fs=6.0, va="center")

# ---- Study 2 ----
box(0.2, 5.62, 9.6, 0.42, "Study 2 — laryngeal SCC CODEX transfer (Simkin et al. 2026)",
    fill=C_HDR, bold=True, fs=7.2)
ax.text(0.45, 5.36, "27 cores = presumed patients (admission: five checklist conditions) · 12-marker selected panel",
        fontsize=6.3, va="center")
ax.text(0.45, 5.14, "Matched inputs: 11 covisible channels per target — identical visible set for both arms",
        fontsize=6.3, va="center")
ax.text(0.45, 4.92, "Patient endpoint: LOPO ridge · six prespecified targets · cell-level aggregation (Not_cells excluded) · 336 forwards including D2 sensitivity",
        fontsize=6.3, va="center")
ax.text(0.45, 4.70, "Same paired endpoints (ordering vs patient-mean error) · descriptive BCa intervals",
        fontsize=6.3, va="center")

# ---- shared finding ----
box(0.2, 3.35, 9.6, 0.80,
    "Shared finding: spatial alignment and patient-level fidelity provide different evidence —\n"
    "informative patient ordering can coexist with larger patient-mean errors than ridge.",
    fill="#F5F5F5", fs=6.3)

# ---- sensitivity ----
box(0.2, 1.55, 9.6, 1.60,
    "Sensitivity checks — four distinct scopes, reported separately (Table S15)\n"
    "(i) streaming adapter self-test on seeded synthetic data\n"
    "(ii) replication of 228 real-data cached reconstruction forwards\n"
    "(iii) attention fallback checked against a manually computed FP64 reference\n"
    "(iv) fold-restricted re-estimation of normalization means/variances (per-ROI clipping rule retained)\n"
    "No single check establishes end-to-end equivalence; scopes cover selected implementation and\n"
    "normalization dependencies, not all differences to the original environment.",
    fill=C_SENS, fs=6.0)

# ---- footnote ----
ax.text(5, 1.15, "Retrospective protocol; frozen VirTues checkpoint (hash recorded); study-specific adaptations declared.",
        ha="center", va="center", fontsize=6.0, style="italic", color=".25")
ax.text(5, 0.85, "The workflow does not certify biological fidelity or clinical use.",
        ha="center", va="center", fontsize=6.0, style="italic", color=".25")

pdf = OUT / "Figure_1.pdf"; png = OUT / "Figure_1.png"
fig.savefig(pdf); fig.savefig(png, dpi=400)

sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
print(f"[QA] min font pt = {min(sizes)}  (page width 7.2 in)")
assert min(sizes) >= 5.5, "font too small"
plt.close(fig)

# ---- self-QA on the rendered text layer ----
import fitz
text = "".join(p.get_text() for p in fitz.open(pdf))
norm = " ".join(text.split())
required = ["Documented input", "conditions", "Patient-aware splitting",
            "Declared aggregation domains", "Separate ordering", "and error endpoints",
            "Explicit", "reporting rules", "leave-one-patient-out", "leave-one-window-out",
            "38 visible channels in UOP; 36 in STA", "36 shared predictors for shared targets",
            "37 for UOP-only targets", "11 covisible channels",
            "OSCC IMC", "Einhaus", "CODEX", "Simkin", "76 marker", "four distinct scopes",
            "seeded synthetic", "228 real-data cached", "FP64 reference",
            "per-ROI clipping rule retained", "Inputs are matched in STA, but differ in UOP",
            "336 forwards including D2 sensitivity"]
banned = ["same visible channels", "Input matching", "all rerun", "bit-identity",
          "every normalization", "full cohort panel", "baseline-level"]
missing = [s for s in required if s not in norm]
present_banned = [s for s in banned if s in norm]
print(f"[QA] required present: {len(required)-len(missing)}/{len(required)}; "
      f"banned present: {len(present_banned)}")
for s in missing: print("  MISSING:", s)
for s in present_banned: print("  BANNED PRESENT:", s)
if missing or present_banned:
    sys.exit(1)
print("Figure 1 round-5 render OK:", pdf)
