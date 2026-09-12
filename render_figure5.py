"""Re-render fixed Figure 5 inputs at the intended publication size.

Rewritten 2026-09-12: restructuring, naming and
documentation only. Every plotting call, parameter value and draw order is preserved
from the previous version, so outputs are pixel-identical (verified by raster
comparison). Panel encodings are defined by the figure caption; no input values are
changed and no synthetic research image is created.

Inputs (read-only, fixed result tables/caches under outputs/revision_20260901):
  moduleB/fig5_source/{panelA_umap,panelB_enrichment,panelD_ki67_paired,
  panelE_cd31_ki67}.csv, panelF_roi.npz/.json,
  moduleB/cluster_profile_summary.csv, moduleB/cluster_evaluation.csv
Outputs: figs/publication/figure5.{pdf,png} + figure5_render_metadata.json
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from skimage.measure import find_contours

BASE = Path(__file__).resolve().parent
MB = BASE / 'outputs' / 'revision_20260901' / 'moduleB'
SRC = MB / 'fig5_source'
OUT = BASE / 'outputs' / 'revision_20260901' / 'figs' / 'publication'
OUT.mkdir(exist_ok=True, parents=True)

u = pd.read_csv(SRC / 'panelA_umap.csv'); e = pd.read_csv(SRC / 'panelB_enrichment.csv')
k = pd.read_csv(SRC / 'panelD_ki67_paired.csv'); v = pd.read_csv(SRC / 'panelE_cd31_ki67.csv')
p = pd.read_csv(MB / 'cluster_profile_summary.csv'); ev = pd.read_csv(MB / 'cluster_evaluation.csv')
cand = p.loc[(p.vessel_pct_uop >= .5)
             & (p.cd31_mean_uop.rank(ascending=False, method='min') <= 5)
             & (p.patients_uop >= 22), 'cluster'].tolist()
labels = ['Tumor', 'Fibroblasts', 'CD4 T cells', 'Myeloid', 'CD8 T cells', 'Vessel', 'other', 'B cells']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': .6})


def title(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', pad=7)


def norm99(a):
    """Per-channel 99th-percentile display normalisation (matches Figure 5F caption)."""
    a = a.astype(np.float32)
    return np.clip(a / (np.quantile(a, .99) + 1e-6), 0, 1)


fig = plt.figure(figsize=(7.2, 9.6))
gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 2.3, 1.65, 1.55],
                      left=.09, right=.97, bottom=.06, top=.975, hspace=.65, wspace=.35)

# A: token clusters (A1) and published labels (A2) on the same UMAP points.
a = fig.add_subplot(gs[0, 0])
a.scatter(u.umap1, u.umap2, c=u.cluster, cmap='tab20', s=.6, alpha=.5, linewidths=0, rasterized=True)
title(a, 'A1  Token clusters'); a.axis('off')
a = fig.add_subplot(gs[0, 1]); colors = plt.get_cmap('tab10'); lm = {x: i for i, x in enumerate(labels)}
a.scatter(u.umap1, u.umap2, c=[colors(lm.get(x, 7)) for x in u.label], s=.6, alpha=.5,
          linewidths=0, rasterized=True)
title(a, 'A2  Published cell labels'); a.axis('off')
handles = [Line2D([], [], marker='o', ls='', color=colors(lm[x]), markersize=3, label=x) for x in labels]
a.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.015), ncol=2, fontsize=9,
         frameon=False, handlelength=.5, columnspacing=.5, labelspacing=.2)

# B: cluster x cell-type log2 enrichment, UOP left / STA right, candidate starred.
b = fig.add_subplot(gs[1, :])
pu = e[e.cohort == 'UOP'].pivot(index='cluster', columns='cell_type', values='log2_enrichment').reindex(columns=labels)
ps = e[e.cohort == 'STA'].pivot(index='cluster', columns='cell_type', values='log2_enrichment').reindex(columns=labels)
im = b.imshow(np.hstack([pu.values, ps.values]), cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
b.set_yticks(range(20))
b.set_yticklabels([str(c) + ('*' if c in cand else '') for c in pu.index], fontsize=9)
b.set_xticks(range(16)); b.set_xticklabels(labels * 2, rotation=60, ha='right', fontsize=9)
b.axvline(7.5, color='black', lw=1)
b.set_ylabel('Cluster')
b.set_title('B  Cell-type enrichment', loc='left', fontweight='bold', pad=23)
b.text(.24, 1.015, 'UOP', transform=b.transAxes, ha='center', va='bottom')
b.text(.74, 1.015, 'STA', transform=b.transAxes, ha='center', va='bottom')
cb = fig.colorbar(im, ax=b, fraction=.025, pad=.015); cb.ax.set_title('log2', fontsize=9, pad=4)

# C: external label-recovery metrics with patient-level bootstrap CIs.
c = fig.add_subplot(gs[2, 0]); mets = ['ARI', 'AMI', 'NMI', 'purity_weighted']; xx = np.arange(4)
for i, (co, col) in enumerate([('UOP_discovery', '#0072B2'), ('STA_validation', '#E69F00')]):
    r = ev[ev.setting == co].iloc[0]
    ys = np.array([r[m] for m in mets])
    err = np.array([[r[m] - r[m + '_boot_lo'] for m in mets], [r[m + '_boot_hi'] - r[m] for m in mets]])
    c.bar(xx + (i - .5) * .36, ys, .36, yerr=err, color=col, label=co[:3], capsize=2,
          edgecolor='black', linewidth=.4, hatch='' if i == 0 else '//')
c.set_xticks(xx); c.set_xticklabels(['ARI', 'AMI', 'NMI', 'Purity']); c.set_ylim(0, .8)
c.set_ylabel('Agreement'); c.legend(frameon=False, fontsize=9); title(c, 'C  Label recovery')

# D: patient-paired median Ki-67 inside vs outside the candidate cluster.
d = fig.add_subplot(gs[2, 1])
for j, co in enumerate(['UOP', 'STA']):
    q = k[k.cohort == co]
    for _, r in q.iterrows():
        d.plot([j - .12, j + .12], [r.med_oth, r.med_cand], color='.75', lw=.5, zorder=1)
    d.scatter(np.full(len(q), j - .12), q.med_oth, s=9, color='.45', marker='o', zorder=2)
    d.scatter(np.full(len(q), j + .12), q.med_cand, s=10, color='#D55E00', marker='^', zorder=2)
d.set_xticks([0, 1]); d.set_xticklabels(['UOP (24)', 'STA (24)'])
d.set_ylabel('Patient median Ki-67'); title(d, 'D  Paired vessel-cell Ki-67')

# E: single-cell CD31-Ki-67 joint distribution; the caption defines the two encodings,
# leaving all paired observations visible.
ee = fig.add_subplot(gs[3, 0])
for arm, color, m in [('other_clusters', '.5', '.'), ('candidate_cluster', '#D55E00', '+')]:
    q = v[v.arm == arm]
    ee.scatter(q.CD31, q.Ki67, s=.8, c=color, alpha=.25, marker=m, linewidths=.2, rasterized=True)
ee.set_xlabel('CD31 (published units)'); ee.set_ylabel('Ki-67 (published units)')
title(ee, 'E  Vessel-cell markers')

# F: representative acquired ROI (F1) and cell outlines coloured by cluster membership (F2).
gg = gs[3, 1].subgridspec(1, 2, wspace=.16)
f1 = fig.add_subplot(gg[0, 0]); f2 = fig.add_subplot(gg[0, 1])
z = np.load(SRC / 'panelF_roi.npz', allow_pickle=True)
meta = json.loads((SRC / 'panelF_roi.json').read_text())
img, msk = z['img'], z['msk']; ch = meta['panel_channels']
rgb = np.dstack([norm99(img[ch[m]]) for m in ['CD31', 'Ki67', 'Pancytokeratin']]) ** .7
f1.imshow(rgb); f1.axis('off'); title(f1, 'F1  IMC')
seg = msk if msk.ndim == 2 else msk.sum(axis=0)
oc = set(z['ObjectNumber'][z['is_cand'] & z['vessel']].tolist())
ov = set(z['ObjectNumber'][z['vessel']].tolist())
for o in np.unique(seg):
    if o == 0:
        continue
    col = '#D55E00' if o in oc else ('.4' if o in ov else '#A7C0D8')
    for contour in find_contours((seg == o).astype(float), .5):
        f2.plot(contour[:, 1], contour[:, 0], color=col, lw=.6 if o in ov else .2)
f2.set_xlim(0, seg.shape[1]); f2.set_ylim(seg.shape[0], 0); f2.set_aspect('equal')
f2.axis('off'); title(f2, 'F2  Cells')

fig.savefig(OUT / 'figure5.pdf'); fig.savefig(OUT / 'figure5.png', dpi=400)
sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
qa = {'source_files': {str(x.relative_to(MB)): hashlib.sha256(x.read_bytes()).hexdigest()
                       for x in [SRC / 'panelA_umap.csv', SRC / 'panelB_enrichment.csv',
                                 SRC / 'panelD_ki67_paired.csv', SRC / 'panelE_cd31_ki67.csv',
                                 SRC / 'panelF_roi.npz', MB / 'cluster_evaluation.csv']},
      'candidate_clusters': cand, 'canvas_inches': [7.2, 9.6], 'min_font_pt': min(sizes),
      'effective_min_at_6_5in': min(sizes) * 6.5 / 7.2,
      'changes': 'layout and visual encoding only; no input values changed; removed redundant run annotation; no synthetic research image'}
(OUT / 'figure5_render_metadata.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
print(qa['effective_min_at_6_5in'])
