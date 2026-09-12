"""Render revision-stage Figures 1-4 and 6 (v5.13 publication numbering) from fixed inputs.

Rewritten 2026-09-12: restructuring, naming and
documentation only. Every plotting call, parameter value and draw order is preserved
from the previous version, so outputs are pixel-identical (verified by raster
comparison at submission width). No new inference or statistical fits are performed.

Inputs (read-only, fixed result tables/caches under outputs/revision_20260901):
  figs/fig2_v2_source_data.csv, figs/fig2_original_umap.png,
  figs/fig4_v2_source_data.csv, figs/fig6_fixed_window.npz,
  pixel_null_v2/permutation_results.csv, pixel_null_v2/baselines_r.csv
Outputs: figs/publication/figure{1,2,3,4,6}.{pdf,png} + render_metadata.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'outputs' / 'revision_20260901'
FIGS = RESULTS / 'figs'
OUTDIR = FIGS / 'publication'
OUTDIR.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42})

qa = {}


def save_figure(fig, number):
    """Write one figure as PDF+PNG and record font-size QA metrics."""
    fig.savefig(OUTDIR / f'figure{number}.pdf')
    fig.savefig(OUTDIR / f'figure{number}.png', dpi=400)
    sizes = [x.get_fontsize() for x in fig.findobj(Text) if x.get_visible() and x.get_text()]
    qa[str(number)] = {
        'width_in': float(fig.get_size_inches()[0]),
        'min_font_pt': min(sizes),
        'effective_min_at_6_5in': min(sizes) * 6.5 / fig.get_size_inches()[0],
    }
    plt.close(fig)


def render_figure1_overview():
    """Audit-design overview: direct claim/control relationships, not a hierarchy of proof."""
    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    fig.subplots_adjust(left=.025, right=.975, bottom=.02, top=.99)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, s, fill='#ECF2F8', bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=fill, ec='#61768C', lw=.8))
        ax.text(x + w / 2, y + h / 2, s, ha='center', va='center', fontsize=9,
                fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={'arrowstyle': '->', 'color': '.35', 'lw': .8})

    box(.2, 8.45, 9.6, 1.3,
        'Primary OSCC study: UOP + STA\n48 patients / 142 ROIs / 498,238 annotated cells / 39 markers\n'
        'Context: 2 SCCHN samples and 5 descriptive cross-cancer cohorts', bold=True)
    box(.2, 6.6, 9.6, 1.25,
        'Frozen virtues-sp32: checkpoint hash and marker identities recorded\n'
        'Blur \u2192 uq0.99 clipping \u2192 log1p \u2192 pooled image z-score\n'
        'Global normalization is transductive; cohort lists do not establish sample exclusion')
    arrow(5, 8.4, 5, 7.9)
    headers = ['Task 1\nCell phenotyping', 'Task 2\nChannel reconstruction', 'Task 3\nToken clustering']
    texts = ['Matched patient folds\nToken / intensity / fusion\n\nLabels derived from\nintensity measurements',
             '72 windows / 36 patients\n39 channels individually\nmasked\n\nSpatial null + ridge\nLOWO / LOPO + paired CI',
             'UOP fitting; STA assignment\nMulti-seed stability\n\nPatient-paired Ki-67\nExploratory recurrence']
    for i in range(3):
        x = .2 + 3.3 * i
        box(x, 5.25, 3, .8, headers[i], bold=True)
        box(x, 2.85, 3, 2.1, texts[i])
        arrow(x + 1.5, 6.55, x + 1.5, 6.1)
        arrow(x + 1.5, 5.2, x + 1.5, 5)
        arrow(x + 1.5, 2.8, x + 1.5, 2.3)
    box(.2, .75, 9.6, 1.45,
        'Outputs: per-marker differences, uncertainty and claim-to-evidence map\n'
        'Spatial alignment \u2260 incremental performance \u2260 patient-disjoint validation\n'
        'Unstable partition and negative Ki-67 findings retained\n'
        'Reusable protocol and executable result summaries', fill='#EFF5EC', bold=True)
    save_figure(fig, 1)


def render_figure2_phenotyping():
    """Phenotyping panel: preserve the original UMAP raster, replot only published F1 data."""
    d = pd.read_csv(FIGS / 'fig2_v2_source_data.csv')
    order = d.cell_type.tolist()
    colors = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(1, 2, left=.03, right=.98, bottom=.26, top=.88,
                          width_ratios=[1.2, 1], wspace=.40)
    a = fig.add_subplot(gs[0, 0])
    a.imshow(plt.imread(FIGS / 'fig2_original_umap.png')); a.axis('off')
    a.set_title('A  Frozen cell tokens', loc='left', fontweight='bold')
    handles = [Line2D([], [], marker='o', ls='', color=colors(i), markersize=4, label=s)
               for i, s in enumerate(order)]
    a.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.03), ncol=2,
             frameon=False, fontsize=9, columnspacing=.5, handletextpad=.4)
    b = fig.add_subplot(gs[0, 1])
    y = np.arange(8)[::-1]
    b.barh(y, d.f1_celllevel, color=[colors(i) for i in range(8)])
    b.set_yticks(y); b.set_yticklabels(order); b.set_xlim(0, 1.14); b.set_xticks([0, .5, 1])
    b.set_xlabel('Per-class F1 (cell-level split)')
    b.set_title('B  Cell phenotyping', loc='left', fontweight='bold')
    b.axvline(d.f1_celllevel.mean(), color='.3', ls='--', lw=.8, zorder=0)
    for yy, v in zip(y, d.f1_celllevel):
        b.text(v + .015, yy, f'{v:.3f}', va='center', fontsize=9,
               bbox={'fc': 'white', 'ec': 'none', 'pad': .2})
    fig.text(.98, .04, 'Dashed line: cell-level macro-F1 0.756', ha='right', fontsize=9)
    save_figure(fig, 2)


def render_figure3_capability_map():
    """Reconstruction capability map: horizontal rows keep marker labels readable at width."""
    d = pd.read_csv(RESULTS / 'pixel_null_v2/permutation_results.csv') \
        .merge(pd.read_csv(RESULTS / 'pixel_null_v2/baselines_r.csv')[['marker', 'r_ridge']],
               on='marker', validate='one_to_one') \
        .sort_values('T_obs_mean_r', ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    fig.subplots_adjust(left=.26, right=.97, bottom=.075, top=.94)
    y = np.arange(len(d))
    ax.barh(y, d.T_obs_mean_r, color='#0072B2', height=.68, label='Frozen VirTues')
    ax.errorbar(d.null_mean, y, xerr=1.96 * d.null_sd, fmt='|', color='.35', elinewidth=.6,
                markersize=3, label='Spatial null mean \u00b1 1.96 SD')
    ax.scatter(d.r_ridge, y, marker='|', s=100, color='#D55E00', linewidths=1.3, label='LOWO ridge')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{m} ({n})' for m, n in zip(d.marker, d.n_windows)])
    ax.invert_yaxis(); ax.axvline(0, color='.5', lw=.6); ax.set_xlim(-.06, .68)
    ax.set_xlabel('Window-mean Pearson r')
    ax.set_title('Marker reconstruction and comparators', loc='left', fontweight='bold')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    save_figure(fig, 3)


def render_figure4_scchn_control():
    """Descriptive same-pipeline SCCHN control: same values and bootstrap endpoints, compact labels."""
    d = pd.read_csv(FIGS / 'fig4_v2_source_data.csv')
    ms = d.marker.unique()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(left=.10, right=.98, bottom=.14, top=.86)
    for i, m in enumerate(ms):
        q = d[d.marker == m]
        # 源 CSV 列名本身不对称（osci_lo / oscc_hi），保持逐字以维持输出等价
        v = q.oscc_mean.iloc[0]; lo = q.osci_lo.iloc[0]; hi = q.oscc_hi.iloc[0]
        ax.bar(i - .18, v, .32, color='#CC6677', yerr=[[v - lo], [hi - v]], capsize=3,
               label='OSCC mean (ROI bootstrap 95% CI)' if i == 0 else None)
        ax.scatter(np.full(len(q), i + .18), q.r, color='#117733', marker='o',
                   edgecolor='black', lw=.4, label='SCCHN individual samples' if i == 0 else None)
        ax.plot([i + .10, i + .26], [q.r.mean()] * 2, color='#117733', lw=1.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(ms); ax.set_ylim(0, 1.03)
    ax.set_ylabel('Full-channel reconstruction r')
    ax.set_title('Descriptive same-pipeline cohort comparison', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    save_figure(fig, 4)


def render_figure6_example_window():
    """Masked-channel reconstruction example: fixed cached predictions; per-residual matched scale."""
    z = np.load(FIGS / 'fig6_fixed_window.npz', allow_pickle=True)
    names = [str(x) for x in z['names']]
    tm = z['tm'].astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=.07, right=.94, bottom=.08, top=.91, wspace=.25, hspace=.40)
    details = {}
    for row, m in enumerate(['CD45', 'pSTAT3']):
        i = names.index(m); meas = z['crop'][i]; rec = z['rec'][i]
        r = float(np.corrcoef(meas[tm].astype(float), rec[tm].astype(float))[0, 1])
        assert round(r, 2) == ([.83, .09][row])
        for col, a in enumerate([meas, rec]):
            lo, hi = np.percentile(a[tm], [1, 99.5])
            axes[row, col].imshow(np.where(tm, np.clip(a, lo, hi), np.nan), cmap='magma', vmin=lo, vmax=hi)
        diff = meas - rec
        lim = float(np.percentile(np.abs(diff[tm]), 98))
        im = axes[row, 2].imshow(np.where(tm, diff, np.nan), cmap='RdBu_r', vmin=-lim, vmax=lim)
        cb = fig.colorbar(im, ax=axes[row, 2], fraction=.055, pad=.03)
        cb.ax.tick_params(labelsize=9)
        for col, label in enumerate(['Measured IMC', f'Virtual r = {r:.2f}', 'Residual']):
            axes[row, col].set_title(f'{chr(65 + row * 3 + col)}  {label}', fontsize=9, loc='left')
            axes[row, col].axis('off')
        axes[row, 0].text(-.08, .5, m, rotation=90, ha='right', va='center',
                          transform=axes[row, 0].transAxes, fontweight='bold', fontsize=9)
        details[m] = {'r': r, 'residual_symmetric_limit': lim}
    fig.text(.5, .97, 'Masked-channel reconstruction in silico', ha='center', fontsize=11, fontweight='bold')
    fig.text(.5, .025, 'Same 128 \u00d7 128 \u00b5m window; residual scales in z-score units',
             ha='center', fontsize=9)
    save_figure(fig, 6)
    qa['6']['fixed_window'] = 'STA_S02_2_w320_384'
    qa['6']['metrics'] = details


if __name__ == '__main__':
    render_figure1_overview()
    render_figure2_phenotyping()
    render_figure3_capability_map()
    render_figure4_scchn_control()
    render_figure6_example_window()
    (OUTDIR / 'render_metadata.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))
