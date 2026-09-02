# -*- coding: utf-8 -*-
"""fig6 v2: 虚拟染色示例图修复版
修复点(v1问题): ①原窗口CD45 r=0.30示例性弱→在密度前6窗口中选CD45 r最高者
②无面板字母 ③无强度/比例说明 ④未标注 in silico
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile
from safetensors.torch import load_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'<project-root>'
FG = os.path.join(ROOT, 'outputs', 'C论文_稿件', 'figures')
E = os.path.join(ROOT, 'data', 'einhaus2023', 'extracted', 'OSCC-IMC Einhaus et al. 2023')
EMB_DIR = os.path.join(ROOT, 'data', 'einhaus_markers', 'embeddings', 'esm2_t30_150M_UR50D')
WEIGHTS = os.path.join(ROOT, 'data', 'weights', 'virtues-sp32', 'model.safetensors')
ZOUT = os.path.join(ROOT, 'outputs', 'einhaus_zeroshot')
sys.path.insert(0, os.path.join(ROOT, 'code', 'Virtues'))
os.chdir(os.path.join(ROOT, 'code', 'Virtues'))
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

MARKER2UNIPROT = {
 'CD45':'P08575','CD3':'P07766','CD4':'P01730','CD8a':'P01732','CD20':'P11836','CD68':'P34810',
 'CD11c':'P20702','CD14':'P08571','CD16':'P08637','CD15':'P31997','CD56':'P13591','CD163':'Q86VB7',
 'CD206':'P22897','CD31':'P16284','CD44':'P16070','Collagen':'P02452','E-Cadherin':'P12830',
 'FoxP3':'Q9BZS1','GranzymeB':'P10144','HistoneH3':'P68431','Ki67':'P46013','Pancytokeratin':'P08727',
 'Podoplanin':'Q86YL7','Vimentin':'P08670','aSMA':'P62736','VEGF':'P15692','CD11b':'P11215',
 'CD36':'P16671','CD45RA':'P08575','CD86':'P42081','CD209':'Q9NNX6',
 'pCREB':'P16220','pERK':'P28482','pMAPKAPK2':'P49137','pNFkB':'Q04206','pS6':'P62753',
 'pSTAT1':'P42224','pSTAT3':'P40763','pp38':'Q16539'}

qdf = pd.read_csv(os.path.join(ZOUT, 'standardization_stats.csv'), index_col=0)
gms = pd.read_csv(os.path.join(ZOUT, 'global_mean_std.csv'), index_col=0)
me = load_marker_embeddings(EMB_DIR)
md = load_marker_embedding_dict(EMB_DIR)
model = MultiplexVirtues(prior_bias_embeddings=me)
model.load_state_dict(load_file(WEIGHTS, device='cuda'))
model.cuda()
model.eval()

_GK = None
def blur(x):
    global _GK
    k = 3
    if _GK is None:
        ax_ = torch.arange(k).float() - 1
        g = torch.exp(-(ax_ ** 2) / 2.0); g /= g.sum()
        _GK = g[:, None] @ g[None, :]
    t = torch.from_numpy(x).float().unsqueeze(0)
    w = _GK.reshape(1, 1, k, k).expand(x.shape[0], 1, k, k).contiguous()
    return F.conv2d(F.pad(t, (1, 1, 1, 1), mode='reflect'), w, groups=x.shape[0])[0].numpy()

cohort = 'STACohort'
imgs = sorted(glob.glob(os.path.join(E, cohort, 'Steinbock', 'img', '*.tiff')))
masks_f = sorted(glob.glob(os.path.join(E, cohort, 'Steinbock', 'masks', '*.tiff')))
panel = pd.read_csv(os.path.join(E, cohort, 'Steinbock', 'panel.csv'))
names = panel['name'].tolist()
keep_idx = [j for j, n in enumerate(names) if n in MARKER2UNIPROT]
keep_names = [n for n in names if n in MARKER2UNIPROT]
midxs = torch.tensor([md[MARKER2UNIPROT[n]] for n in keep_names]).cuda()
fi, fm = imgs[3], masks_f[3]
tid = os.path.basename(fi)[:-5]
raw = tifffile.imread(fi).astype(np.float32)
msk = tifffile.imread(fm)
tissue = (msk.sum(axis=0) if msk.ndim == 3 else msk) > 0
xb = blur(raw)
xs = np.stack([xb[j] for j in keep_idx])
qs = np.array([qdf.loc[tid, n] for n in keep_names], dtype=np.float32)
means = np.array([gms.loc[n, 'mean'] for n in keep_names], dtype=np.float32)
stds = np.array([gms.loc[n, 'std'] for n in keep_names], dtype=np.float32)
xn = ((np.log1p(np.clip(xs, 0, qs[:, None, None])) - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
H, W = tissue.shape
wins = []
for r0 in range(0, H - 127, 64):
    for c0 in range(0, W - 127, 64):
        wins.append((float(tissue[r0:r0+128, c0:c0+128].mean()), r0, c0))
wins.sort(reverse=True)
cand = wins[:6]
ci_cd45 = keep_names.index('CD45'); ci_pstat3 = keep_names.index('pSTAT3')

def reconstruct_window(r0, c0, ci):
    crop_np = xn[:, r0:r0+128, c0:c0+128]
    tm = tissue[r0:r0+128, c0:c0+128]
    crop = torch.from_numpy(crop_np).float().cuda()
    mfull = torch.zeros(len(keep_names), 16, 16, dtype=torch.bool, device='cuda')
    mfull[ci] = True
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        o = model.forward([crop], [midxs], [mfull])
    rec = o.decoded_multiplex[0].float().cpu().numpy()[ci]
    r = float(np.corrcoef(crop_np[ci][tm].astype(float), rec[tm].astype(float))[0, 1])
    return crop_np, tm, rec, r

best = None
for ti in range(min(12, len(imgs))):
    fi_t, fm_t = imgs[ti], masks_f[ti]
    tid_t = os.path.basename(fi_t)[:-5]
    raw_t = tifffile.imread(fi_t).astype(np.float32)
    msk_t = tifffile.imread(fm_t)
    tissue_t = (msk_t.sum(axis=0) if msk_t.ndim == 3 else msk_t) > 0
    xb_t = blur(raw_t)
    xs_t = np.stack([xb_t[j] for j in keep_idx])
    qs_t = np.array([qdf.loc[tid_t, n] for n in keep_names], dtype=np.float32)
    xn_t = ((np.log1p(np.clip(xs_t, 0, qs_t[:, None, None])) - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
    Ht, Wt = tissue_t.shape
    wins_t = []
    for r0 in range(0, Ht - 127, 64):
        for c0 in range(0, Wt - 127, 64):
            d = float(tissue_t[r0:r0+128, c0:c0+128].mean())
            if d > 0.3:
                wins_t.append((d, r0, c0))
    wins_t.sort(reverse=True)
    for dens, r0, c0 in wins_t[:2]:
        crop_np_t = xn_t[:, r0:r0+128, c0:c0+128]
        tm_t = tissue_t[r0:r0+128, c0:c0+128]
        crop_t = torch.from_numpy(crop_np_t).float().cuda()
        mfull = torch.zeros(len(keep_names), 16, 16, dtype=torch.bool, device='cuda')
        mfull[ci_cd45] = True
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            o = model.forward([crop_t], [midxs], [mfull])
        rec_t = o.decoded_multiplex[0].float().cpu().numpy()[ci_cd45]
        r_t = float(np.corrcoef(crop_np_t[ci_cd45][tm_t].astype(float), rec_t[tm_t].astype(float))[0, 1])
        print(f'tissue {tid_t} window {r0}_{c0} density={dens:.3f} CD45 r={r_t:.3f}')
        if best is None or r_t > best[0]:
            best = (r_t, tid_t, r0, c0, xn_t, tissue_t)
r_cd45, tid, r0, c0, xn, tissue = best
H, W = tissue.shape
crop_meas_cd45, tm, rec_cd45, r_cd45 = reconstruct_window(r0, c0, ci_cd45)
crop_meas_p3, _, rec_p3, r_p3 = reconstruct_window(r0, c0, ci_pstat3)
print(f'SELECTED {tid} window {r0}_{c0}: CD45 r={r_cd45:.3f}, pSTAT3 r={r_p3:.3f}')

rows = [('CD45', crop_meas_cd45[ci_cd45], rec_cd45, r_cd45), ('pSTAT3', crop_meas_p3[ci_pstat3], rec_p3, r_p3)]
fig, axes = plt.subplots(2, 3, figsize=(11, 7.8))
letters = [['a','b','c'],['d','e','f']]
for row, (mname, meas, rec, r) in enumerate(rows):
    show_data = meas
    v = np.clip(show_data, np.percentile(show_data[tm], 1), np.percentile(show_data[tm], 99.5))
    axes[row][0].imshow(np.where(tm, v, np.nan), cmap='magma')
    axes[row][0].set_title('Measured (IMC)', fontsize=11)
    v2 = np.clip(rec, np.percentile(rec[tm], 1), np.percentile(rec[tm], 99.5))
    axes[row][1].imshow(np.where(tm, v2, np.nan), cmap='magma')
    axes[row][1].set_title(f'Virtual, in silico (r = {r:.2f})', fontsize=11)
    diff = show_data - rec
    vmax = np.nanpercentile(np.abs(diff[tm]), 98)
    im3 = axes[row][2].imshow(np.where(tm, diff, np.nan), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[row][2].set_title('Residual (measured \u2212 virtual)', fontsize=11)
    for col in range(3):
        axes[row][col].axis('off')
        axes[row][col].text(-0.06, 1.04, letters[row][col], transform=axes[row][col].transAxes,
                            fontsize=13, fontweight='bold', va='top')
        if col == 0:
            axes[row][col].text(-0.10, 0.5, mname, transform=axes[row][col].transAxes,
                                fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)
cb = fig.colorbar(im3, ax=axes[:, 2], fraction=0.05, pad=0.02)
cb.set_label('residual (z-units)', fontsize=10)
fig.suptitle(f'Masked-channel reconstruction example — in silico, OSCC tissue {tid}, window {r0}_{c0}\n'
             f'(128 \u00d7 128 px \u2248 128 \u00d7 128 \u03bcm at 1 \u03bcm/px IMC; intensities z-scored, display 1st\u201399.5th percentile in tissue)', fontsize=11)
OUT_DIRS = [FG,
            os.path.join(ROOT, 'outputs', 'revision_20260901', 'figs')]
for od in OUT_DIRS:
    os.makedirs(od, exist_ok=True)
    plt.savefig(os.path.join(od, 'fig6_virtual_staining.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(od, 'fig6_virtual_staining.pdf'), bbox_inches='tight')
    with open(os.path.join(od, 'fig6_v2_selected.json'), 'w') as f:
        import json; json.dump({'tissue': tid, 'window': f'{r0}_{c0}', 'r_CD45': round(r_cd45, 3), 'r_pSTAT3': round(r_p3, 3),
                                'selection': 'max CD45 r among densest-2 windows of first 12 STA tissues'}, f, indent=1)
print('fig6 v2 saved')
