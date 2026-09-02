# -*- coding: utf-8 -*-
"""
阶段1a（GPU缓存）：像素null v2 —— 复刻kao2_marker_matrix.py主分析的选窗与标准化（隔4取1组织×密度最高2窗=72窗），
逐窗逐标记整通道抹除重建，缓存标准化crop/组织mask/VirTues重建，供置换检验与轻量基线使用。
锁定规范：shift bank K=40（dr,dc∈[8,120)）、B=9999、交换单位=window、add-one P、BH-FDR——见pn2_analysis.py。
产出: cache/*.npz + cache_manifest.json
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile
from safetensors.torch import load_file

ROOT = r'<project-root>'
E = os.path.join(ROOT, 'data', 'einhaus2023', 'extracted', 'OSCC-IMC Einhaus et al. 2023')
EMB_DIR = os.path.join(ROOT, 'data', 'einhaus_markers', 'embeddings', 'esm2_t30_150M_UR50D')
WEIGHTS = os.path.join(ROOT, 'data', 'weights', 'virtues-sp32', 'model.safetensors')
ZOUT = os.path.join(ROOT, 'outputs', 'einhaus_zeroshot')
OUT = os.path.join(ROOT, 'outputs', 'revision_20260901', 'pixel_null_v2')
CACHE = os.path.join(OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, 'code', 'Virtues'))
os.chdir(os.path.join(ROOT, 'code', 'Virtues'))
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

MARKER2UNIPROT = {
 'CD11b':'P11215','CD11c':'P20702','CD14':'P08571','CD15':'P31997','CD16':'P08637',
 'CD163':'Q86VB7','CD20':'P11836','CD206':'P22897','CD209':'Q9NNX6','CD3':'P07766',
 'CD31':'P16284','CD36':'P16671','CD4':'P01730','CD44':'P16070','CD45':'P08575',
 'CD45RA':'P08575','CD56':'P13591','CD68':'P34810','CD86':'P42081','CD8a':'P01732',
 'Collagen':'P02452','E-Cadherin':'P12830','FoxP3':'Q9BZS1','GranzymeB':'P10144',
 'HistoneH3':'P68431','Ki67':'P46013','Pancytokeratin':'P08727','Podoplanin':'Q86YL7',
 'VEGF':'P15692','Vimentin':'P08670','aSMA':'P62736','pCREB':'P16220','pERK':'P28482',
 'pMAPKAPK2':'P49137','pNFkB':'Q04206','pS6':'P62753','pSTAT1':'P42224','pSTAT3':'P40763','pp38':'Q16539'}

qdf = pd.read_csv(os.path.join(ZOUT, 'standardization_stats.csv'), index_col=0)
gms = pd.read_csv(os.path.join(ZOUT, 'global_mean_std.csv'), index_col=0)
me = load_marker_embeddings(EMB_DIR)
md = load_marker_embedding_dict(EMB_DIR)
model = MultiplexVirtues(prior_bias_embeddings=me)
model.load_state_dict(load_file(WEIGHTS, device='cuda'))
model = model.cuda().eval()

_GK = None
def blur(x):
    global _GK
    k = 3
    if _GK is None:
        ax = torch.arange(k).float() - 1
        g = torch.exp(-(ax ** 2) / 2.0); g /= g.sum()
        _GK = g[:, None] @ g[None, :]
    t = torch.from_numpy(x).float().unsqueeze(0)
    w = _GK.reshape(1, 1, k, k).expand(x.shape[0], 1, k, k).contiguous()
    return F.conv2d(F.pad(t, (1, 1, 1, 1), mode='reflect'), w, groups=x.shape[0])[0].numpy()

t0 = time.time()
manifest = []
n_win = 0
for cohort in ['UOPCohort', 'STACohort']:
    imgs = sorted(glob.glob(os.path.join(E, cohort, 'Steinbock', 'img', '*.tiff')))
    masks_f = sorted(glob.glob(os.path.join(E, cohort, 'Steinbock', 'masks', '*.tiff')))
    panel = pd.read_csv(os.path.join(E, cohort, 'Steinbock', 'panel.csv'))
    names = panel['name'].tolist()
    keep_idx = [j for j, n in enumerate(names) if n in MARKER2UNIPROT]
    keep_names = [n for n in names if n in MARKER2UNIPROT]
    midxs = torch.tensor([md[MARKER2UNIPROT[n]] for n in keep_names]).cuda()
    for k, (fi, fm) in enumerate(zip(imgs, masks_f)):
        if k % 4 != 0:
            continue
        tid = os.path.basename(fi)[:-5]
        raw = tifffile.imread(fi).astype(np.float32)
        msk = tifffile.imread(fm)
        tissue = (msk.sum(axis=0) if msk.ndim == 3 else msk) > 0
        xb = blur(raw)
        xs = np.stack([xb[j] for j in keep_idx])
        try:
            qs = np.array([qdf.loc[tid, n] for n in keep_names], dtype=np.float32)
        except KeyError:
            continue
        means = np.array([gms.loc[n, 'mean'] for n in keep_names], dtype=np.float32)
        stds = np.array([gms.loc[n, 'std'] for n in keep_names], dtype=np.float32)
        xn = ((np.log1p(np.clip(xs, 0, qs[:, None, None])) - means[:, None, None]) / stds[:, None, None]).astype(np.float32)
        H, W = tissue.shape
        if H < 128 or W < 128:
            continue
        wins = []
        for r0 in range(0, H - 127, 64):
            for c0 in range(0, W - 127, 64):
                wins.append((float(tissue[r0:r0+128, c0:c0+128].mean()), r0, c0))
        wins.sort(reverse=True)
        for dens, r0, c0 in wins[:2]:
            key = f"{cohort[:3]}_{tid}_w{r0}_{c0}"
            cpath = os.path.join(CACHE, key + '.npz')
            if os.path.exists(cpath):
                # 幂等修复(v5.2): 命中已有缓存时也回填manifest, 避免补跑覆盖出不完整清单
                z = np.load(cpath, allow_pickle=True)
                manifest.append({'key': key, 'cohort': 'UOP' if 'UOP' in cohort else 'STA',
                                 'tissue': tid, 'r0': r0, 'c0': c0,
                                 'density': round(float(z['tm'].mean()), 4),
                                 'n_markers': int(z['crop'].shape[0])})
                n_win += 1
                continue
            crop = torch.from_numpy(xn[:, r0:r0+128, c0:c0+128]).float().cuda()
            tm = tissue[r0:r0+128, c0:c0+128]
            rec_all = np.zeros_like(xn[:, r0:r0+128, c0:c0+128])
            for ci in range(len(keep_names)):
                mfull = torch.zeros(len(keep_names), 16, 16, dtype=torch.bool, device='cuda')
                mfull[ci] = True
                with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                    o = model.forward([crop], [midxs], [mfull])
                rec_all[ci] = o.decoded_multiplex[0].float().cpu().numpy()[ci]
            np.savez_compressed(cpath, crop=crop.cpu().numpy(), rec=rec_all, tm=tm,
                                names=np.array(keep_names))
            manifest.append({'key': key, 'cohort': 'UOP' if 'UOP' in cohort else 'STA',
                             'tissue': tid, 'r0': r0, 'c0': c0, 'density': round(float(dens), 4),
                             'n_markers': len(keep_names)})
            n_win += 1
            print(f"{key} done ({n_win} wins, {time.time()-t0:.0f}s)", flush=True)

json.dump({'n_windows': n_win, 'generated': 'pn2_cache.py',
           'weights_sha256_expected': '974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7',
           'selection_rule': '复刻kao2_marker_matrix.py: 每队列sorted顺序k%4==0的组织×组织密度最高2个128窗',
           'standardization': 'q99 clip(组织级)+log1p+全局mean/std (einhaus_zeroshot/standardization_stats.csv+global_mean_std.csv)',
           'windows': manifest},
          open(os.path.join(OUT, 'cache_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"CACHE_DONE n_windows={n_win} elapsed={time.time()-t0:.0f}s")
