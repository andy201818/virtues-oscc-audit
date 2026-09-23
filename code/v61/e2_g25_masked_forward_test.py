#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G2 condition 5: runnability mini-test on REAL CODEX data (Core_A5).

Checks: (a) I/O decode timing, (b) streaming encoder VRAM + crops/s on a real
1500x1500 region, (c) target-masked full forward on a real 128x128 window
(demo1 mask convention: patch-level mask, True=masked, full channel),
(d) output shape/finiteness. Mini-test only: region-level log1p+robust-z here
is NOT the frozen E2 normalization protocol.

Output: outputs/e2/gating/g25_runnability.json
"""
import json
import os
import sys
import time

ROOT = r"<data-root>\virtual cell and tissue"
sys.path.insert(0, os.path.join(ROOT, "code", "Virtues"))
os.chdir(os.path.join(ROOT, "code", "Virtues"))
sys.path.insert(0, r"<project-root>\scripts")

import numpy as np
import torch
import tifffile

from e0_streaming_adapter import load_model, EMB_DIR, DEFAULTS
from virtues.utils.utils import load_marker_embedding_dict

IMG = r"<project-root>\external_data\S-BIAD3612\images\cores\Core_A5.ome.tif"
OUT = r"<project-root>\outputs\e2\gating\g25_runnability.json"

# published-table markers whose UniProts exist in the local ESM embedding set.
# unambiguous overlap only (six targets + shared lineage/ECM markers);
# ambiguous glycan epitopes (CD15) and absent UniProts are excluded on purpose.
MARKER2CH = {  # CODEX channel name -> image channel index (from channel_map)
    "CD45": 69, "CD31": 41, "CD68": 53, "CD163": 32, "Ki67": 29, "aSMA": 6,
    "CD3": 72, "CD8": 40, "CD14": None, "Vimentin": 49, "PanCK": 50,
    "Podoplanin": 9, "GranzymeB": 54,
}
MARKER2UNIPROT = {
    "CD45": "P08575", "CD31": "P16284", "CD68": "P34810", "CD163": "Q86VB7",
    "Ki67": "P46013", "aSMA": "P62736", "CD3": "P07766", "CD8": "P01732",
    "Vimentin": "P08670", "PanCK": "P08727", "Podoplanin": "Q86YL7",
    "GranzymeB": "P10144",
}


def main():
    rep = {}
    md = load_marker_embedding_dict(EMB_DIR)
    markers = [(m, MARKER2CH[m], md[u]) for m, u in MARKER2UNIPROT.items()
               if MARKER2CH.get(m) is not None and u in md]
    rep["channels_used"] = [(m, ch, emb) for m, ch, emb in markers]
    print(f"usable marker channels: {len(markers)} / attempted {len(MARKER2UNIPROT)}")

    # ---- (a) I/O decode timing on real pages ----
    t0 = time.time()
    decoded = {}
    with tifffile.TiffFile(IMG) as tf:
        h, w = tf.series[0].shape[1:]
        for m, ch, emb in markers:
            decoded[m] = tf.pages[ch].asarray().astype(np.float32)
    t_decode = time.time() - t0
    rep["io"] = {"pages_decoded": len(decoded), "total_s": round(t_decode, 2),
                 "s_per_page": round(t_decode / len(decoded), 3),
                 "dtype": "uint16->float32", "page_shape": [h, w]}
    print("decode:", rep["io"])

    # ---- region for streaming encoder ----
    R, C0 = (h - 1500) // 2, (w - 1500) // 2
    region = np.stack([decoded[m][R:R + 1500, C0:C0 + 1500] for m, _, _ in markers])  # (n,1500,1500)
    del decoded
    # mini-test standardization only (NOT the frozen protocol): log1p + robust z
    region = np.log1p(region)
    med = np.median(region, axis=(1, 2), keepdims=True)
    q75, q25 = np.percentile(region, [75, 25], axis=(1, 2), keepdims=True)
    region = (region - med) / np.maximum(q75 - q25, 1e-6)
    region = region.astype(np.float32)  # np.percentile promotes to float64; autocast needs fp32
    img_t = torch.from_numpy(region)
    ch_ids = torch.tensor([emb for _, _, emb in markers])

    model = load_model()

    # ---- (b) streaming encoder on real region: VRAM + crops/s ----
    from virtues.utils import cell_tokens as ct
    crops, indices = ct._get_uniform_crops(img_t, DEFAULTS["stride"], tile_size=DEFAULTS["tile_size"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for i in range(0, len(crops), DEFAULTS["chunk_size"]):
            chunk = [c.cuda() for c in crops[i:i + DEFAULTS["chunk_size"]]]
            with torch.amp.autocast("cuda"):
                out = model.encoder.forward_list(chunk, [ch_ids.cuda()] * len(chunk), multiplex_mask=None)
            n += len(chunk)
            del chunk, out
    torch.cuda.synchronize()
    t_enc = time.time() - t0
    rep["streaming_encoder_real"] = {
        "region": "1500x1500", "n_channels": len(markers), "n_crops": len(crops),
        "total_s": round(t_enc, 2), "crops_per_s": round(n / t_enc, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    print("encoder:", rep["streaming_encoder_real"])

    # ---- (c) masked full forward on a real 128x128 window ----
    wy, wx = R + 750 - 64, C0 + 750 - 64  # center window of the region
    win = img_t[:, 750 - 64:750 + 64, 750 - 64:750 + 64].clone()
    P = DEFAULTS["patch_size"]
    # demo1 convention: patch-level mask (C, H/8, W/8); True = masked
    mask = torch.zeros(len(markers), 128 // P, 128 // P, dtype=torch.bool)
    target_local = [m for m, _, _ in markers].index("CD68")
    mask[target_local] = True
    mask_l = [mask]
    ids_l = [ch_ids]
    # timing: warmup + repeated s measurement
    times = []
    with torch.no_grad():
        for k in range(6):
            t0 = time.time()
            with torch.amp.autocast("cuda"):
                out = model([win.cuda()], [ids_l[0].cuda()], [mask_l[0].cuda()])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    rec = out.decoded_multiplex[0].float().cpu()  # (C, 128, 128)
    rep["masked_forward"] = {
        "window": "128x128 native CODEX px", "n_channels": len(markers),
        "target": "CD68 (channel idx 53, local %d)" % target_local,
        "output_shape": list(rec.shape),
        "output_finite": bool(torch.isfinite(rec).all().item()),
        "repeated_s": [round(t, 3) for t in times],
        "median_s": round(float(np.median(times)), 4),
        "target_fully_masked": bool(mask[target_local].all().item()),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    print("masked forward:", rep["masked_forward"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("written:", OUT)
    ok = (rep["masked_forward"]["output_finite"]
          and rep["masked_forward"]["median_s"] < 30
          and rep["streaming_encoder_real"]["peak_gpu_gb"] < 11)
    print("G2-5 RUNNABILITY", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
