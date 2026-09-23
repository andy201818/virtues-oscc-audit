#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0/S1-a streaming adapter for VirTues inference (RTX 5060 Ti 16GB).

Purpose: replace virtues.utils.cell_tokens.compute_cell_tokens with a
memory-safe version. The library path moves ALL crops to GPU at once
(cell_tokens.py:80, `[c.to(device) for c in crops]`); this adapter keeps
crops on CPU and moves one chunk at a time.

Semantics are bit-identical BY CONSTRUCTION:
  - crop grid/edge-completion/sorted order: reuses library _get_uniform_crops
  - chunk batching boundaries: identical chunk_size
  - autocast('cuda') + no_grad: identical
  - patch->cell assignment + pixel-count weighting: reuses library function

Self-test (--selftest): old vs new on seeded synthetic data,
  - determinism baseline (old run twice)
  - bit-equality or exact max-abs-diff
  - peak GPU memory old vs new
  - per-crop timing (feeds the s estimate)

Run with the project venv:
  <data-root>/virtual cell and tissue/code/venv/Scripts/python.exe \
      <project-root>/scripts/e0_streaming_adapter.py --selftest
"""
import argparse
import json
import os
import sys
import time
from typing import List, Tuple

ROOT = r"<data-root>\virtual cell and tissue"
sys.path.insert(0, os.path.join(ROOT, "code", "Virtues"))
os.chdir(os.path.join(ROOT, "code", "Virtues"))

import torch

from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils import cell_tokens as ct
from virtues.utils.utils import load_marker_embeddings

EMB_DIR = os.path.join(ROOT, "data", "einhaus_markers", "embeddings", "esm2_t30_150M_UR50D")
WEIGHTS = os.path.join(ROOT, "data", "weights", "virtues-sp32", "model.safetensors")

# default analysis parameters are frozen exactly as in the library signature
DEFAULTS = dict(tile_size=128, patch_size=8, stride=42, chunk_size=32)


def load_model(weights: str = WEIGHTS, emb_dir: str = EMB_DIR, device: str = "cuda") -> MultiplexVirtues:
    from safetensors.torch import load_file
    me = load_marker_embeddings(emb_dir)
    model = MultiplexVirtues(prior_bias_embeddings=me)
    model.load_state_dict(load_file(weights, device=device))
    return model.to(device).eval()


def compute_cell_tokens_streaming(
    model: MultiplexVirtues,
    img: torch.Tensor,
    channel: torch.Tensor,
    segmentation_mask: torch.Tensor,
    device: str = "cuda",
    tile_size: int = 128,
    patch_size: int = 8,
    stride: int = 42,
    chunk_size: int = 32,
):
    """Drop-in replacement for virtues.utils.cell_tokens.compute_cell_tokens.

    Identical signature, identical return (cell_ids, cell_tokens, crop_tokens,
    indices). Only memory behavior differs: crops stay on CPU; one chunk of
    crops (plus the channel tensor) is moved to GPU per iteration and the
    patch summary tokens are moved back to CPU immediately.
    """
    # library function: same grid, edge completion, sorted order (views only)
    crops, indices = ct._get_uniform_crops(img, stride, tile_size=tile_size)
    channel = channel.to(device)

    crop_tokens = []
    for i in range(0, len(crops), chunk_size):
        crops_chunk = [c.to(device) for c in crops[i : i + chunk_size]]  # <-- the fix
        channels_chunk = [channel for _ in range(len(crops_chunk))]
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                encoder_output = model.encoder.forward_list(
                    crops_chunk,
                    channels_chunk,
                    multiplex_mask=None,
                )
                pss = [ps.cpu() for ps in encoder_output.patch_summary_tokens]
                crop_tokens.extend(pss)
        del crops_chunk, channels_chunk  # free GPU copies before next chunk

    cell_tokens = {}
    weights = {}
    for crop_token, (row, col) in zip(crop_tokens, indices):
        crop_mask = segmentation_mask[row : row + tile_size, col : col + tile_size]
        crop_cell_tokens, crop_weights = ct._assign_patch_tokens_to_cells(
            crop_token, crop_mask, patch_size
        )
        for cell_id, tokens in crop_cell_tokens.items():
            if cell_id not in cell_tokens:
                cell_tokens[cell_id] = []
            if cell_id not in weights:
                weights[cell_id] = []
            cell_tokens[cell_id].extend(tokens)
            weights[cell_id].extend(crop_weights[cell_id])

    avg_cell_tokens = {}
    for cell_id, tokens in cell_tokens.items():
        tokens = torch.stack(tokens)
        weights_array = torch.tensor(weights[cell_id], dtype=tokens.dtype, device=tokens.device)
        weights_array = weights_array / weights_array.sum()
        avg_cell_tokens[cell_id] = torch.sum(tokens * weights_array[:, None], dim=0)
    cell_ids = sorted(avg_cell_tokens.keys())
    cell_tokens_t = torch.stack([avg_cell_tokens[cell_id] for cell_id in cell_ids])
    cell_ids_t = torch.tensor(cell_ids, dtype=torch.long)
    return cell_ids_t.cpu(), cell_tokens_t.cpu(), crop_tokens, indices


def forward_reconstruction_chunked(
    model: MultiplexVirtues,
    img: torch.Tensor,
    channel: torch.Tensor,
    multiplex_mask: torch.Tensor,
    device: str = "cuda",
    tile_size: int = 128,
    stride: int = 42,
    chunk_size: int = 8,
):
    """Chunked full forward (encoder+decoder) with multiplex_mask.

    For Task-2 style masked-target reconstruction. Returns decoded_multiplex
    list per crop (on CPU) plus indices, mirroring model.forward semantics
    while bounding GPU memory to one chunk of crops.
    """
    crops, indices = ct._get_uniform_crops(img, stride, tile_size=tile_size)
    # mask must travel with each crop: mask is (C,H,W) aligned with img
    masks, _ = ct._get_uniform_crops(multiplex_mask, stride, tile_size=tile_size)
    channel = channel.to(device)
    decoded = []
    for i in range(0, len(crops), chunk_size):
        crops_chunk = [c.to(device) for c in crops[i : i + chunk_size]]
        masks_chunk = [m.to(device) for m in masks[i : i + chunk_size]]
        channels_chunk = [channel for _ in range(len(crops_chunk))]
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                out = model(crops_chunk, channels_chunk, multiplex_mask=masks_chunk)
        decoded.extend([d.cpu() for d in out.decoded_multiplex])
        del crops_chunk, masks_chunk, channels_chunk, out
    return decoded, indices


# ---------------------------------------------------------------- self-test

def _synthetic(C: int, H: int, W: int, n_cells: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(C, H, W, generator=g)
    channel = torch.arange(C)
    mask = torch.zeros(H, W, dtype=torch.long)
    for cid in range(1, n_cells + 1):
        y = int(torch.randint(0, H - 12, (1,), generator=g))
        x = int(torch.randint(0, W - 12, (1,), generator=g))
        mask[y : y + 10, x : x + 10] = cid
    return img, channel, mask


def _peak_mem():
    return torch.cuda.max_memory_allocated() / 1024**3


def selftest(small=(320, 320, 8, 60), big=(1500, 1500, 38, 400), out_json=None):
    # NOTE: channel ids index the 38-entry ESM embedding table (einhaus markers);
    # synthetic channel=arange(C) requires C <= 38 or the gather asserts.
    report = {"determinism": None, "small": {}, "big": {}, "timing": {}}
    model = load_model()
    print("model loaded (sp32).")

    # ---------- small: old vs new ----------
    H, W, C, nc = small
    img, channel, mask = _synthetic(C, H, W, nc, seed=20260920)
    args = dict(img=img, channel=channel, segmentation_mask=mask, device="cuda", **DEFAULTS)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ids_a, tok_a, _, idx_a = ct.compute_cell_tokens(model, **args)
    torch.cuda.synchronize()
    t_old_1 = time.time() - t0
    mem_old_1 = _peak_mem()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ids_b, tok_b, _, idx_b = ct.compute_cell_tokens(model, **args)
    torch.cuda.synchronize()
    t_old_2 = time.time() - t0
    det = bool(torch.equal(ids_a, ids_b) and torch.equal(tok_a, tok_b))
    report["determinism"] = {
        "old_run1_vs_run2_bit_identical": det,
        "t_old_1_s": round(t_old_1, 3),
        "t_old_2_s": round(t_old_2, 3),
    }
    print(f"old determinism (run1==run2 bitwise): {det}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ids_n, tok_n, _, idx_n = compute_cell_tokens_streaming(model, **args)
    torch.cuda.synchronize()
    t_new = time.time() - t0
    mem_new = _peak_mem()

    bit_eq = bool(torch.equal(ids_a, ids_n) and torch.equal(tok_a, tok_n))
    max_diff = float((tok_a.float() - tok_n.float()).abs().max()) if tok_a.shape == tok_n.shape else None
    report["small"] = {
        "img": f"{C}x{H}x{W}", "n_crops": len(idx_a),
        "indices_identical": idx_a == idx_n,
        "cell_ids_bit_identical": bool(torch.equal(ids_a, ids_n)),
        "tokens_bit_identical": bit_eq,
        "tokens_max_abs_diff": max_diff,
        "n_cells_old": int(ids_a.numel()), "n_cells_new": int(ids_n.numel()),
        "peak_gpu_gb_old": round(mem_old_1, 3), "peak_gpu_gb_new": round(mem_new, 3),
        "t_old_s": round(t_old_1, 3), "t_new_s": round(t_new, 3),
    }
    print(json.dumps(report["small"], indent=1))

    # ---------- big: memory behavior (new only if old would be risky) ----------
    H, W, C, nc = big
    img2, channel2, mask2 = _synthetic(C, H, W, nc, seed=20260921)
    args2 = dict(img=img2, channel=channel2, segmentation_mask=mask2, device="cuda", **DEFAULTS)
    est_old_crops_gb = (
        (len(ct._get_uniform_crops(img2, 42, tile_size=128)[0]) * C * 128 * 128 * 4) / 1024**3
    )
    report["big"]["estimated_gpu_gb_just_for_crops_old_path"] = round(est_old_crops_gb, 2)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ids_b2, tok_b2, _, idx_b2 = compute_cell_tokens_streaming(model, **args2)
    torch.cuda.synchronize()
    t_big = time.time() - t0
    report["big"].update({
        "img": f"{C}x{H}x{W}", "n_crops": len(idx_b2), "n_cells": int(ids_b2.numel()),
        "peak_gpu_gb_new": round(_peak_mem(), 3),
        "t_new_s": round(t_big, 3),
        "crops_per_s": round(len(idx_b2) / t_big, 2),
    })
    print(json.dumps(report["big"], indent=1))

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("written:", out_json)
    ok = bit_eq and idx_a == idx_n
    print("SELFTEST", "PASS" if ok else "FAIL")
    return report, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=r"<project-root>\outputs\e0\streaming_regression.json")
    a = ap.parse_args()
    if a.selftest:
        selftest(out_json=a.out)
    else:
        print("nothing to do; use --selftest")
