#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0/S1-c: operator-level attention parity for the varlen fallback path.

Target: MHAwithPosEmb Branch B fallback (attention_flashattention.py L258-293),
used on this machine because flash_attn is unavailable.

Reference: per-sequence, unpadded, FP64 plain attention, computed manually
from the module's own weights (W_q/W_k/W_v/RoPE/W_o). The fallback pads
variable-length sequences with zeros and applies a block-diagonal + key-valid
mask in a single SDPA call; parity = valid-row outputs equal the reference.

Cases (per enhancement plan 3.2):
  A len-1 sequence only
  B different lengths (1,5,9,13)
  C padding stress (short + one long sequence)
  D reordering invariance (same sequences, different order -> outputs reorder)
  E length-0 sequence inside cu_seq_len (masked-target degenerate case)
  F total length 0 (documented reject/OK behavior)

Dtypes: FP64 ref vs FP32 fallback (atol 1e-5 / rtol 1e-4 screening start);
FP64 ref vs autocast-fp16 fallback (atol 1e-3 / rtol 1e-2). Backend check:
sdpa MATH vs default backends on the FP32 fallback.

Output: <project-root>/outputs/e0/attention_parity.csv
Run: <project venv python> scripts/e0_attention_parity.py
"""
import csv
import math
import os
import sys

ROOT = r"<data-root>\virtual cell and tissue"
sys.path.insert(0, os.path.join(ROOT, "code", "Virtues"))
os.chdir(os.path.join(ROOT, "code", "Virtues"))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from virtues.modules.layers.attention_flashattention import MHAwithPosEmb

DEV = "cuda"
E, H = 64, 8
SEED = 20260920
OUT_CSV = r"<project-root>\outputs\e0\attention_parity.csv"


def build_mha(seed=SEED):
    torch.manual_seed(seed)
    m = MHAwithPosEmb(embed_dim=E, num_heads=H, dropout=0.0, bias=True,
                      inbuilt_pos_emb="rope")
    return m


def make_inputs(lens, seed, E=E):
    g = torch.Generator().manual_seed(seed)
    total = sum(lens)
    x = torch.randn(1, total, E, generator=g)
    pos = torch.randint(0, 16, (1, total, 2), generator=g)  # LongTensor, per contract
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    maxlen = max(max(lens), 1) if lens else 1
    return x, pos, cu, maxlen


def reference_fp64(module_cpu, x, pos, cu):
    """Per-sequence, unpadded, FP64 plain attention with the module's weights."""
    import copy
    m = copy.deepcopy(module_cpu).double()  # never mutate the shared module in place
    xd = x.double()  # positions stay long: they index the RoPE cos/sin tables
    B, L, _ = xd.shape
    if L == 0:  # total length 0: reference is the empty projection
        return m.W_o(xd)
    Q = m.W_q(xd); K = m.W_k(xd); V = m.W_v(xd)
    Q = Q.view(B, L, H, -1); K = K.view(B, L, H, -1); V = V.view(B, L, H, -1)
    # RoPE after projection, BNH layout like the fallback path
    # (pass raw (B, L, 2) positions; the helper expands per-head itself)
    Q, K = m._apply_pos_after_linear_heads(Q, K, pos, pos, heads_expansion="BNH")
    Q = Q.permute(0, 2, 1, 3); K = K.permute(0, 2, 1, 3); V = V.permute(0, 2, 1, 3)
    outs = []
    cu_l = cu.tolist()
    for s, e in zip(cu_l[:-1], cu_l[1:]):
        if e <= s:
            continue
        q_i, k_i, v_i = Q[:, :, s:e], K[:, :, s:e], V[:, :, s:e]
        d = q_i.shape[-1]
        scores = (q_i * (1.0 / math.sqrt(d))) @ k_i.transpose(-2, -1)
        o_i = torch.softmax(scores, dim=-1) @ v_i  # (B, H, len, D)
        outs.append(o_i.permute(0, 2, 1, 3).reshape(1, e - s, E))
    if not outs:
        return m.W_o(xd.new_zeros(1, 0, E))
    return m.W_o(torch.cat(outs, dim=1))


def run_fallback(module_gpu, x, pos, cu, maxlen, autocast_fp16=False, backend=None):
    import copy
    m = copy.deepcopy(module_gpu).to(DEV).eval()  # never move the shared module in place
    xg, posg, cug = x.to(DEV), pos.to(DEV), cu.to(DEV)
    def _call():
        with torch.no_grad():
            if autocast_fp16:
                with torch.amp.autocast("cuda"):
                    return m(xg, xg, xg, query_pos=posg, key_pos=posg,
                             cu_seq_len=cug, max_seq_len=maxlen)
            return m(xg, xg, xg, query_pos=posg, key_pos=posg,
                     cu_seq_len=cug, max_seq_len=maxlen)
    if backend is not None:
        with sdpa_kernel(backend):
            return _call().float().cpu()
    return _call().float().cpu()


def metrics(ref, got):
    if ref.numel() == 0:
        return dict(max_abs=0.0, rel_l2=0.0, q99=0.0)
    diff = (ref - got).abs()
    rel_l2 = (diff.norm() / ref.norm().clamp_min(1e-30)).item()
    return dict(max_abs=diff.max().item(), rel_l2=rel_l2,
                q99=diff.flatten().kthvalue(max(1, int(0.99 * diff.numel()))).values.item())


def check(tag, m, atol, rtol):
    ok = (m["max_abs"] <= atol) or (m["rel_l2"] <= rtol)
    return ok


def main():
    rows = []
    module = build_mha()
    cases = {
        "A_len1": [1],
        "B_diff_lens": [1, 5, 9, 13],
        "C_padding_stress": [2, 30, 3],
        "E_len0_inside": [5, 0, 7],
        "F_total0": [0],
    }
    for name, lens in cases.items():
        x, pos, cu, maxlen = make_inputs(lens, seed=SEED)
        ref = reference_fp64(module, x, pos, cu)
        # FP32 fallback
        got32 = run_fallback(module, x, pos, cu, maxlen)
        m32 = metrics(ref, got32)
        ok32 = check(name, m32, atol=1e-5, rtol=1e-4)
        # autocast fp16 fallback (production dtype path)
        got16 = run_fallback(module, x, pos, cu, maxlen, autocast_fp16=True)
        m16 = metrics(ref, got16)
        ok16 = check(name, m16, atol=1e-3, rtol=1e-2)
        # backend: MATH vs default (FP32)
        gmath = run_fallback(module, x, pos, cu, maxlen, backend=SDPBackend.MATH)
        gdef = run_fallback(module, x, pos, cu, maxlen)
        mback = metrics(gdef, gmath)
        rows.append(dict(case=name, lens=str(lens), n_tokens=sum(lens),
                         fp32_max_abs=m32["max_abs"], fp32_rel_l2=m32["rel_l2"], fp32_q99=m32["q99"], fp32_pass=ok32,
                         fp16_max_abs=m16["max_abs"], fp16_rel_l2=m16["rel_l2"], fp16_q99=m16["q99"], fp16_pass=ok16,
                         backend_math_vs_default_max_abs=mback["max_abs"]))
        print(rows[-1])

    # D reordering invariance
    lens1, lens2 = [1, 5, 9, 13], [9, 13, 1, 5]
    x1, p1, cu1, ml1 = make_inputs(lens1, seed=SEED)
    x2, p2, cu2, ml2 = make_inputs(lens2, seed=SEED)
    # build second input by permuting the packed segments of the first
    def segments(x, pos, lens):
        xs, ps, o = [], [], 0
        for ln in lens:
            xs.append(x[0, o:o+ln]); ps.append(pos[0, o:o+ln]); o += ln
        return xs, ps
    xs, ps = segments(x1, p1, lens1)
    order = [2, 3, 0, 1]
    x2 = torch.cat([xs[i] for i in order]).unsqueeze(0)
    p2 = torch.cat([ps[i] for i in order]).unsqueeze(0)
    cu2 = torch.tensor([0] + list(torch.tensor(lens2).cumsum(0).tolist()), dtype=torch.int32)
    o1 = run_fallback(module, x1, p1, cu1, ml1)
    o2 = run_fallback(module, x2, p2, cu2, ml2)
    # reorder o2 back to original segment order
    xs2 = segments(o2, o2, lens2)[0]
    o2_back = torch.cat([xs2[order.index(i)] for i in range(4)]).unsqueeze(0)
    mD = metrics(o1, o2_back)
    rows.append(dict(case="D_reorder", lens=str(lens1) + "->" + str(lens2), n_tokens=sum(lens1),
                     fp32_max_abs=mD["max_abs"], fp32_rel_l2=mD["rel_l2"], fp32_q99=mD["q99"],
                     fp32_pass=mD["max_abs"] <= 1e-6,
                     fp16_max_abs=None, fp16_rel_l2=None, fp16_q99=None, fp16_pass=None,
                     backend_math_vs_default_max_abs=None))
    print(rows[-1])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("written:", OUT_CSV)
    fails = [r["case"] for r in rows if r["fp32_pass"] is False] + \
            [r["case"] for r in rows if r.get("fp16_pass") is False]
    print("PARITY", "PASS" if not fails else f"FAIL: {fails}")


if __name__ == "__main__":
    main()
