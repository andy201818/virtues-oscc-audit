import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from virtues.modules.layers.positional_embeddings import (
    PositionalEmbedding2D, LearnablePositionalEmbedding2D, RotaryPositionalEmbedding2D
)

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    # Windows 等无 flash-attn 环境：SDPA 路径（Branch A）不受影响，仅 varlen 打包路径不可用
    flash_attn_varlen_qkvpacked_func = None


class MHAwithPosEmb(nn.Module):
    """
    Multi-Head Attention with optional 2D positional embeddings (absolute / RoPE / learnable),
    supporting two execution paths:
      (1) Standard SDPA path with key_padding_mask (can return attention weights).
      (2) FlashAttention varlen path with cu_seq_len / max_seq_len (no attention weights).

    Args:
        embed_dim (int): Model dimensionality for Q/K/V and output projection.
        num_heads (int): Number of attention heads (embed_dim must be divisible by num_heads).
        dropout (float): Dropout probability applied to attention (and in SDPA path).
        bias (bool): Whether to use bias in linear projections.
        inbuilt_pos_emb (str|None): One of {"absolute", "rope", "protein_learnable", "learnable",
            "absolute_beginning", None}. Controls where/how positional embedding is applied:
              - "absolute": sinusoidal 2D (added BEFORE Q/K linear proj)
              - "rope"   : rotary 2D (applied AFTER Q/K head split; per-head)
              - "protein_learnable": learnable 2D (added directly to Q/K inputs BEFORE proj)
              - "learnable": no-op here (expect caller to add external learnable pos-emb if needed)
              - "absolute_beginning" or None: no internal pos-emb added
        keyval_embed_dim (int|None): If provided, the input dim for K/V projections (when K/V come
            from a different source than Q). Defaults to embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        inbuilt_pos_emb: Optional[str] = "absolute",
        keyval_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if keyval_embed_dim is None:
            keyval_embed_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)

        # Projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(keyval_embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(keyval_embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Positional embedding policy
        self.pos_after_linear = False
        self.pos_before_linear = False
        self.pos_emb: Optional[nn.Module] = None

        if inbuilt_pos_emb == "absolute":
            self.pos_emb = PositionalEmbedding2D(model_dim=self.embed_dim)
            self.pos_before_linear = True
        elif inbuilt_pos_emb == "rope":
            # RoPE is applied per-head AFTER linear projections
            self.pos_emb = RotaryPositionalEmbedding2D(model_dim=self.head_dim)
            self.pos_after_linear = True
        elif inbuilt_pos_emb == "protein_learnable":
            self.pos_emb = LearnablePositionalEmbedding2D(model_dim=self.embed_dim)
            # Added directly to inputs BEFORE linear projections
            self.pos_before_linear = False
        elif inbuilt_pos_emb in {"learnable", "absolute_beginning", None}:
            # No internal application here (caller may handle separately)
            pass
        else:
            raise ValueError(
                "inbuilt_pos_emb must be one of "
                "{'absolute','rope','protein_learnable','learnable','absolute_beginning',None}"
            )

    def _apply_pos_before_linear(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        key_pos: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_emb is not None and self.pos_before_linear:
            if query_pos is not None:
                query = self.pos_emb(query, query_pos)
            if key_pos is not None:
                key = self.pos_emb(key, key_pos)
        return query, key

    def _apply_pos_after_linear_heads(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        key_pos: Optional[torch.Tensor],
        heads_expansion: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply per-head positional embedding (e.g., RoPE) after linear projections.
        `heads_expansion` controls how positions are broadcast:
          - "BHN" means we already have shape (B, H, N, 2), i.e., we expanded along heads.
          - "BNH" means (B, N, H, 2).
        """
        if self.pos_emb is None or not self.pos_after_linear:
            return q, k

        if query_pos is not None:
            if heads_expansion == "BHN":
                q = self.pos_emb(q, query_pos)  # (B, H, L, D)
            else:
                # Expand to per-head
                query_pos_exp = query_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # (B, L, H, 2)
                q = self.pos_emb(q, query_pos_exp)  # (B, L, H, D)

        if key_pos is not None:
            if heads_expansion == "BHN":
                k = self.pos_emb(k, key_pos)  # (B, H, S, D)
            else:
                key_pos_exp = key_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # (B, S, H, 2)
                k = self.pos_emb(k, key_pos_exp)  # (B, S, H, D)

        return q, k

    def forward(
        self,
        query: torch.Tensor,        # (B, L, embed_dim)
        key: torch.Tensor,          # (B, S, keyval_embed_dim or embed_dim)
        value: torch.Tensor,        # (B, S, keyval_embed_dim or embed_dim)
        query_pos: Optional[torch.Tensor] = None,  # (B, L, 2)
        key_pos: Optional[torch.Tensor] = None,    # (B, S, 2)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool or float(-inf/0) mask
        return_attention: bool = False,
        cu_seq_len: Optional[torch.Tensor] = None,  # FlashAttention varlen
        max_seq_len: Optional[int] = None,          # FlashAttention varlen
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            If return_attention is False:
                attn_out: (B, L, embed_dim)
            If return_attention is True (only SDPA/manual path):
                (attn_out, attn_probs): (B, L, embed_dim), (B, num_heads, L, S)
        """

        # Sanity: mutually exclusive paths
        if (cu_seq_len is not None or max_seq_len is not None) and key_padding_mask is not None:
            raise ValueError("Provide either (cu_seq_len, max_seq_len) OR key_padding_mask, not both.")

        B, L = query.shape[0], query.shape[1]
        S = key.shape[1]

        # (1) Optional positional embedding BEFORE linear projections
        query, key = self._apply_pos_before_linear(query, key, query_pos, key_pos)

        # (2) Linear projections
        Q = self.W_q(query)   # (B, L, E)
        K = self.W_k(key)     # (B, S, E)
        V = self.W_v(value)   # (B, S, E)

        # Branch A: SDPA/manual path (supports attention return)
        if key_padding_mask is not None:
            # Shape check
            if key_padding_mask.shape != (B, S):
                raise ValueError(f"key_padding_mask must be (B, S), got {tuple(key_padding_mask.shape)}")

            # Prepare mask: (B, 1, 1, S) broadcastable to (B, H, L, S)
            if key_padding_mask.dtype == torch.bool:
                # True = pad → add -inf
                attn_mask = torch.zeros(B, 1, 1, S, device=Q.device, dtype=Q.dtype)
                attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            else:
                # Assume already additive mask with -inf/0 of shape (B, S)
                attn_mask = key_padding_mask[:, None, None, :].to(Q.dtype)

            # Reshape to heads: (B, H, N, D)
            Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            # Optional positional embedding AFTER linear projections (per head)
            if self.pos_after_linear:
                # Expand positions to (B, H, N, 2)
                qpos = query_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) if query_pos is not None else None
                kpos = key_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) if key_pos is not None else None
                Q, K = self._apply_pos_after_linear_heads(Q, K, qpos, kpos, heads_expansion="BHN")

            scale = 1.0 / math.sqrt(self.head_dim)

            if not return_attention:
                # Fast path: use PyTorch SDPA
                attn_out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=attn_mask, dropout_p=self.dropout
                )  # (B, H, L, D)
            else:
                # Manual attention to return weights
                scores = torch.matmul(Q * scale, K.transpose(-2, -1))  # (B, H, L, S)
                if attn_mask is not None:
                    scores = scores + attn_mask
                attn_probs = F.softmax(scores, dim=-1)
                attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
                attn_out = torch.matmul(attn_probs, V)  # (B, H, L, D)

            # Merge heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
            out = self.W_o(attn_out)

            if return_attention:
                return out, attn_probs  # (B, L, E), (B, H, L, S)
            return out

        # Branch B: FlashAttention varlen path (no attention return)
        if return_attention:
            raise ValueError("return_attention=True is not supported with FlashAttention varlen path.")

        if cu_seq_len is None or max_seq_len is None:
            raise ValueError("FlashAttention path requires both cu_seq_len and max_seq_len.")

        # Heads in (B, N, H, D) for packing
        Q = Q.view(B, L, self.num_heads, self.head_dim)
        K = K.view(B, S, self.num_heads, self.head_dim)
        V = V.view(B, S, self.num_heads, self.head_dim)

        # Optional positional embedding AFTER linear projections (per head)
        if self.pos_after_linear:
            Q, K = self._apply_pos_after_linear_heads(Q, K, query_pos, key_pos, heads_expansion="BNH")

        if torch.is_autocast_enabled():
            Q, K, V = Q.half(), K.half(), V.half()


        qkv=torch.stack([Q.squeeze(0), K.squeeze(0), V.squeeze(0)], dim=1)

        
        if flash_attn_varlen_qkvpacked_func is not None:
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv=qkv,
                cu_seqlens=cu_seq_len,
                max_seqlen=max_seq_len,
                deterministic=True,
                dropout_p=self.dropout if self.training else 0.0,
            ).unsqueeze(0)  # (total_Q, H, D) collapsed over batch by varlen kernel
        else:
            # flash-attn 不可用（如 Windows）：分块对角掩码的单次 SDPA，与逐序列注意力数学等价
            cu = cu_seq_len.tolist()
            lens = [e - s for s, e in zip(cu[:-1], cu[1:])]
            H, D = self.num_heads, self.head_dim
            maxlen = max(max(lens), 1)
            nseq = len(lens)
            Lq = Q.shape[1]
            # 打包为 (nseq, maxlen, H, D) 的填充批
            Qp = Q.new_zeros(nseq, maxlen, H, D)
            Kp = K.new_zeros(nseq, maxlen, H, D)
            Vp = V.new_zeros(nseq, maxlen, H, D)
            pos = 0
            valid = torch.zeros(nseq, maxlen, dtype=torch.bool, device=Q.device)
            for si, ln in enumerate(lens):
                if ln == 0:
                    continue
                Qp[si, :ln] = Q[0, pos:pos + ln]
                Kp[si, :ln] = K[0, pos:pos + ln]
                Vp[si, :ln] = V[0, pos:pos + ln]
                valid[si, :ln] = True
                pos += ln
            Qp = Qp.permute(0, 2, 1, 3)  # (nseq, H, maxlen, D)
            Kp = Kp.permute(0, 2, 1, 3)
            Vp = Vp.permute(0, 2, 1, 3)
            # 序列内可见的分块对角掩码 + 键无效位屏蔽
            seq_id = torch.arange(nseq, device=Q.device).unsqueeze(1).expand(nseq, maxlen)
            diag = seq_id[:, None, :, None] == seq_id[:, None, None, :]
            mask = diag & valid[:, None, None, :]
            o_ = F.scaled_dot_product_attention(Qp, Kp, Vp, attn_mask=mask, dropout_p=0.0)
            o_ = o_.permute(0, 2, 1, 3).reshape(nseq * maxlen, H, D)
            # 展平回 packed 顺序
            out_list = []
            for si, ln in enumerate(lens):
                out_list.append(o_[si * maxlen: si * maxlen + ln])
            attn_out = torch.cat(out_list, dim=0).unsqueeze(0)  # (1, total_Q, H, D)

        out = attn_out.reshape(B, -1, self.num_heads * self.head_dim)  # (B, L, embed_dim)

        return self.W_o(out)  # (B, L, E)
