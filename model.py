"""Autoregressive transformer for ASCII image generation.

v2: unconditional (no class embedding). Default config:
8 layers x 384 hidden x 6 heads (64/head) x 1536 FFN  ~ 14.2M params.

2D learned positional encoding (row + col), pre-LayerNorm, causal
self-attention via F.scaled_dot_product_attention (flash attention on A100+).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn)

        x = x + self.ffn(self.ln2(x))
        return x


class AsciiTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8,
        grid_h: int = 32,
        grid_w: int = 64,
        dim: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        ffn_mult: int = 4,
    ):
        super().__init__()
        seq_len = grid_h * grid_w

        self.vocab_size = vocab_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.row_emb = nn.Embedding(grid_h, dim)
        self.col_emb = nn.Embedding(grid_w, dim)

        # Precomputed row/col index for every position in raster order
        positions = torch.arange(seq_len)
        self.register_buffer('row_idx', positions // grid_w, persistent=False)
        self.register_buffer('col_idx', positions % grid_w, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dim * ffn_mult)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T] long
        returns: [B, T, vocab_size] logits
        """
        B, T = tokens.shape
        x = (
            self.tok_emb(tokens)
            + self.row_emb(self.row_idx[:T])
            + self.col_emb(self.col_idx[:T])
        )
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
