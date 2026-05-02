"""Diffusion Transformer (DiT) for flow-matching latent generation.

Operates on the VAE's compressed latent space [B, C, H', W']. For our default
setup (128x64 input, 8x VAE compression) the latent is [16, 8, 16] = 128
positions x 16 channels — small enough that we can afford a relatively deep
transformer cheaply.

Architecture:
- Project [C, H', W'] -> [seq_len, dim] via 1x1 conv (per-position linear)
- Add learned 2D positional embedding (row + col)
- N transformer blocks with bidirectional self-attention and adaLN-Zero
  conditioning on timestep
- Final projection back to [C, H', W']

adaLN-Zero (Peebles & Xie 2022):
- Each block has 6 modulations from the time embedding: shift/scale/gate
  for the attention residual, and shift/scale/gate for the FFN residual
- Modulation Linear weights are zero-initialized so the model starts as
  the identity (predicts zero velocity, gates are zero) -- training is
  stable from step 0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding for continuous timesteps in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero modulation."""

    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # No learned scale/shift in the norms — adaLN takes care of that
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

        # adaLN-Zero: 6 modulations per block (scale1, shift1, gate1, scale2, shift2, gate2)
        self.ada_ln = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], t_emb: [B, D]
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada_ln(t_emb).chunk(6, dim=-1)
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # --- attention ---
        h = self.norm1(x) * (1 + scale1) + shift1
        B, T, D = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + gate1 * self.out_proj(attn)

        # --- FFN ---
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(h)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        latent_h: int = 8,
        latent_w: int = 16,
        dim: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        ffn_mult: int = 4,
        time_freq_dim: int = 256,
        clip_dim: int = 0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.dim = dim
        self.seq_len = latent_h * latent_w
        self.clip_dim = clip_dim

        # --- Input projection: per-position [C] -> [D] ---
        self.in_proj = nn.Linear(latent_channels, dim)

        # --- 2D positional embedding ---
        self.row_emb = nn.Embedding(latent_h, dim)
        self.col_emb = nn.Embedding(latent_w, dim)
        positions = torch.arange(self.seq_len)
        self.register_buffer('row_idx', positions // latent_w, persistent=False)
        self.register_buffer('col_idx', positions % latent_w, persistent=False)

        # --- Time embedding ---
        self.time_freq_dim = time_freq_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_freq_dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # --- CLIP conditioning (optional) ---
        # When clip_dim > 0, the model accepts a [B, clip_dim] embedding per
        # sample. The embedding is projected to model dim and added to the
        # time embedding before adaLN. A learned `null_clip` parameter is used
        # for classifier-free guidance: replace 10% of training samples with it,
        # and at inference time do two forward passes (cond + null) for CFG.
        if clip_dim > 0:
            self.clip_proj = nn.Sequential(
                nn.Linear(clip_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            # Learned null embedding (at clip_dim, gets projected through clip_proj)
            self.null_clip = nn.Parameter(torch.zeros(clip_dim))
        else:
            self.clip_proj = None
            self.null_clip = None

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads, dim * ffn_mult) for _ in range(n_layers)
        ])

        # --- Output: final adaLN + projection back to [C] ---
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln_final = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.ada_ln_final.weight)
        nn.init.zeros_(self.ada_ln_final.bias)
        self.out_proj = nn.Linear(dim, latent_channels)
        # Zero-init output for adaLN-Zero — model starts as identity (zero velocity)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Default init for everything else
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # Don't reinitialize the zero-initialized layers
            if m.weight.detach().abs().sum() == 0:
                return
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        clip_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        z: [B, C, H', W'] noised latent
        t: [B] timestep in [0, 1]
        clip_emb: [B, clip_dim] CLIP embedding (only if model was built with clip_dim>0).
                  Caller is responsible for substituting null embeddings as needed
                  for CFG dropout (during training) or unconditional sampling.
        returns: [B, C, H', W'] predicted velocity
        """
        B, C, H, W = z.shape
        assert C == self.latent_channels and H == self.latent_h and W == self.latent_w

        # [B, C, H, W] -> [B, H*W, C] -> [B, H*W, D]
        x = z.flatten(2).transpose(1, 2)
        x = self.in_proj(x)

        # 2D positional encoding
        x = x + self.row_emb(self.row_idx)[None, :, :]
        x = x + self.col_emb(self.col_idx)[None, :, :]

        # Time embedding
        t_freq = sinusoidal_time_embedding(t, self.time_freq_dim)
        t_emb = self.time_mlp(t_freq)  # [B, D]

        # Conditioning: time + (optional) CLIP
        cond = t_emb
        if self.clip_proj is not None:
            assert clip_emb is not None, 'Model expects clip_emb but None was passed'
            c_emb = self.clip_proj(clip_emb)  # [B, D]
            cond = cond + c_emb

        for block in self.blocks:
            x = block(x, cond)

        # Final adaLN + projection
        scale, shift = self.ada_ln_final(cond).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.out_proj(x)  # [B, H*W, C]

        # [B, H*W, C] -> [B, C, H, W]
        return x.transpose(1, 2).reshape(B, C, H, W)

    def get_null_clip(self, batch_size: int = 1) -> torch.Tensor:
        """Return the learned null embedding broadcast to a batch.
        Used for CFG dropout in training and unconditional pass in sampling.
        """
        assert self.null_clip is not None, 'Model has no CLIP conditioning'
        return self.null_clip[None].expand(batch_size, -1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
