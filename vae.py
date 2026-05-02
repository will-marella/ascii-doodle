"""Continuous beta-VAE for ASCII canvases.

Encodes a [B, H, W] grid of integer ASCII tokens into a small continuous
latent [B, C, H', W'], and decodes from a latent to per-position logits over
the 8 ASCII tokens. Trained jointly with CE reconstruction + KL regularization
(with optional free bits to prevent posterior collapse).

This is the "image tokenizer" for v5 latent diffusion. The encoder/decoder
are intentionally lean — the heavy generative lifting happens downstream in
latent space.

Architecture (default, 4x spatial compression):

    [B, 32, 64]  int tokens
        -> tok_emb                 [B, 32, 64, 32]
        -> input_proj              [B, 64, 32, 64]
        -> encoder stage 1 (down)  [B, 128, 16, 32]
        -> encoder stage 2 (down)  [B, 256, 8, 16]
        -> latent_proj             [B, 32, 8, 16]  (16 mean + 16 logvar)
        -> reparameterize          [B, 16, 8, 16]  (latent z)
        -> latent_unproj           [B, 256, 8, 16]
        -> decoder stage 1 (up)    [B, 128, 16, 32]
        -> decoder stage 2 (up)    [B, 64, 32, 64]
        -> output_proj             [B, 8, 32, 64]  (logits over chars)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a channel-count-compatible group size."""
    g = num_groups
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    """Pre-norm residual block: norm -> act -> conv -> norm -> act -> conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class AsciiVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8,
        grid_h: int = 32,
        grid_w: int = 64,
        embed_dim: int = 32,
        base_channels: int = 64,
        latent_channels: int = 16,
        downsample_stages: int = 2,
    ):
        super().__init__()
        assert grid_h % (2 ** downsample_stages) == 0, (
            f'grid_h {grid_h} must divide {2 ** downsample_stages}'
        )
        assert grid_w % (2 ** downsample_stages) == 0, (
            f'grid_w {grid_w} must divide {2 ** downsample_stages}'
        )

        self.vocab_size = vocab_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.embed_dim = embed_dim
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.downsample_stages = downsample_stages
        self.latent_h = grid_h // (2 ** downsample_stages)
        self.latent_w = grid_w // (2 ** downsample_stages)

        # Token embedding (plain nn.Embedding; no ordinal initialization in this baseline)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        # --- Encoder ---
        self.input_proj = nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1)

        enc_blocks = []
        ch = base_channels
        for _ in range(downsample_stages):
            out_ch = ch * 2
            enc_blocks.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),  # 2x downsample
                _group_norm(out_ch),
                nn.GELU(),
                ResBlock(out_ch),
            ))
            ch = out_ch
        self.encoder_blocks = nn.ModuleList(enc_blocks)
        self.encoder_out_ch = ch

        # Final encoder projection to (mean + log_var)
        self.latent_proj = nn.Conv2d(ch, latent_channels * 2, kernel_size=1)

        # --- Decoder ---
        # Project latent back up to encoder output dimensionality
        self.latent_unproj = nn.Conv2d(latent_channels, ch, kernel_size=1)

        dec_blocks = []
        for _ in range(downsample_stages):
            out_ch = ch // 2
            dec_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                _group_norm(out_ch),
                nn.GELU(),
                ResBlock(out_ch),
            ))
            ch = out_ch
        self.decoder_blocks = nn.ModuleList(dec_blocks)

        # Final output conv to per-position logits over vocab
        self.output_proj = nn.Conv2d(ch, vocab_size, kernel_size=3, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def encode(self, tokens: torch.Tensor):
        """
        tokens: [B, H, W] long
        returns: (mean, log_var) each [B, latent_channels, H', W']
        """
        x = self.tok_emb(tokens)                          # [B, H, W, D]
        x = x.permute(0, 3, 1, 2).contiguous()             # [B, D, H, W]
        x = self.input_proj(x)                             # [B, C, H, W]
        for block in self.encoder_blocks:
            x = block(x)                                   # downsample
        z_params = self.latent_proj(x)                     # [B, 2C_lat, H', W']
        mean, log_var = z_params.chunk(2, dim=1)
        # Clamp log_var for numerical stability
        log_var = log_var.clamp(min=-30.0, max=20.0)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_channels, H', W']
        returns: logits [B, vocab_size, H, W]
        """
        x = self.latent_unproj(z)
        for block in self.decoder_blocks:
            x = block(x)
        return self.output_proj(x)

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, H, W] long
        returns: (logits [B, V, H, W], mean, log_var)
        """
        mean, log_var = self.encode(tokens)
        z = self.reparameterize(mean, log_var)
        logits = self.decode(z)
        return logits, mean, log_var

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
