"""Inference helpers for the current QuickDraw ASCII pipeline."""

from dataclasses import dataclass

import torch

from ascii_utils import decode
from clip_utils import text_to_clip_embeddings
from dit import DiT
from train_dit import load_vae, sample_flow, unnormalize_latent


DEFAULT_DIT_CHECKPOINT = 'checkpoints/dit_vae_full_250m/step_60000.pt'
DEFAULT_VAE_CHECKPOINT = 'checkpoints/vae_qd_full_b01/step_10000.pt'
PROMPT_PREFIX = 'a drawing of a '


@dataclass
class AsciiInferencePipeline:
    dit: DiT
    vae: torch.nn.Module
    grid_w: int
    device: str
    latent_mean: torch.Tensor
    latent_std: torch.Tensor


def load_pipeline(
    checkpoint: str = DEFAULT_DIT_CHECKPOINT,
    vae_checkpoint: str | None = None,
    device: str | None = None,
) -> AsciiInferencePipeline:
    """Load the pretrained VAE + DiT stack for prompt-conditioned generation."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    clip_dim = ckpt.get('clip_dim', 0)

    vae_path = vae_checkpoint or config.get('vae_checkpoint') or DEFAULT_VAE_CHECKPOINT
    vae, _, grid_w = load_vae(vae_path, device)

    dit = DiT(
        latent_channels=vae.latent_channels,
        latent_h=vae.latent_h,
        latent_w=vae.latent_w,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        ffn_mult=config.get('ffn_mult', 4),
        clip_dim=clip_dim,
    ).to(device)
    dit.load_state_dict(ckpt['model'])
    dit.eval()

    return AsciiInferencePipeline(
        dit=dit,
        vae=vae,
        grid_w=grid_w,
        device=device,
        latent_mean=ckpt['latent_mean'].to(device),
        latent_std=ckpt['latent_std'].to(device),
    )


@torch.no_grad()
def generate_ascii(
    pipeline: AsciiInferencePipeline,
    prompt: str,
    guidance_scale: float = 10.0,
    sample_steps: int = 50,
    prompt_prefix: str = PROMPT_PREFIX,
) -> str:
    """Generate ASCII art for a single prompt."""
    clip_emb = text_to_clip_embeddings(
        [f'{prompt_prefix}{prompt}'],
        device=torch.device(pipeline.device),
    )
    z_norm = sample_flow(
        pipeline.dit,
        1,
        n_steps=sample_steps,
        clip_emb=clip_emb,
        guidance_scale=guidance_scale,
    )
    z = unnormalize_latent(z_norm, pipeline.latent_mean, pipeline.latent_std)
    logits = pipeline.vae.decode(z)
    tokens = logits.argmax(dim=1)
    flat = tokens[0].reshape(-1).tolist()
    return decode(flat, grid_w=pipeline.grid_w)
