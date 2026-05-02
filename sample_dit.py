"""Sample from a trained DiT checkpoint for a random set of categories.

Usage:
    python sample_dit.py --checkpoint checkpoints/dit_vae_full_250m/step_60000.pt --n-categories 50
"""

import argparse
import json
import random

import numpy as np
import torch

from data import N_ASCII_TOKENS
from dit import DiT
from sample import decode
from train_dit import load_vae, sample_flow, normalize_latent, unnormalize_latent
from vae import AsciiVAE


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--vae-checkpoint', default=None,
                   help='VAE checkpoint. If omitted, inferred from DiT config.')
    p.add_argument('--n-categories', type=int, default=50)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--guidance-scale', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    # Load DiT checkpoint
    print(f'Loading DiT from {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    clip_dim = ckpt.get('clip_dim', 0)

    # Load VAE
    vae_path = args.vae_checkpoint or config.get('vae_checkpoint')
    vae, grid_h, grid_w = load_vae(vae_path, device)

    # Build DiT
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
    print(f'  {dit.num_params():,} params, step {ckpt["step"]}')

    # Latent stats
    latent_mean = ckpt['latent_mean'].to(device)
    latent_std = ckpt['latent_std'].to(device)

    # Load categories and CLIP embeddings
    train_npy = config.get('train_npy', 'quickdraw/full_32x16')
    cats_path = f'{train_npy}/categories.json'
    clip_path = f'{train_npy}/clip_embeddings.npy'

    with open(cats_path) as f:
        categories = json.load(f)
    clip_embs = torch.from_numpy(np.load(clip_path)).float().to(device)

    # Pick random categories
    n = min(args.n_categories, len(categories))
    indices = sorted(random.sample(range(len(categories)), n))
    selected = [categories[i] for i in indices]

    print(f'\nSampling {n} categories (seed={args.seed}, '
          f'guidance={args.guidance_scale}, steps={args.sample_steps})\n')

    # Sample one at a time to keep memory low and output readable
    for cat_idx, cat_name in zip(indices, selected):
        emb = clip_embs[cat_idx].unsqueeze(0)  # [1, clip_dim]
        z_norm = sample_flow(
            dit, 1, n_steps=args.sample_steps,
            clip_emb=emb, guidance_scale=args.guidance_scale,
        )
        z = unnormalize_latent(z_norm, latent_mean, latent_std)
        with torch.no_grad():
            logits = vae.decode(z)
            tokens = logits.argmax(dim=1)
        flat = tokens[0].reshape(-1).tolist()
        print(f'--- {cat_name} ---')
        for line in decode(flat, grid_w=grid_w).split('\n'):
            print(f'  {line}')
        print()


if __name__ == '__main__':
    main()
