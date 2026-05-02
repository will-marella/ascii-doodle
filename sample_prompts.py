"""Sample ASCII art from free-text prompts via a trained DiT.

Usage:
    python sample_prompts.py --checkpoint checkpoints/dit_vae_full_250m/step_60000.pt \
        --prompts "cat" "robot" "sunset"
"""

import argparse
import torch
from sample import decode, text_to_clip_embeddings
from train_dit import load_vae, sample_flow, unnormalize_latent
from dit import DiT


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--vae-checkpoint', default=None)
    p.add_argument('--prompts', nargs='+', required=True)
    p.add_argument('--guidance-scale', type=float, default=10.0)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = args.device

    # Load DiT
    print(f'Loading DiT from {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    clip_dim = ckpt.get('clip_dim', 0)

    # Load VAE
    vae_path = args.vae_checkpoint or config.get('vae_checkpoint')
    vae, grid_h, grid_w = load_vae(vae_path, device)

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

    latent_mean = ckpt['latent_mean'].to(device)
    latent_std = ckpt['latent_std'].to(device)

    # Encode prompts via CLIP text encoder
    clip_prompts = [f'a drawing of a {p}' for p in args.prompts]
    print(f'Encoding {len(args.prompts)} prompts via CLIP...')
    clip_embs = text_to_clip_embeddings(clip_prompts, device=torch.device(device))

    print(f'\nguidance_scale={args.guidance_scale}, steps={args.sample_steps}\n')

    for i, prompt in enumerate(args.prompts):
        emb = clip_embs[i].unsqueeze(0)
        z_norm = sample_flow(
            dit, 1, n_steps=args.sample_steps,
            clip_emb=emb, guidance_scale=args.guidance_scale,
        )
        z = unnormalize_latent(z_norm, latent_mean, latent_std)
        with torch.no_grad():
            logits = vae.decode(z)
            tokens = logits.argmax(dim=1)
        flat = tokens[0].reshape(-1).tolist()
        print(f'--- {prompt} ---')
        for line in decode(flat, grid_w=grid_w).split('\n'):
            print(f'  {line}')
        print()


if __name__ == '__main__':
    main()
