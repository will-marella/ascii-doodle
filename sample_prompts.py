"""Sample ASCII art from free-text prompts via the current QuickDraw stack."""

import argparse

import torch

from inference import DEFAULT_DIT_CHECKPOINT, generate_ascii, load_pipeline


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=DEFAULT_DIT_CHECKPOINT)
    p.add_argument('--vae-checkpoint', default=None)
    p.add_argument('--prompts', nargs='+', required=True)
    p.add_argument('--guidance-scale', type=float, default=10.0)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)

    print(f'Loading DiT from {args.checkpoint}...')
    pipeline = load_pipeline(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        device=args.device,
    )
    print(f'  {pipeline.dit.num_params():,} params')

    print(f'\nguidance_scale={args.guidance_scale}, steps={args.sample_steps}\n')

    for prompt in args.prompts:
        ascii_art = generate_ascii(
            pipeline,
            prompt,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
        )
        print(f'--- {prompt} ---')
        for line in ascii_art.split('\n'):
            print(f'  {line}')
        print()


if __name__ == '__main__':
    main()
