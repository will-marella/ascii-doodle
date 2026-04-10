"""Load a trained MaskGIT checkpoint and generate samples under several configs.

Prints samples produced by varying the iterative-decoding hyperparameters:
number of unmasking steps, sampling temperature, and Gumbel noise scale. Use
this to explore the decoding tradeoff space on a trained checkpoint.

Usage (local):
    python inspect_samples.py --checkpoint checkpoints/v3_maskgit/step_19000.pt

Usage (Modal):
    modal run modal_app.py::inspect
"""

import argparse

import torch

from data import VOCAB_SIZE
from model import AsciiTransformer
from sample import decode, generate


# (label, n_steps, temperature, noise_temperature)
SAMPLING_CONFIGS = [
    ('fast     (steps=8,  T=1.0, noise=4.5)',  8, 1.0, 4.5),
    ('standard (steps=12, T=1.0, noise=4.5)', 12, 1.0, 4.5),
    ('slow     (steps=24, T=1.0, noise=4.5)', 24, 1.0, 4.5),
    ('sharp    (steps=12, T=0.7, noise=2.0)', 12, 0.7, 2.0),
]


def load_model(checkpoint_path: str, device: str):
    print(f'Loading checkpoint from {checkpoint_path}...')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    version = ckpt.get('version', 'unknown')

    model = AsciiTransformer(
        vocab_size=VOCAB_SIZE,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'  step {ckpt["step"]}, {model.num_params():,} params, version={version}')
    return model


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
    )
    p.add_argument('--n-samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device)

    for label, n_steps, temperature, noise_temp in SAMPLING_CONFIGS:
        print()
        print('=' * 80)
        print(f'  SAMPLING CONFIG: {label}')
        print('=' * 80)

        sampled = generate(
            model,
            n_samples=args.n_samples,
            n_steps=n_steps,
            temperature=temperature,
            noise_temperature=noise_temp,
        )

        for i in range(args.n_samples):
            print()
            print(f'--- sample {i+1}/{args.n_samples} ---')
            print(decode(sampled[i]))


if __name__ == '__main__':
    main()
