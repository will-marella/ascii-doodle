"""Load a trained checkpoint and generate samples under multiple sampling configs.

Prints a side-by-side view of what each sampling strategy produces for a fixed
number of unconditional samples. Use this to diagnose whether the trained model
has more to offer under different decoding strategies.

Usage (local, auto-detects device):
    python inspect_samples.py --checkpoint checkpoints/v2_humans/step_19000.pt

Usage (via Modal, GPU-backed):
    modal run modal_app.py::inspect
"""

import argparse

import torch

from data import VOCAB_SIZE
from model import AsciiTransformer
from sample import decode, generate


# (label, temperature, top_k). top_k=1 is effectively greedy decoding.
SAMPLING_CONFIGS = [
    ('greedy  (T=1.0 k=1)', 1.0, 1),
    ('low-T   (T=0.3 k=5)', 0.3, 5),
    ('default (T=0.8 k=3)', 0.8, 3),
    ('high-T  (T=1.0 k=5)', 1.0, 5),
]


def load_model(checkpoint_path: str, device: str):
    print(f'Loading checkpoint from {checkpoint_path}...')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = AsciiTransformer(
        vocab_size=VOCAB_SIZE,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'  step {ckpt["step"]}, {model.num_params():,} params')
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

    for label, temperature, top_k in SAMPLING_CONFIGS:
        print()
        print('=' * 80)
        print(f'  SAMPLING CONFIG: {label}')
        print('=' * 80)

        sampled = generate(model, args.n_samples, temperature=temperature, top_k=top_k)

        for i in range(args.n_samples):
            print()
            print(f'--- sample {i+1}/{args.n_samples} ---')
            print(decode(sampled[i]))


if __name__ == '__main__':
    main()
