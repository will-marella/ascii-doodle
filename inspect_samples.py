"""Load a trained checkpoint and generate samples under multiple sampling configs.

Prints a side-by-side view of what each sampling strategy produces for each of
the curated eval classes. Use this to diagnose whether the trained model has
more to offer under different decoding strategies, or whether its current
limits are actually at the model level.

Usage (local, auto-detects device):
    python inspect_samples.py --checkpoint checkpoints/step_19000.pt

Usage (force CPU):
    python inspect_samples.py --checkpoint checkpoints/step_19000.pt --device cpu

Usage (via Modal, GPU-backed):
    modal run modal_app.py::inspect
"""

import argparse

import torch

from data import VOCAB_SIZE
from model import AsciiTransformer
from sample import decode, generate


CLASS_NAMES = [
    'Girl',
    'Car',
    'Flower',
    'Dog',
    'Bird',
    'Guitar',
    'Fixed-wing aircraft',
    'Cake',
]


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
    class_map = ckpt['class_map']

    model = AsciiTransformer(
        vocab_size=VOCAB_SIZE,
        n_classes=len(class_map),
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'  step {ckpt["step"]}, {model.num_params():,} params, {len(class_map)} classes')
    return model, class_map


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
    )
    args = p.parse_args(argv)

    model, class_map = load_model(args.checkpoint, args.device)

    # Resolve curated class IDs; skip any missing and warn.
    class_ids = []
    found_names = []
    for name in CLASS_NAMES:
        if name in class_map:
            class_ids.append(class_map[name])
            found_names.append(name)
        else:
            print(f'Warning: {name!r} not in class_map, skipping')
    class_id_tensor = torch.tensor(class_ids, dtype=torch.long, device=args.device)

    for label, temperature, top_k in SAMPLING_CONFIGS:
        print()
        print('=' * 80)
        print(f'  SAMPLING CONFIG: {label}')
        print('=' * 80)

        sampled = generate(model, class_id_tensor, temperature=temperature, top_k=top_k)

        for i, name in enumerate(found_names):
            print()
            print(f'--- {name} ---')
            print(decode(sampled[i]))


if __name__ == '__main__':
    main()
