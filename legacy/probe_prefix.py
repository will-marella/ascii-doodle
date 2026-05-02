"""Probe: does the MaskGIT model fill in missing regions correctly?

For a MaskGIT model, "prefix priming" generalizes to "arbitrary positional
conditioning": we can expose any subset of real tokens and let the model fill
in the rest. This script runs several variants of that test:

  - Top-N rows visible (fill the bottom)
  - Bottom-N rows visible (fill the top)
  - Left-half visible (fill the right)
  - Random positions visible (fill the rest)

For each, it prints the original, the masked input, and the generated
completion, so you can eyeball how well the model uses partial context.

Usage (local):
    python probe_prefix.py --checkpoint checkpoints/v3_maskgit/step_19000.pt

Usage (Modal):
    modal run modal_app.py::probe
"""

import argparse
import json
import random

import numpy as np
import torch

from data import (
    GRID_H,
    GRID_W,
    MASK_TOKEN,
    SEQ_LEN,
    VOCAB_SIZE,
    build_char_lookup,
    tokenize_ascii,
)
from model import AsciiTransformer
from sample import decode, generate


FILTER_LABELS = frozenset({'Girl', 'Woman', 'Boy', 'Man'})


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


def load_real_samples(jsonl_path: str, n: int, seed: int):
    table = build_char_lookup()
    matching = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r['label'] in FILTER_LABELS:
                matching.append(r)

    rng = random.Random(seed)
    rng.shuffle(matching)
    picks = matching[:n]

    arr = np.stack([tokenize_ascii(p['ascii'], table) for p in picks])
    labels = [p['label'] for p in picks]
    return torch.from_numpy(arr).long(), labels


def make_top_mask(n_rows: int) -> torch.Tensor:
    """Boolean mask [SEQ_LEN]: True = visible (keep real), False = masked (to fill)."""
    visible = torch.zeros(SEQ_LEN, dtype=torch.bool)
    visible[: n_rows * GRID_W] = True
    return visible


def make_bottom_mask(n_rows: int) -> torch.Tensor:
    visible = torch.zeros(SEQ_LEN, dtype=torch.bool)
    visible[(GRID_H - n_rows) * GRID_W:] = True
    return visible


def make_left_half_mask() -> torch.Tensor:
    visible = torch.zeros(GRID_H, GRID_W, dtype=torch.bool)
    visible[:, : GRID_W // 2] = True
    return visible.reshape(-1)


def make_random_mask(fraction_visible: float, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    u = torch.rand(SEQ_LEN, generator=rng)
    return u < fraction_visible


def apply_visibility(real_tokens: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Return a copy of real_tokens where masked positions are replaced with MASK_TOKEN."""
    masked = real_tokens.clone()
    masked[..., ~visible] = MASK_TOKEN
    return masked


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train-path', default='openimages/train_ascii_64x32_cc.jsonl')
    p.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
    )
    p.add_argument('--n-samples', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-steps', type=int, default=12)
    p.add_argument('--temperature', type=float, default=1.0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device)

    print(f'Loading {args.n_samples} real samples from {args.train_path}...')
    real, labels = load_real_samples(args.train_path, args.n_samples, args.seed)
    real = real.to(args.device)
    print(f'  picked: {labels}')

    # Visibility scenarios: (label, visible-mask tensor)
    scenarios = [
        ('unconditional (all masked)', torch.zeros(SEQ_LEN, dtype=torch.bool)),
        ('top 4 rows visible',          make_top_mask(4)),
        ('top 8 rows visible',          make_top_mask(8)),
        ('top 16 rows visible',         make_top_mask(16)),
        ('bottom 8 rows visible',       make_bottom_mask(8)),
        ('left half visible',           make_left_half_mask()),
        ('50% random positions visible', make_random_mask(0.5, args.seed)),
    ]

    for sample_idx in range(args.n_samples):
        original = real[sample_idx]  # [SEQ_LEN]
        print()
        print('#' * 80)
        print(f'#  SAMPLE {sample_idx + 1}/{args.n_samples}  — real label: {labels[sample_idx]}')
        print('#' * 80)
        print()
        print('--- ORIGINAL (real training sample) ---')
        print(decode(original))

        for scenario_label, visible in scenarios:
            visible_dev = visible.to(args.device)
            initial = apply_visibility(original.unsqueeze(0), visible_dev)  # [1, SEQ_LEN]

            print()
            print('-' * 80)
            print(f'GENERATED: {scenario_label}  ({visible.sum().item()} positions visible)')
            print('-' * 80)

            result = generate(
                model,
                initial_tokens=initial,
                n_steps=args.n_steps,
                temperature=args.temperature,
            )
            print(decode(result[0]))


if __name__ == '__main__':
    main()
