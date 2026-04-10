"""Probe: does the model produce better output when primed with a real prefix?

Diagnostic for the "raster-order compounding errors" hypothesis. For each of N
real training examples, runs the trained model with several prefix lengths
(0, 4, 8, 16, 24 rows) and prints the generated continuations side by side.

- If outputs get dramatically better as the prefix grows, the model knows the
  distribution but struggles to bootstrap early tokens. Decoding improvements
  (diffusion, MaskGIT, beam search) would help.
- If outputs stay mediocre at all prefix lengths, the model hasn't learned
  sharp per-position predictions even with perfect context. Model/objective
  issue, not a decoding issue.

Usage (local):
    python probe_prefix.py --checkpoint checkpoints/v2_humans/step_19000.pt

Usage (Modal GPU):
    modal run modal_app.py::probe
"""

import argparse
import json
import random

import numpy as np
import torch

from data import (
    BG_TOKEN,
    GRID_H,
    GRID_W,
    SEQ_LEN,
    VOCAB_SIZE,
    build_char_lookup,
    tokenize_ascii,
)
from model import AsciiTransformer
from sample import decode, generate


FILTER_LABELS = frozenset({'Girl', 'Woman', 'Boy', 'Man'})

# Prefix lengths in rows (each row is GRID_W=64 tokens)
PREFIX_ROWS = [0, 4, 8, 16, 24]


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


def load_real_samples(jsonl_path: str, n: int, seed: int) -> torch.Tensor:
    """Return [n, SEQ_LEN] tensor of real human training samples, seeded."""
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
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=3)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device)

    print(f'Loading {args.n_samples} real samples from {args.train_path}...')
    real, labels = load_real_samples(args.train_path, args.n_samples, args.seed)
    real = real.to(args.device)
    print(f'  picked: {labels}')

    for sample_idx in range(args.n_samples):
        original = real[sample_idx:sample_idx + 1]  # [1, SEQ_LEN]
        print()
        print('#' * 80)
        print(f'#  SAMPLE {sample_idx + 1}/{args.n_samples}  — real label: {labels[sample_idx]}')
        print('#' * 80)
        print()
        print('--- ORIGINAL (real training sample) ---')
        print(decode(original[0]))

        for rows in PREFIX_ROWS:
            prefix_len = rows * GRID_W  # rows * 64
            print()
            print('-' * 80)
            if rows == 0:
                print(f'GENERATED: unconditional (no prefix)')
                result = generate(
                    model, n_samples=1,
                    temperature=args.temperature, top_k=args.top_k,
                )
            else:
                print(f'GENERATED: primed with first {rows} rows ({prefix_len} tokens)')
                prefix = original[:, :prefix_len]
                result = generate(
                    model, prefix=prefix,
                    temperature=args.temperature, top_k=args.top_k,
                )
            print('-' * 80)
            print(decode(result[0]))


if __name__ == '__main__':
    main()
