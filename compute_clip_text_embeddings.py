"""Compute CLIP TEXT embeddings from synthetic captions of JSONL labels.

For each example in the input JSONL, generates a caption like
"a photo of a {label}", encodes it via CLIP's text encoder, and saves the
result to a parallel .npy file aligned line-by-line with the JSONL.

This is the text-conditioning sibling of compute_clip_embeddings.py.
Same output shape, same alignment, but the conditioning signal comes from
CLIP's text branch instead of its image branch — eliminating the modality
gap that hurts text-prompt inference when the DiT was trained on image
embeddings.

Usage:
    python compute_clip_text_embeddings.py \\
        --jsonl openimages/train_ascii_128x64_relaxed.jsonl \\
        --output openimages/train_clip_text_128x64.npy
"""

import argparse
import json
import time

import numpy as np
import torch


CAPTION_TEMPLATE = "a photo of a {label}"


def get_device(prefer: str = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_clip(model_name: str, device: torch.device):
    from transformers import CLIPModel, CLIPTokenizer
    print(f'Loading CLIP: {model_name} on {device}...')
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    embed_dim = model.config.projection_dim
    print(f'  embed_dim = {embed_dim}')
    return model, tokenizer, embed_dim


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', required=True, help='Input JSONL with labels')
    p.add_argument('--output', required=True, help='Output .npy path')
    p.add_argument('--model', default='openai/clip-vit-base-patch32')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default=None,
                   help='Override device: cuda / mps / cpu. Default: auto.')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N lines (for testing).')
    p.add_argument('--template', default=CAPTION_TEMPLATE,
                   help='Caption template with {label} placeholder. '
                        'Used when an example has no "caption" field.')
    p.add_argument('--force-synthetic', action='store_true',
                   help='Always use the template, even if the JSONL has '
                        'real captions in a "caption" field.')
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, tokenizer, embed_dim = load_clip(args.model, device)

    print(f'Counting lines in {args.jsonl}...')
    with open(args.jsonl) as f:
        n_lines = sum(1 for _ in f)
    if args.limit is not None:
        n_lines = min(n_lines, args.limit)
    print(f'  {n_lines:,} lines')
    print(f'Caption template: "{args.template}"')

    embeddings = np.zeros((n_lines, embed_dim), dtype=np.float32)

    t_start = time.time()
    with open(args.jsonl) as f:
        batch_indices = []
        batch_captions = []
        line_idx = -1

        def flush_batch():
            if not batch_captions:
                return
            inputs = tokenizer(
                batch_captions,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
            feats = feats.float().cpu().numpy()
            for idx, vec in zip(batch_indices, feats):
                embeddings[idx] = vec
            batch_indices.clear()
            batch_captions.clear()

        for line in f:
            line_idx += 1
            if args.limit is not None and line_idx >= args.limit:
                break
            try:
                r = json.loads(line)
                if 'caption' in r and not args.force_synthetic:
                    # Real caption from COCO (or any other dataset)
                    caption = r['caption']
                else:
                    # Synthetic caption from class label
                    label = r.get('label', 'thing').lower()
                    caption = args.template.format(label=label)
                batch_captions.append(caption)
                batch_indices.append(line_idx)
            except Exception:
                # Leave row as zeros
                continue

            if len(batch_captions) >= args.batch_size:
                flush_batch()

            if (line_idx + 1) % 20000 == 0:
                elapsed = time.time() - t_start
                rate = (line_idx + 1) / elapsed
                eta = (n_lines - line_idx - 1) / max(rate, 1e-6)
                print(f'  [{line_idx+1}/{n_lines}] {rate:.1f} cap/s, eta={eta/60:.1f}m')

        flush_batch()

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'  Saving to {args.output}...')
    np.save(args.output, embeddings)
    print(f'  Saved {embeddings.shape} float32 array '
          f'({embeddings.nbytes / 1024 / 1024:.1f} MB)')


if __name__ == '__main__':
    main()
