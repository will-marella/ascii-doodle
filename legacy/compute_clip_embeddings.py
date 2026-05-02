"""Compute CLIP image embeddings for each example in an ASCII training JSONL.

For each line in the input JSONL (which must have `image_id` and `bbox` fields
from the updated rebuild_fixed_canvas.py), this script:

  1. Loads the original photo from --images-dir
  2. Crops to the normalized bbox in pixel coordinates (no masking)
  3. Runs CLIP image encoder to get a 512-d embedding
  4. Writes the embedding to a parallel .npy file aligned line-by-line

The output .npy is a [N, 512] float32 array where row i corresponds to line i
of the input JSONL. Failed encodings are filled with zeros and logged.

Uses MPS on macOS Apple Silicon, CUDA on Linux/Windows GPU, CPU as fallback.

Usage:
    python compute_clip_embeddings.py \\
        --jsonl openimages/train_ascii_128x64_relaxed.jsonl \\
        --images-dir openimages/train-images \\
        --output openimages/train_clip_128x64.npy

Default model: openai/clip-vit-base-patch32 (~150 MB, 512-d embeddings).
Pulled lazily from HuggingFace on first run.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image


def get_device(prefer: str = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_clip(model_name: str, device: torch.device):
    """Load a CLIP model + processor lazily so the dependency is only
    imported when this script actually runs."""
    from transformers import CLIPModel, CLIPProcessor
    print(f'Loading CLIP: {model_name} on {device}...')
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    processor = CLIPProcessor.from_pretrained(model_name)
    embed_dim = model.config.projection_dim
    print(f'  embed_dim = {embed_dim}')
    return model, processor, embed_dim


def crop_to_bbox(img: Image.Image, bbox: list) -> Image.Image:
    """Crop a PIL image to a normalized [xmin, ymin, xmax, ymax] bbox."""
    w, h = img.size
    xmin, ymin, xmax, ymax = bbox
    px = (
        max(0, int(round(xmin * w))),
        max(0, int(round(ymin * h))),
        min(w, int(round(xmax * w))),
        min(h, int(round(ymax * h))),
    )
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return img.crop(px)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', required=True, help='Input JSONL with image_id + bbox')
    p.add_argument('--images-dir', required=True, help='Where to find source photos')
    p.add_argument('--output', required=True, help='Output .npy path')
    p.add_argument('--model', default='openai/clip-vit-base-patch32')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None,
                   help='Override device: cuda / mps / cpu. Default: auto.')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N lines (for testing).')
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, processor, embed_dim = load_clip(args.model, device)

    # Count lines in JSONL so we can preallocate
    print(f'Counting lines in {args.jsonl}...')
    with open(args.jsonl) as f:
        n_lines = sum(1 for _ in f)
    if args.limit is not None:
        n_lines = min(n_lines, args.limit)
    print(f'  {n_lines:,} lines')

    embeddings = np.zeros((n_lines, embed_dim), dtype=np.float32)
    n_failed = 0

    t_start = time.time()
    with open(args.jsonl) as f:
        batch_indices = []
        batch_imgs = []
        line_idx = -1

        def flush_batch():
            nonlocal n_failed
            if not batch_imgs:
                return
            inputs = processor(images=batch_imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_image_features(**inputs)  # [B, embed_dim]
            feats = feats.float().cpu().numpy()
            for idx, vec in zip(batch_indices, feats):
                embeddings[idx] = vec
            batch_indices.clear()
            batch_imgs.clear()

        for line in f:
            line_idx += 1
            if args.limit is not None and line_idx >= args.limit:
                break

            try:
                r = json.loads(line)
                img_id = r['image_id']
                bbox = r['bbox']
                img_path = os.path.join(args.images_dir, f'{img_id}.jpg')
                if not os.path.exists(img_path):
                    n_failed += 1
                    continue
                img = Image.open(img_path).convert('RGB')
                cropped = crop_to_bbox(img, bbox)
                if cropped is None:
                    n_failed += 1
                    continue
                batch_imgs.append(cropped)
                batch_indices.append(line_idx)
            except Exception:
                n_failed += 1
                continue

            if len(batch_imgs) >= args.batch_size:
                flush_batch()

            if (line_idx + 1) % 5000 == 0:
                elapsed = time.time() - t_start
                rate = (line_idx + 1) / elapsed
                eta = (n_lines - line_idx - 1) / max(rate, 1e-6)
                print(f'  [{line_idx+1}/{n_lines}] {rate:.1f} img/s, '
                      f'failed={n_failed}, eta={eta/60:.1f}m')

        flush_batch()

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'  {n_lines - n_failed:,} embeddings computed, {n_failed:,} failed')
    print(f'  Saving to {args.output}...')
    np.save(args.output, embeddings)
    print(f'  Saved {embeddings.shape} float32 array '
          f'({embeddings.nbytes / 1024 / 1024:.1f} MB)')


if __name__ == '__main__':
    main()
