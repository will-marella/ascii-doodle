"""Build the full QuickDraw dataset as a memory-mappable numpy array.

Downloads all 345 QuickDraw categories (28x28 bitmaps), resizes each
drawing to the target ASCII canvas with aspect-ratio correction, quantizes
to the 8-level RAMP, and saves:

  - tokens.npy:  [N, grid_h, grid_w] uint8  (token IDs 0-7)
  - labels.npy:  [N] int16                  (category index)
  - categories.json: list of category names  (index → name)

These are memory-mapped during training for zero-copy random access.

Usage:
    python build_quickdraw_full.py --output-dir quickdraw/full_32x16 --canvas-size 32x16
    python build_quickdraw_full.py --output-dir quickdraw/full_32x16 --canvas-size 32x16 --workers 8
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

QUICKDRAW_BITMAP_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy'
CACHE_DIR = 'quickdraw_cache'
WIDTH_RATIO = 2.2

# All 345 QuickDraw categories
# Source: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt
ALL_CATEGORIES = None  # loaded dynamically


def get_categories():
    """Fetch the category list from GitHub."""
    url = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'
    cache_path = os.path.join(CACHE_DIR, '_categories.txt')
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        print(f'Downloading category list...')
        urllib.request.urlretrieve(url, cache_path)
    with open(cache_path) as f:
        return [line.strip() for line in f if line.strip()]


def download_bitmap(category):
    """Download the 28x28 numpy bitmap for a category. Returns path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = category.replace(' ', '_').replace('/', '_')
    path = os.path.join(CACHE_DIR, f'{safe_name}.npy')
    if not os.path.exists(path):
        url = QUICKDRAW_BITMAP_URL.format(urllib.parse.quote(category))
        urllib.request.urlretrieve(url, path)
    return path


def render_category(category, canvas_w, canvas_h):
    """Download + render one category to token arrays.

    Returns (tokens_array [N, H, W] uint8, count).
    """
    path = download_bitmap(category)
    bitmaps = np.load(path)  # [N, 784] uint8, 0=bg, 255=ink
    n = len(bitmaps)

    target_w = max(1, canvas_w - 2)
    target_h = max(1, canvas_h - 1)

    all_tokens = np.zeros((n, canvas_h, canvas_w), dtype=np.uint8)

    for i in range(n):
        img_arr = bitmaps[i].reshape(28, 28)

        # Find bounding box of ink
        fg = img_arr > 20
        if not fg.any():
            continue
        rows = np.any(fg, axis=1)
        cols = np.any(fg, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        crop = Image.fromarray(img_arr[rmin:rmax+1, cmin:cmax+1])
        cw_px, ch_px = crop.size

        # Fit to canvas with aspect ratio correction
        fit_h = ch_px * target_w / (cw_px * WIDTH_RATIO)
        if fit_h <= target_h:
            new_w, new_h = target_w, max(1, int(round(fit_h)))
        else:
            new_w = max(1, int(round(target_h * cw_px * WIDTH_RATIO / ch_px)))
            new_h = target_h

        resized = np.array(crop.resize((new_w, new_h), Image.LANCZOS))

        # Center on canvas (all zeros = background)
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y_off = (canvas_h - new_h) // 2
        x_off = (canvas_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Quantize to 8 RAMP levels (0=space/bg, 7=darkest)
        # Bitmap: 0=bg, 255=ink. Map to 0-7.
        tokens = (canvas.astype(np.float32) / 255.0 * 7).round().astype(np.uint8)
        tokens = np.clip(tokens, 0, 7)
        all_tokens[i] = tokens

    return all_tokens, n


def process_one(args_tuple):
    """Worker function for parallel processing."""
    cat_idx, category, canvas_w, canvas_h = args_tuple
    try:
        tokens, count = render_category(category, canvas_w, canvas_h)
        return cat_idx, category, tokens, count
    except Exception as e:
        print(f'  ERROR: {category}: {e}')
        return cat_idx, category, None, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--canvas-size', default='32x16')
    p.add_argument('--workers', type=int, default=4,
                   help='Parallel download+render workers')
    p.add_argument('--categories', nargs='*', default=None,
                   help='Subset of categories to process. Default: all 345.')
    args = p.parse_args()

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    all_cats = args.categories or get_categories()
    print(f'Categories: {len(all_cats)}')
    print(f'Canvas: {cw}x{ch}')
    print(f'Workers: {args.workers}')

    os.makedirs(args.output_dir, exist_ok=True)

    # First pass: download + render each category, save per-category chunks
    # to avoid holding 24GB in memory at once. Then concatenate via mmap.
    chunk_dir = os.path.join(args.output_dir, '_chunks')
    os.makedirs(chunk_dir, exist_ok=True)

    category_names = []
    category_counts = []

    # Process sequentially to control memory (downloads are the bottleneck anyway)
    for i, cat in enumerate(all_cats):
        try:
            tokens, count = render_category(cat, cw, ch)
        except Exception as e:
            print(f'  [{i+1}/{len(all_cats)}] {cat}: FAILED ({e})')
            continue
        safe = cat.replace(' ', '_').replace('/', '_')
        np.save(os.path.join(chunk_dir, f'{safe}.npy'), tokens)
        category_names.append(cat)
        category_counts.append(count)
        print(f'  [{i+1}/{len(all_cats)}] {cat}: {count:,}')
        # Delete cached bitmap to free disk space
        cache_path = os.path.join(CACHE_DIR, f'{safe}.npy')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    total = sum(category_counts)
    print(f'\nTotal: {total:,} examples across {len(category_names)} categories')

    # Concatenate chunks into final arrays
    print('Concatenating chunks...')
    tokens_path = os.path.join(args.output_dir, 'tokens.npy')
    labels_path = os.path.join(args.output_dir, 'labels.npy')
    cats_path = os.path.join(args.output_dir, 'categories.json')

    # Pre-allocate output file as mmap
    tokens_arr = np.lib.format.open_memmap(
        tokens_path, mode='w+', dtype=np.uint8, shape=(total, ch, cw),
    )
    labels_arr = np.zeros(total, dtype=np.int16)

    offset = 0
    for cat_idx, (cat, count) in enumerate(zip(category_names, category_counts)):
        safe = cat.replace(' ', '_').replace('/', '_')
        chunk = np.load(os.path.join(chunk_dir, f'{safe}.npy'))
        tokens_arr[offset:offset+count] = chunk
        labels_arr[offset:offset+count] = cat_idx
        offset += count
        # Delete chunk after copying
        os.remove(os.path.join(chunk_dir, f'{safe}.npy'))

    del tokens_arr  # flush mmap
    np.save(labels_path, labels_arr)

    with open(cats_path, 'w') as f:
        json.dump(category_names, f, indent=2)

    # Clean up chunk dir
    try:
        os.rmdir(chunk_dir)
    except OSError:
        pass

    print(f'\nSaved to {args.output_dir}/')
    print(f'  tokens.npy: ({total}, {ch}, {cw}) = {os.path.getsize(tokens_path) / 1024**3:.1f} GB')
    print(f'  labels.npy: ({total},) = {os.path.getsize(labels_path) / 1024**2:.1f} MB')
    print(f'  categories.json: {len(category_names)} categories')

    # Pre-compute CLIP text embeddings for all categories
    clip_path = os.path.join(args.output_dir, 'clip_embeddings.npy')
    try:
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        print(f'\nComputing CLIP text embeddings for {len(category_names)} categories...')
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        clip_model.eval()
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        prompts = [f'a drawing of a {cat}' for cat in category_names]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embs = clip_model.get_text_features(**inputs).float().numpy()
        np.save(clip_path, embs)
        print(f'  clip_embeddings.npy: {embs.shape} ({embs.nbytes / 1024:.1f} KB)')
        del clip_model, tokenizer
    except ImportError:
        print('  (skipping CLIP — transformers not installed)')

    # Print per-category counts
    print(f'\nPer-category counts:')
    for i, cat in enumerate(category_names):
        n = (labels_arr == i).sum()
        print(f'  {cat:<30} {n:>8,}')

    print('\nDone.')


if __name__ == '__main__':
    main()
