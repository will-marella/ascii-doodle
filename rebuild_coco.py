"""Render COCO 2017 instances to ASCII canvases.

For each annotated instance in COCO, this script:
  1. Loads the source image
  2. Decodes the segmentation mask (RLE or polygon)
  3. Filters by bbox span (same threshold as Open Images: >= 40%)
  4. Renders to a 128x64 ASCII canvas using the same masked-rendering pipeline
     as rebuild_fixed_canvas.py (shared `render_masked_image()` function)
  5. Looks up the COCO image-level captions and pairs them with the instance
  6. Writes a JSONL row per (instance, caption) pair

Output JSONL format (one row per example):
    {
      "label": "person",                # COCO category name
      "ascii": "...",                   # multi-line ASCII (newline-separated)
      "image_id": 123456,
      "instance_id": 7890,
      "bbox": [xmin, ymin, xmax, ymax], # normalized
      "caption": "a young woman holding a guitar"   # one of the 5 captions
    }

By default we emit one row per instance, picking the first caption. Pass
--captions-per-instance N to emit N rows per instance with different captions
(useful for caption diversity at training time, since the ASCII is identical
across the N rows).

Usage:
    python rebuild_coco.py \\
        --coco-dir coco \\
        --split train \\
        --output openimages/train_ascii_coco_128x64.jsonl \\
        --canvas-size 128x64 \\
        --workers 16
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# Reuse the shared rendering pipeline + canvas constants from the OI rebuild
sys.path.insert(0, '.')
import rebuild_fixed_canvas as rfc
from rebuild_fixed_canvas import render_masked_image


def load_coco_annotations(coco_dir, split):
    """Load COCO instances + captions JSONs and return dicts we can iterate."""
    instances_path = os.path.join(coco_dir, f'annotations/instances_{split}2017.json')
    captions_path = os.path.join(coco_dir, f'annotations/captions_{split}2017.json')

    print(f'Loading {instances_path}...')
    with open(instances_path) as f:
        instances = json.load(f)
    print(f'  {len(instances["images"]):,} images')
    print(f'  {len(instances["annotations"]):,} instance annotations')
    print(f'  {len(instances["categories"]):,} categories')

    print(f'Loading {captions_path}...')
    with open(captions_path) as f:
        captions_data = json.load(f)
    print(f'  {len(captions_data["annotations"]):,} caption annotations')

    # Build lookup tables
    images_by_id = {img['id']: img for img in instances['images']}
    cat_id_to_name = {c['id']: c['name'] for c in instances['categories']}

    # Group captions by image_id
    captions_by_image = {}
    for cap in captions_data['annotations']:
        captions_by_image.setdefault(cap['image_id'], []).append(cap['caption'].strip())

    return instances['annotations'], images_by_id, cat_id_to_name, captions_by_image


def decode_segmentation(seg, height, width):
    """Decode a COCO segmentation field into a binary mask numpy array.

    Handles both polygon (list of polygons) and RLE (dict with 'counts')
    formats. Returns a uint8 array of shape (height, width) with values
    0 or 255 (compatible with rebuild_fixed_canvas pipeline).
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise RuntimeError(
            'pycocotools is required to decode COCO segmentations. '
            'Install with: pip install pycocotools'
        )

    if isinstance(seg, list):
        # Polygon segmentation: list of polygons (each a flat [x1,y1,x2,y2,...] list)
        if len(seg) == 0:
            return None
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)  # [H, W] uint8 (0 or 1)
    elif isinstance(seg, dict):
        # RLE segmentation
        # If 'counts' is a list, it's uncompressed RLE; if bytes/str, compressed
        if isinstance(seg.get('counts'), list):
            rle = mask_utils.frPyObjects(seg, height, width)
        else:
            rle = seg
        m = mask_utils.decode(rle)
    else:
        return None

    # Some annotations have multiple components — frPyObjects already merges,
    # but for robustness ensure we have a single 2D mask.
    if m.ndim == 3:
        m = m.any(axis=-1).astype(np.uint8)

    return (m * 255).astype(np.uint8)


def passes_bbox_filter(bbox_pixels, img_w, img_h, min_span=0.40):
    """Match the relaxed filter from rebuild_fixed_canvas.passes_filter."""
    x, y, w, h = bbox_pixels
    if w <= 0 or h <= 0:
        return False
    xmin = x / img_w
    ymin = y / img_h
    xmax = (x + w) / img_w
    ymax = (y + h) / img_h
    span = max(xmax - xmin, ymax - ymin)
    if span < min_span:
        return False
    edge_count = sum([xmin < 0.02, xmax > 0.98, ymin < 0.02, ymax > 0.98])
    if edge_count > 1:
        return False
    return True


def process_one(ann, images_by_id, cat_id_to_name, captions_by_image,
                images_dir, captions_per_instance):
    """Render a single COCO annotation to one or more JSONL records."""
    img_info = images_by_id.get(ann['image_id'])
    if img_info is None:
        return []

    img_w = img_info['width']
    img_h = img_info['height']
    bbox = ann['bbox']  # [x, y, w, h] in pixels

    if not passes_bbox_filter(bbox, img_w, img_h):
        return []

    img_path = os.path.join(images_dir, img_info['file_name'])
    if not os.path.exists(img_path):
        return []

    try:
        img = Image.open(img_path).convert('RGBA')
    except Exception:
        return []

    try:
        mask_arr = decode_segmentation(ann['segmentation'], img_h, img_w)
    except Exception:
        return []
    if mask_arr is None or not mask_arr.any():
        return []

    ascii_str = render_masked_image(img, mask_arr)
    if ascii_str is None:
        return []

    label = cat_id_to_name.get(ann['category_id'], 'unknown')
    captions = captions_by_image.get(ann['image_id'], [])
    if not captions:
        captions = [f'a photo of a {label}']

    # Take up to N captions for this instance, repeating the ASCII per caption
    captions = captions[:captions_per_instance]
    xmin = bbox[0] / img_w
    ymin = bbox[1] / img_h
    xmax = (bbox[0] + bbox[2]) / img_w
    ymax = (bbox[1] + bbox[3]) / img_h

    records = []
    for cap in captions:
        records.append({
            'label': label,
            'ascii': ascii_str,
            'image_id': ann['image_id'],
            'instance_id': ann['id'],
            'bbox': [xmin, ymin, xmax, ymax],
            'caption': cap,
        })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-dir', default='coco',
                        help='Directory containing train2017/ and annotations/')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--output', required=True, help='Output JSONL path')
    parser.add_argument('--canvas-size', default='128x64',
                        help='WxH canvas size (must match downstream training).')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N annotations (for testing).')
    parser.add_argument('--captions-per-instance', type=int, default=1,
                        help='How many captions to emit per instance. COCO has 5; '
                             'using >1 increases caption diversity at the cost of '
                             'duplicating ASCII rows.')
    args = parser.parse_args(argv)

    # Apply canvas size override to the rfc module so render_masked_image uses it
    if args.canvas_size:
        w, h = args.canvas_size.split('x')
        rfc.CANVAS_W, rfc.CANVAS_H = int(w), int(h)
        rfc.TARGET_W = rfc.CANVAS_W - 4
        rfc.TARGET_H = rfc.CANVAS_H - 2
        rfc.MIN_W = max(20, rfc.CANVAS_W // 3)
        rfc.MIN_H = max(10, rfc.CANVAS_H // 3)
        print(f'Canvas: {rfc.CANVAS_W}x{rfc.CANVAS_H} (target {rfc.TARGET_W}x{rfc.TARGET_H})')

    images_dir = os.path.join(args.coco_dir, f'{args.split}2017')
    if not os.path.exists(images_dir):
        raise SystemExit(f'Missing image dir: {images_dir}. Run download_coco.py first.')

    annotations, images_by_id, cat_id_to_name, captions_by_image = load_coco_annotations(
        args.coco_dir, args.split,
    )

    # Pre-filter to annotations whose bbox passes the span check (cheap)
    print('Pre-filtering annotations by bbox span...')
    candidates = []
    for ann in annotations:
        img_info = images_by_id.get(ann['image_id'])
        if img_info is None:
            continue
        if not passes_bbox_filter(ann['bbox'], img_info['width'], img_info['height']):
            continue
        candidates.append(ann)
        if args.limit is not None and len(candidates) >= args.limit:
            break
    print(f'  {len(candidates):,} annotations pass bbox filter')

    print(f'Processing with {args.workers} workers '
          f'({args.captions_per_instance} caption(s) per instance)...')
    written = 0
    failed = 0
    t_start = time.time()
    with open(args.output, 'w') as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_one, ann, images_by_id, cat_id_to_name, captions_by_image,
                    images_dir, args.captions_per_instance,
                )
                for ann in candidates
            ]
            for i, future in enumerate(futures):
                try:
                    records = future.result()
                except Exception:
                    records = []
                if records:
                    for r in records:
                        out_f.write(json.dumps(r) + '\n')
                    written += len(records)
                else:
                    failed += 1
                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(candidates) - i - 1) / max(rate, 1e-6)
                    print(f'  [{i+1}/{len(candidates)}] written={written} failed={failed} '
                          f'({rate:.0f}/s, eta={eta/60:.0f}m)')

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'  {written:,} examples written to {args.output}')
    print(f'  {failed:,} annotations skipped (mask/render failure)')


if __name__ == '__main__':
    main()
