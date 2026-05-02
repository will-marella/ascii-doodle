"""
Re-process cached Open Images into fixed 64x32 ASCII canvas.

Reads cached images from --images-dir, applies mask, crops to subject bbox,
fits into 60x30 char target preserving aspect ratio, centers on 64x32 canvas.

Usage:
    python rebuild_fixed_canvas.py \
        --annotations openimages/train-annotations-object-segmentation.csv \
        --masks-dir openimages/train-masks \
        --images-dir openimages/train-images \
        --output openimages/train_ascii_64x32.jsonl \
        --workers 16
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import label as cc_label

# Standard ASCII-art density ramp: light → dark.
# Background = space (index 0) = empty/copy-pasteable.
# Foreground/dark pixels = dense characters (#, @).
RAMP = ' .:-+*#@'

# Canvas dimensions — override via --canvas-size (e.g. "128x64" or "256x128")
CANVAS_W = 64
CANVAS_H = 32
TARGET_W = 60   # max subject width inside canvas (auto-scaled if canvas overridden)
TARGET_H = 30   # max subject height inside canvas
WIDTH_RATIO = 2.2
MIN_W = 20
MIN_H = 10


def load_classes(path):
    classes = {}
    with open(path) as f:
        for row in csv.reader(f):
            classes[row[0]] = row[1]
    return classes


def passes_filter(row, min_span: float = 0.40):
    """Filter annotations to subjects that span at least `min_span` of the image
    and don't touch more than one edge.

    Default min_span is 0.40 (relaxed from the original 0.60). Smaller subjects
    render with more padding around them in the canvas, but that's still useful
    training data and gives us ~50% more examples.
    """
    xmin, xmax = float(row['BoxXMin']), float(row['BoxXMax'])
    ymin, ymax = float(row['BoxYMin']), float(row['BoxYMax'])
    span = max(xmax - xmin, ymax - ymin)
    if span < min_span:
        return False
    edge_count = sum([xmin < 0.02, xmax > 0.98, ymin < 0.02, ymax > 0.98])
    if edge_count > 1:
        return False
    return True


def fit_dimensions(w_px, h_px):
    """Compute (new_w, new_h) char dims for a subject of given pixel dims,
    fitted into TARGET_W x TARGET_H while preserving aspect ratio."""
    # If we constrain width to TARGET_W:
    fit_w_h = h_px * TARGET_W / (w_px * WIDTH_RATIO)
    if fit_w_h <= TARGET_H:
        return TARGET_W, max(1, int(round(fit_w_h)))
    # Otherwise constrain height to TARGET_H:
    new_w = TARGET_H * w_px * WIDTH_RATIO / h_px
    return max(1, int(round(new_w))), TARGET_H


def largest_component_mask(mask_arr):
    """Return mask keeping only the largest connected component.
    Returns None if no foreground, or the unchanged mask if only one component."""
    binary = mask_arr > 128
    if not binary.any():
        return None
    labeled, num = cc_label(binary)
    if num <= 1:
        return mask_arr
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest = int(sizes.argmax())
    return np.where(labeled == largest, mask_arr, 0).astype(np.uint8)


def render_one(img_path, mask_path, img_w=None, img_h=None):
    """Open an image+mask file pair and render to ASCII (or None on failure)."""
    try:
        img = Image.open(img_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L').resize(img.size, Image.LANCZOS)
        mask_arr = np.array(mask)
        return render_masked_image(img, mask_arr)
    except Exception:
        return None


def render_masked_image(img, mask_arr):
    """Core ASCII rendering for a (PIL RGBA image, mask numpy array) pair.

    Shared rendering pipeline used by both rebuild_fixed_canvas (Open Images,
    mask files on disk) and rebuild_coco (COCO, mask arrays decoded from
    RLE/polygon segmentations in JSON). Identical filters, normalization, and
    ASCII conversion across both.

    Returns multi-line ASCII string, or None if any filter rejects the example.
    """
    try:
        # Pick largest connected component
        mask_arr = largest_component_mask(mask_arr)
        if mask_arr is None:
            return None

        # Re-check filters on the largest component's bbox
        fg = mask_arr > 128
        if not fg.any():
            return None
        rows = np.any(fg, axis=1)
        cols = np.any(fg, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        full_h, full_w = mask_arr.shape
        span_h = (rmax - rmin + 1) / full_h
        span_w = (cmax - cmin + 1) / full_w
        # Match the relaxed filter from passes_filter (was hardcoded 0.60)
        if max(span_h, span_w) < 0.40:
            return None
        edge_count = sum([
            rmin == 0, rmax == full_h - 1,
            cmin == 0, cmax == full_w - 1,
        ])
        if edge_count > 1:
            return None

        # Crop to bbox
        img_crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        mask_crop = mask_arr[rmin:rmax + 1, cmin:cmax + 1]

        # Apply mask onto white background
        bg = Image.new('RGBA', img_crop.size, (255, 255, 255, 255))
        mask_crop_pil = Image.fromarray(mask_crop)
        bg.paste(img_crop, mask=mask_crop_pil)
        gray = bg.convert('L')
        gray_arr = np.array(gray)

        h_px, w_px = gray_arr.shape

        # Fit to char dims
        new_w, new_h = fit_dimensions(w_px, h_px)
        if new_w < MIN_W or new_h < MIN_H:
            return None

        # Foreground normalization
        fg_pixels = gray_arr[mask_crop > 128].astype(float)
        if len(fg_pixels) < 50:
            return None
        lo, hi = np.percentile(fg_pixels, [2, 98])
        if hi - lo < 20:
            return None

        norm = np.full_like(gray_arr, 255)
        fg_mask = mask_crop > 128
        stretched = np.clip((gray_arr.astype(float) - lo) / (hi - lo) * 255, 0, 255)
        norm[fg_mask] = stretched[fg_mask].astype(np.uint8)

        # Enhance
        norm_pil = Image.fromarray(norm)
        enhanced = norm_pil.filter(ImageFilter.UnsharpMask(radius=10, percent=100, threshold=0))
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
        enh_arr = np.array(enhanced.convert('L'))
        enh_arr[mask_crop <= 128] = 255

        # Resize to (new_w, new_h)
        resized = np.array(Image.fromarray(enh_arr).resize((new_w, new_h), Image.LANCZOS))

        # Center on canvas. Fill with white (255) — represents "no subject"
        # which maps to the lightest character (space) under the inverted ramp.
        canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        y_off = (CANVAS_H - new_h) // 2
        x_off = (CANVAS_W - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Convert to ASCII. Inverted convention:
        # dark source pixels (low v) -> dense characters (high index)
        # bright source pixels (high v) -> sparse characters (low index)
        n = len(RAMP) - 1
        lines = []
        for row in canvas:
            lines.append(''.join(RAMP[int(((255 - v) / 255.0) * n)] for v in row))

        return '\n'.join(lines)
    except Exception:
        return None


def process_one(row, classes, masks_dir, images_dir):
    img_id = row['ImageID']
    mask_file = row['MaskPath']
    label = classes.get(row['LabelName'], row['LabelName'])

    img_path = os.path.join(images_dir, f'{img_id}.jpg')
    mask_path = os.path.join(masks_dir, mask_file)

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    ascii_art = render_one(img_path, mask_path)
    if ascii_art is None:
        return None

    # Include image_id and bbox so a downstream script can compute CLIP
    # embeddings of the bbox-cropped photo. bbox is normalized [xmin, ymin, xmax, ymax].
    return {
        'label': label,
        'ascii': ascii_art,
        'image_id': img_id,
        'bbox': [
            float(row['BoxXMin']),
            float(row['BoxYMin']),
            float(row['BoxXMax']),
            float(row['BoxYMax']),
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--masks-dir', required=True)
    parser.add_argument('--classes', default='openimages/oidv7-class-descriptions-boxable.csv')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--canvas-size', default=None,
                        help='WxH canvas override, e.g. "128x64" or "256x128". '
                             'Default: 64x32 (original).')
    args = parser.parse_args(argv)

    # Allow canvas size override for higher-resolution renders
    global CANVAS_W, CANVAS_H, TARGET_W, TARGET_H, MIN_W, MIN_H
    if args.canvas_size:
        w, h = args.canvas_size.split('x')
        CANVAS_W, CANVAS_H = int(w), int(h)
        TARGET_W = CANVAS_W - 4          # leave 2-char padding each side
        TARGET_H = CANVAS_H - 2          # leave 1-row padding top/bottom
        MIN_W = max(20, CANVAS_W // 3)
        MIN_H = max(10, CANVAS_H // 3)
        print(f'Canvas override: {CANVAS_W}x{CANVAS_H} (target {TARGET_W}x{TARGET_H})')

    print('Loading classes...')
    classes = load_classes(args.classes)

    print('Filtering annotations...')
    candidates = []
    total = 0
    with open(args.annotations) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if passes_filter(row):
                candidates.append(row)
            if args.limit and len(candidates) >= args.limit:
                break

    print(f'  {total:,} total, {len(candidates):,} pass filters')
    print(f'Processing with {args.workers} workers...')

    written = 0
    failed = 0
    with open(args.output, 'w') as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_one, row, classes, args.masks_dir, args.images_dir)
                for row in candidates
            ]
            # Iterate in submission order so the JSONL is deterministically
            # aligned with `candidates` (needed for parallel CLIP embedding files).
            for i, future in enumerate(futures):
                result = future.result()
                if result:
                    out_f.write(json.dumps(result) + '\n')
                    written += 1
                else:
                    failed += 1
                if (i + 1) % 5000 == 0:
                    print(f'  [{i+1}/{len(candidates)}] written={written} failed={failed}')

    print(f'\nDone! {written:,} examples written to {args.output}')
    print(f'  ({failed:,} skipped: not cached, too small, or low contrast)')


if __name__ == '__main__':
    main()
