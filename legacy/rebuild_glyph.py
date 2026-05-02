"""
Re-process cached Open Images into fixed 64x32 ASCII canvas via glyph matching.

For each cell in the output, we select the printable ASCII character whose
rendered glyph best approximates the corresponding pixel block in the source
image (under MSE). This captures sub-cell structure like edges, corners, and
densities — dramatically more information than a plain brightness ramp.

Convention: dark pixels in the source become dense characters (e.g. @ # M).
Bright pixels become sparse characters (space, . , :). This is the standard
ASCII-art convention and is inverted from the previous rebuild script.

Usage:
    python rebuild_glyph.py \\
        --annotations openimages/train-annotations-object-segmentation.csv \\
        --masks-dir openimages/train-masks \\
        --images-dir openimages/train-images \\
        --output openimages/train_ascii_64x32_glyph.jsonl \\
        --workers 16
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import label as cc_label

# All 95 printable ASCII characters, ordered by ASCII code.
# Index 0 is space, which serves as the canvas background (BG_TOKEN).
CHARS = ''.join(chr(i) for i in range(32, 127))
N_CHARS = len(CHARS)  # 95

CANVAS_W = 64
CANVAS_H = 32
TARGET_W = 60   # max subject width inside canvas
TARGET_H = 30   # max subject height inside canvas
WIDTH_RATIO = 2.2
MIN_W = 20
MIN_H = 10

# Glyph bitmap dimensions. Each character is rendered to a GLYPH_H x GLYPH_W
# grayscale bitmap, and each cell in the canvas is matched against a block of
# the same size sampled from the source image.
GLYPH_H = 16
GLYPH_W = 8


def load_monospace_font(size: int = GLYPH_H):
    """Try a few common monospace fonts. Fall back to PIL default."""
    candidates = [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/Monaco.dfont',
        '/System/Library/Fonts/Courier.dfont',
        '/Library/Fonts/Courier New.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size), path
        except (OSError, IOError):
            continue
    return ImageFont.load_default(), '<PIL default>'


def build_glyph_codebook(font, chars: str = CHARS, width: int = GLYPH_W, height: int = GLYPH_H):
    """Render each character to a [height, width] grayscale bitmap.

    Returns: [N, height, width] float32 array where 1.0 = ink, 0.0 = background.
    """
    glyphs = np.zeros((len(chars), height, width), dtype=np.float32)
    for i, ch in enumerate(chars):
        img = Image.new('L', (width, height), color=255)  # white background
        draw = ImageDraw.Draw(img)
        # Draw character at origin. Monospace fonts generally advance predictably.
        draw.text((0, 0), ch, fill=0, font=font)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1], 1 = bg
        glyphs[i] = 1.0 - arr                           # invert: 1 = ink
    return glyphs


def load_classes(path: str) -> dict:
    classes = {}
    with open(path) as f:
        for row in csv.reader(f):
            classes[row[0]] = row[1]
    return classes


def passes_filter(row) -> bool:
    xmin, xmax = float(row['BoxXMin']), float(row['BoxXMax'])
    ymin, ymax = float(row['BoxYMin']), float(row['BoxYMax'])
    span = max(xmax - xmin, ymax - ymin)
    if span < 0.60:
        return False
    edge_count = sum([xmin < 0.02, xmax > 0.98, ymin < 0.02, ymax > 0.98])
    if edge_count > 1:
        return False
    return True


def fit_dimensions(w_px: int, h_px: int):
    """Compute (new_w, new_h) character dims for a subject of the given pixel
    dims, fitted into TARGET_W x TARGET_H while preserving aspect ratio."""
    fit_w_h = h_px * TARGET_W / (w_px * WIDTH_RATIO)
    if fit_w_h <= TARGET_H:
        return TARGET_W, max(1, int(round(fit_w_h)))
    new_w = TARGET_H * w_px * WIDTH_RATIO / h_px
    return max(1, int(round(new_w))), TARGET_H


def largest_component_mask(mask_arr):
    """Return mask keeping only the largest connected component."""
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


def match_glyphs(ink_blocks: np.ndarray, glyphs_flat: np.ndarray) -> np.ndarray:
    """Vectorized glyph matching under MSE.

    ink_blocks:  [N_cells, cell_h*cell_w] float32, 1 = dark source (ink)
    glyphs_flat: [N_chars, cell_h*cell_w] float32, 1 = character ink

    Returns: [N_cells] int32 of best-matching character indices.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a . b)
    a2 = (ink_blocks ** 2).sum(axis=1, keepdims=True)       # [N_cells, 1]
    b2 = (glyphs_flat ** 2).sum(axis=1, keepdims=True).T    # [1, N_chars]
    ab = ink_blocks @ glyphs_flat.T                         # [N_cells, N_chars]
    mse = a2 + b2 - 2 * ab
    return mse.argmin(axis=1).astype(np.int32)


def render_one(img_path: str, mask_path: str, glyphs_flat: np.ndarray):
    """Process a single image into a multi-line ASCII string, or None on failure."""
    try:
        img = Image.open(img_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L').resize(img.size, Image.LANCZOS)
        mask_arr = np.array(mask)

        mask_arr = largest_component_mask(mask_arr)
        if mask_arr is None:
            return None

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
        if max(span_h, span_w) < 0.60:
            return None
        edge_count = sum([
            rmin == 0, rmax == full_h - 1,
            cmin == 0, cmax == full_w - 1,
        ])
        if edge_count > 1:
            return None

        # Crop to bbox and paste onto white background
        img_crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        mask_crop = mask_arr[rmin:rmax + 1, cmin:cmax + 1]

        bg = Image.new('RGBA', img_crop.size, (255, 255, 255, 255))
        mask_crop_pil = Image.fromarray(mask_crop)
        bg.paste(img_crop, mask=mask_crop_pil)
        gray = bg.convert('L')
        gray_arr = np.array(gray)

        h_px, w_px = gray_arr.shape
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

        # Resize to (new_w * GLYPH_W, new_h * GLYPH_H) — each cell gets a
        # full glyph-sized pixel block to match against.
        target_w = new_w * GLYPH_W
        target_h = new_h * GLYPH_H
        resized = np.array(
            Image.fromarray(enh_arr).resize((target_w, target_h), Image.LANCZOS)
        )  # [target_h, target_w] uint8

        # Convert to "ink" convention: dark source = high ink, bright = low ink
        ink = (255 - resized.astype(np.float32)) / 255.0                    # [H, W]

        # Tile into per-cell blocks: [new_h, new_w, GLYPH_H, GLYPH_W]
        ink_blocks = ink.reshape(new_h, GLYPH_H, new_w, GLYPH_W).transpose(0, 2, 1, 3)
        ink_flat = ink_blocks.reshape(new_h * new_w, GLYPH_H * GLYPH_W)

        best = match_glyphs(ink_flat, glyphs_flat)                          # [new_h * new_w]
        result = best.reshape(new_h, new_w)

        # Center on 32x64 canvas with space (index 0) as background
        canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.int32)
        y_off = (CANVAS_H - new_h) // 2
        x_off = (CANVAS_W - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = result

        lines = [''.join(CHARS[i] for i in row) for row in canvas]
        return '\n'.join(lines)

    except Exception:
        return None


def process_one(row, classes, masks_dir, images_dir, glyphs_flat):
    img_id = row['ImageID']
    mask_file = row['MaskPath']
    label = classes.get(row['LabelName'], row['LabelName'])

    img_path = os.path.join(images_dir, f'{img_id}.jpg')
    mask_path = os.path.join(masks_dir, mask_file)

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    ascii_art = render_one(img_path, mask_path, glyphs_flat)
    if ascii_art is None:
        return None

    return {'label': label, 'ascii': ascii_art}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--masks-dir', required=True)
    parser.add_argument('--classes', default='openimages/oidv7-class-descriptions-boxable.csv')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    print('Loading font and building glyph codebook...')
    font, font_path = load_monospace_font()
    print(f'  font: {font_path}')
    glyphs = build_glyph_codebook(font)                       # [N_chars, GLYPH_H, GLYPH_W]
    glyphs_flat = glyphs.reshape(N_CHARS, -1).astype(np.float32)
    print(f'  glyph codebook: {glyphs.shape}, mean ink per char: {glyphs.mean():.3f}')

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
                executor.submit(process_one, row, classes, args.masks_dir, args.images_dir, glyphs_flat)
                for row in candidates
            ]
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    out_f.write(json.dumps(result) + '\n')
                    written += 1
                else:
                    failed += 1
                if (i + 1) % 5000 == 0:
                    print(f'  [{i+1}/{len(candidates)}] written={written} failed={failed}')

    print(f'\nDone! {written:,} examples written to {args.output}')
    print(f'  ({failed:,} skipped)')


if __name__ == '__main__':
    main()
