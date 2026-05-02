"""
Build ASCII art training dataset from Open Images segmentation masks.

Pipeline:
1. Read segmentation annotations CSV
2. Filter: span >= 60%, edges <= 1
3. Download image from Flickr URL
4. Apply mask, normalize foreground contrast, enhance, convert to 8-char ASCII
5. Write to output JSONL: {"label": "Cat", "ascii": "..."}

Usage:
    python build_dataset.py --annotations train-annotations-object-segmentation.csv \
                            --masks-dir masks/ \
                            --image-urls validation-images-with-rotation.csv \
                            --output dataset.jsonl \
                            --workers 8
"""

import argparse
import csv
import json
import os
import sys
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFilter
import numpy as np

# 8-char density ramp (from lightest to darkest)
RAMP = ' "roy48Q'
COLUMNS = 64
WIDTH_RATIO = 2.2


def load_classes(path):
    classes = {}
    with open(path) as f:
        for row in csv.reader(f):
            classes[row[0]] = row[1]
    return classes


def load_urls(path):
    urls = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            urls[row['ImageID']] = row['Thumbnail300KURL'] or row['OriginalURL']
    return urls


def passes_filter(row):
    """Geometric filters: span >= 60%, edges <= 1."""
    xmin, xmax = float(row['BoxXMin']), float(row['BoxXMax'])
    ymin, ymax = float(row['BoxYMin']), float(row['BoxYMax'])
    span = max(xmax - xmin, ymax - ymin)
    if span < 0.60:
        return False
    edge_count = sum([xmin < 0.02, xmax > 0.98, ymin < 0.02, ymax > 0.98])
    if edge_count > 1:
        return False
    return True


def download_image(url, dest):
    """Download image, return True on success."""
    try:
        r = subprocess.run(
            ['curl', '-sfL', '--max-time', '10', url, '-o', dest],
            capture_output=True, timeout=15
        )
        return r.returncode == 0 and os.path.getsize(dest) > 0
    except:
        return False


def image_to_ascii(img_path, mask_path):
    """Convert masked image to ASCII art string."""
    try:
        img = Image.open(img_path).convert('RGBA')
        mask = Image.open(mask_path).convert('L').resize(img.size, Image.LANCZOS)

        # Apply mask onto white background
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, mask=mask)
        gray = bg.convert('L')

        gray_arr = np.array(gray)
        mask_arr = np.array(mask.resize(gray.size, Image.LANCZOS))

        # Foreground normalization (2nd-98th percentile)
        fg_pixels = gray_arr[mask_arr > 128].astype(float)
        if len(fg_pixels) < 100:
            return None
        lo, hi = np.percentile(fg_pixels, [2, 98])
        if hi - lo < 20:  # skip very low contrast
            return None

        norm = np.full_like(gray_arr, 255)
        fg_mask = mask_arr > 128
        stretched = np.clip((gray_arr.astype(float) - lo) / (hi - lo) * 255, 0, 255)
        norm[fg_mask] = stretched[fg_mask].astype(np.uint8)

        # Enhancement: local contrast + sharpen
        norm_pil = Image.fromarray(norm)
        enhanced = norm_pil.filter(ImageFilter.UnsharpMask(radius=10, percent=100, threshold=0))
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
        enh_arr = np.array(enhanced.convert('L'))
        enh_arr[mask_arr <= 128] = 255

        # Resize and convert to ASCII
        h, w = enh_arr.shape
        scalar = w * WIDTH_RATIO / COLUMNS
        new_w = int(w * WIDTH_RATIO / scalar)
        new_h = int(h / scalar)
        resized = np.array(Image.fromarray(enh_arr).resize((new_w, new_h), Image.LANCZOS))

        n = len(RAMP) - 1
        lines = []
        for row in resized:
            lines.append(''.join(RAMP[int((v / 255.0) * n)] for v in row))

        return '\n'.join(lines)
    except Exception as e:
        return None


def process_one(row, classes, urls, masks_dir, images_dir):
    """Process a single annotation. Returns (label, ascii) or None."""
    img_id = row['ImageID']
    mask_file = row['MaskPath']
    label = classes.get(row['LabelName'], row['LabelName'])

    mask_path = os.path.join(masks_dir, mask_file)
    if not os.path.exists(mask_path):
        return None

    url = urls.get(img_id)
    if not url:
        return None

    img_path = os.path.join(images_dir, f'{img_id}.jpg')

    # Download if needed
    if not os.path.exists(img_path):
        if not download_image(url, img_path):
            # Clean up failed download
            if os.path.exists(img_path):
                os.remove(img_path)
            return None

    ascii_art = image_to_ascii(img_path, mask_path)
    if ascii_art is None:
        return None

    return {'label': label, 'ascii': ascii_art}


def main():
    parser = argparse.ArgumentParser(description='Build ASCII art dataset from Open Images')
    parser.add_argument('--annotations', required=True, help='Segmentation annotations CSV')
    parser.add_argument('--masks-dir', required=True, help='Directory containing mask PNGs')
    parser.add_argument('--classes', default='openimages/oidv7-class-descriptions-boxable.csv')
    parser.add_argument('--image-urls', required=True, help='Image URLs CSV')
    parser.add_argument('--images-dir', default='openimages/images', help='Directory to cache downloaded images')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--workers', type=int, default=8, help='Download/processing threads')
    parser.add_argument('--limit', type=int, default=None, help='Max annotations to process (for testing)')
    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    print('Loading metadata...')
    classes = load_classes(args.classes)
    urls = load_urls(args.image_urls)

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

    print(f'  {total} total annotations, {len(candidates)} pass filters ({len(candidates)/max(total,1):.0%})')

    print(f'Processing with {args.workers} workers...')
    written = 0
    failed = 0
    with open(args.output, 'w') as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for row in candidates:
                future = executor.submit(
                    process_one, row, classes, urls, args.masks_dir, args.images_dir
                )
                futures[future] = row

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    out_f.write(json.dumps(result) + '\n')
                    written += 1
                else:
                    failed += 1

                if (i + 1) % 100 == 0:
                    print(f'  [{i+1}/{len(candidates)}] written={written} failed={failed}')

    print(f'\nDone! {written} examples written to {args.output}')
    print(f'  ({failed} failed downloads/conversions)')


if __name__ == '__main__':
    main()
