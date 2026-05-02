"""Resume an interrupted/incomplete rebuild_fixed_canvas.py run.

Reads the existing output JSONL, builds a set of (image_id, bbox) tuples for
already-processed examples, then iterates through the annotations CSV and
processes only candidates that AREN'T already in the JSONL. New examples are
APPENDED to the existing file.

Use this after fixing a bug in the rendering pipeline (e.g. the 0.60 internal
mask filter that we just relaxed to 0.40) — it lets you add the previously-
rejected examples without re-doing the work for examples that already passed.

Usage:
    python rebuild_append.py \\
        --annotations openimages/train-annotations-object-segmentation.csv \\
        --masks-dir openimages/train-masks \\
        --images-dir openimages/train-images \\
        --output openimages/train_ascii_128x64_relaxed.jsonl \\
        --canvas-size 128x64 \\
        --workers 16
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '.')
import rebuild_fixed_canvas as rfc
from rebuild_fixed_canvas import (
    load_classes,
    passes_filter,
    process_one,
)


def bbox_key(xmin, ymin, xmax, ymax, precision: int = 6):
    """Create a hashable bbox key with rounded floats for comparison."""
    return (
        round(float(xmin), precision),
        round(float(ymin), precision),
        round(float(xmax), precision),
        round(float(ymax), precision),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--masks-dir', required=True)
    parser.add_argument('--classes', default='openimages/oidv7-class-descriptions-boxable.csv')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output', required=True,
                        help='Existing JSONL — new examples are appended.')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--canvas-size', default=None,
                        help='Must match the original run. e.g. "128x64".')
    parser.add_argument('--min-span', type=float, default=0.40,
                        help='Bbox span filter (matches passes_filter default).')
    args = parser.parse_args()

    # Apply canvas-size override to the rfc module if provided
    if args.canvas_size:
        w, h = args.canvas_size.split('x')
        rfc.CANVAS_W, rfc.CANVAS_H = int(w), int(h)
        rfc.TARGET_W = rfc.CANVAS_W - 4
        rfc.TARGET_H = rfc.CANVAS_H - 2
        rfc.MIN_W = max(20, rfc.CANVAS_W // 3)
        rfc.MIN_H = max(10, rfc.CANVAS_H // 3)
        print(f'Canvas: {rfc.CANVAS_W}x{rfc.CANVAS_H} (target {rfc.TARGET_W}x{rfc.TARGET_H})')

    # ---- Load existing JSONL into a "seen" set ----
    if not os.path.exists(args.output):
        print(f'Error: {args.output} does not exist. '
              f'Use rebuild_fixed_canvas.py for a fresh run.')
        sys.exit(1)

    print(f'Loading existing examples from {args.output}...')
    seen = set()
    n_existing = 0
    with open(args.output) as f:
        for line in f:
            r = json.loads(line)
            if 'image_id' not in r or 'bbox' not in r:
                print(f'Error: existing JSONL is missing image_id/bbox fields. '
                      f'Cannot resume — re-run rebuild_fixed_canvas.py from scratch.')
                sys.exit(1)
            seen.add((r['image_id'], bbox_key(*r['bbox'])))
            n_existing += 1
    print(f'  {n_existing:,} examples already in JSONL')

    # ---- Load classes ----
    print('Loading class names...')
    classes = load_classes(args.classes)

    # ---- Iterate annotations, find candidates not yet processed ----
    print('Filtering annotations + checking against seen set...')
    new_candidates = []
    n_total = 0
    n_filtered = 0
    n_already_done = 0
    with open(args.annotations) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if not passes_filter(row, min_span=args.min_span):
                continue
            n_filtered += 1
            key = (
                row['ImageID'],
                bbox_key(row['BoxXMin'], row['BoxYMin'], row['BoxXMax'], row['BoxYMax']),
            )
            if key in seen:
                n_already_done += 1
                continue
            new_candidates.append(row)

    print(f'  {n_total:,} total annotations')
    print(f'  {n_filtered:,} pass filter (min_span={args.min_span})')
    print(f'  {n_already_done:,} already in JSONL')
    print(f'  {len(new_candidates):,} NEW candidates to process')

    if not new_candidates:
        print('Nothing to do!')
        return

    # ---- Process and APPEND to existing file ----
    print(f'Processing {len(new_candidates):,} new candidates with {args.workers} workers...')

    written = 0
    failed = 0
    with open(args.output, 'a') as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_one, row, classes, args.masks_dir, args.images_dir)
                for row in new_candidates
            ]
            # Iterate in submission order — keeps the JSONL deterministic
            for i, future in enumerate(futures):
                result = future.result()
                if result:
                    out_f.write(json.dumps(result) + '\n')
                    written += 1
                else:
                    failed += 1
                if (i + 1) % 2000 == 0:
                    print(f'  [{i+1}/{len(new_candidates)}] written={written} failed={failed}')

    final_total = n_existing + written
    print(f'\nDone!')
    print(f'  Added {written:,} new examples ({failed:,} still failed)')
    print(f'  Total in {args.output}: {final_total:,}')


if __name__ == '__main__':
    main()
