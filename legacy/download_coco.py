"""Download and extract COCO 2017 train + val images and annotations.

After running this, the layout under --output-dir is:

    coco/
      train2017/                        ~118k JPEGs (~18 GB)
      val2017/                          ~5k JPEGs (~1 GB)
      annotations/
        instances_train2017.json        instance segmentations + bboxes
        instances_val2017.json
        captions_train2017.json         5 captions per image
        captions_val2017.json

Usage:
    python download_coco.py --output-dir coco
    python download_coco.py --output-dir coco --skip-train-images   # annotations only
"""

import argparse
import os
import sys
import urllib.request
import zipfile


COCO_URLS = {
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
}


def download_with_progress(url, dest):
    if os.path.exists(dest):
        print(f'  exists: {dest}  (skipping download)')
        return
    print(f'  downloading {url}')
    print(f'  -> {dest}')

    last_print = [0.0]
    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        # Throttle prints to once per percent
        if pct - last_print[0] >= 0.5 or pct >= 100.0:
            last_print[0] = pct
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            sys.stdout.write(f'\r    {pct:5.1f}%  ({mb:6.0f} / {total_mb:6.0f} MB)')
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f'  missing: {zip_path}  (cannot extract)')
        return
    print(f'  extracting {os.path.basename(zip_path)} -> {extract_to}/')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_to)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='coco')
    parser.add_argument('--skip-annotations', action='store_true')
    parser.add_argument('--skip-train-images', action='store_true')
    parser.add_argument('--skip-val-images', action='store_true')
    parser.add_argument('--keep-zips', action='store_true',
                        help='Do not delete zip files after extracting (default: delete)')
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output dir: {args.output_dir}')

    # Annotations (~250 MB)
    if not args.skip_annotations:
        print('\n--- Annotations (~250 MB) ---')
        ann_zip = os.path.join(args.output_dir, 'annotations.zip')
        download_with_progress(COCO_URLS['annotations'], ann_zip)
        extract_zip(ann_zip, args.output_dir)
        if not args.keep_zips and os.path.exists(ann_zip):
            os.remove(ann_zip)

    # Train images (~18 GB) — the slow one
    if not args.skip_train_images:
        print('\n--- Train images (~18 GB, this is the slow one) ---')
        train_zip = os.path.join(args.output_dir, 'train2017.zip')
        download_with_progress(COCO_URLS['train_images'], train_zip)
        extract_zip(train_zip, args.output_dir)
        if not args.keep_zips and os.path.exists(train_zip):
            os.remove(train_zip)

    # Val images (~1 GB) — fast, optional
    if not args.skip_val_images:
        print('\n--- Val images (~1 GB) ---')
        val_zip = os.path.join(args.output_dir, 'val2017.zip')
        download_with_progress(COCO_URLS['val_images'], val_zip)
        extract_zip(val_zip, args.output_dir)
        if not args.keep_zips and os.path.exists(val_zip):
            os.remove(val_zip)

    # Sanity check final layout
    print('\n--- Layout check ---')
    expected = [
        os.path.join(args.output_dir, 'annotations/instances_train2017.json'),
        os.path.join(args.output_dir, 'annotations/captions_train2017.json'),
        os.path.join(args.output_dir, 'train2017'),
    ]
    for p in expected:
        status = 'ok' if os.path.exists(p) else 'MISSING'
        print(f'  {status:7}  {p}')

    print('\nDone.')


if __name__ == '__main__':
    main()
