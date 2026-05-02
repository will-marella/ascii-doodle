"""Count examples per category across Open Images and COCO.

For Open Images: tally `label` from the rendered JSONL (already filtered).
For COCO: walk instances_train2017.json, apply the same bbox-span filter
that rebuild_coco.py uses, and tally by category name. (No rendering — we
just want the upper bound on what would survive.)

Prints a merged table of (category, oi_count, coco_count, total) so we
can pick a category whitelist before committing to a full COCO render.

Usage:
    python count_categories.py
    python count_categories.py --categories person dog cat car bicycle
"""

import argparse
import json
from collections import Counter


def count_openimages(jsonl_path):
    counts = Counter()
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                counts[r['label'].lower()] += 1
            except Exception:
                continue
    return counts


def count_coco(instances_path, min_span=0.40):
    with open(instances_path) as f:
        data = json.load(f)
    images_by_id = {img['id']: img for img in data['images']}
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    counts = Counter()
    for ann in data['annotations']:
        img = images_by_id.get(ann['image_id'])
        if img is None:
            continue
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            continue
        img_w, img_h = img['width'], img['height']
        xmin, ymin = x / img_w, y / img_h
        xmax, ymax = (x + w) / img_w, (y + h) / img_h
        span = max(xmax - xmin, ymax - ymin)
        if span < min_span:
            continue
        edge_count = sum([xmin < 0.02, xmax > 0.98, ymin < 0.02, ymax > 0.98])
        if edge_count > 1:
            continue
        counts[cat_id_to_name[ann['category_id']].lower()] += 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oi-jsonl', default='openimages/train_ascii_128x64_relaxed.jsonl')
    p.add_argument('--coco-instances', default='coco/annotations/instances_train2017.json')
    p.add_argument('--categories', nargs='*', default=None,
                   help='Restrict the table to these category names. '
                        'Default: print top 40 by combined count.')
    p.add_argument('--top', type=int, default=40)
    args = p.parse_args()

    print(f'Counting Open Images from {args.oi_jsonl}...')
    oi = count_openimages(args.oi_jsonl)
    print(f'  {sum(oi.values()):,} total examples across {len(oi):,} labels')

    print(f'Counting COCO from {args.coco_instances} (post bbox filter)...')
    coco = count_coco(args.coco_instances)
    print(f'  {sum(coco.values()):,} total annotations across {len(coco):,} categories')

    # Merge
    all_cats = set(oi.keys()) | set(coco.keys())
    rows = []
    for cat in all_cats:
        rows.append((cat, oi.get(cat, 0), coco.get(cat, 0)))

    if args.categories:
        wanted = {c.lower() for c in args.categories}
        rows = [r for r in rows if r[0] in wanted]
        # Also report any requested categories that didn't match anything
        found = {r[0] for r in rows}
        missing = wanted - found
        if missing:
            print(f'\n  WARNING: no matches for: {sorted(missing)}')

    rows.sort(key=lambda r: r[1] + r[2], reverse=True)
    if not args.categories:
        rows = rows[:args.top]

    print()
    print(f'  {"category":<25} {"open_images":>12} {"coco":>10} {"total":>10}')
    print(f'  {"-"*25} {"-"*12} {"-"*10} {"-"*10}')
    total_oi = total_coco = 0
    for cat, n_oi, n_coco in rows:
        print(f'  {cat:<25} {n_oi:>12,} {n_coco:>10,} {n_oi + n_coco:>10,}')
        total_oi += n_oi
        total_coco += n_coco
    print(f'  {"-"*25} {"-"*12} {"-"*10} {"-"*10}')
    print(f'  {"TOTAL (shown)":<25} {total_oi:>12,} {total_coco:>10,} {total_oi + total_coco:>10,}')


if __name__ == '__main__':
    main()
