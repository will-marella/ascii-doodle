"""Build a quality-filtered QuickDraw ASCII dataset.

Downloads QuickDraw data, trains a small sketch classifier to score
drawing quality, filters to the best drawings, renders them from
vector strokes to ASCII at the target canvas size, and writes a JSONL
file compatible with the existing training pipeline.

Usage:
    # Single category quick test
    python build_quickdraw_dataset.py \
        --categories dog \
        --output quickdraw/dog_64x32.jsonl \
        --canvas-size 64x32 \
        --threshold 0.5

    # Multi-category dataset
    python build_quickdraw_dataset.py \
        --categories dog cat car bird horse \
        --output quickdraw/train_64x32.jsonl \
        --canvas-size 64x32 \
        --threshold 0.5
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

RAMP = ' .:-+*#@'
QUICKDRAW_NDJSON_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'
QUICKDRAW_BITMAP_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy'
WIDTH_RATIO = 2.2
CACHE_DIR = 'quickdraw_cache'

# All categories used for classifier training (superset of target categories)
CLASSIFIER_CATEGORIES = [
    'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe',
    'fish', 'frog', 'rabbit', 'pig', 'monkey', 'bear', 'lion', 'tiger',
    'car', 'truck', 'bus', 'bicycle', 'motorbike', 'airplane', 'train',
    'flower', 'tree', 'house',
]


def download_bitmap(category):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = category.replace(' ', '_')
    path = os.path.join(CACHE_DIR, f'{safe_name}.npy')
    if not os.path.exists(path):
        url = QUICKDRAW_BITMAP_URL.format(urllib.parse.quote(category))
        print(f'    Downloading bitmap: {category}...')
        urllib.request.urlretrieve(url, path)
    return np.load(path)


def download_ndjson(category):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = category.replace(' ', '_')
    path = os.path.join(CACHE_DIR, f'{safe_name}.ndjson')
    if not os.path.exists(path):
        url = QUICKDRAW_NDJSON_URL.format(urllib.parse.quote(category))
        print(f'    Downloading NDJSON: {category}...')
        urllib.request.urlretrieve(url, path)
    return path


class SketchCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_classifier(categories, max_per_class=10000, epochs=5, batch_size=256, device='cpu'):
    print(f'\n--- Training sketch classifier on {len(categories)} categories ---')
    all_images, all_labels = [], []
    for i, cat in enumerate(categories):
        bitmaps = download_bitmap(cat)
        n = min(len(bitmaps), max_per_class)
        idx = np.random.choice(len(bitmaps), n, replace=False)
        all_images.append(bitmaps[idx])
        all_labels.append(np.full(n, i, dtype=np.int64))
        print(f'    {cat}: {len(bitmaps):,} total, using {n:,} for training')

    X = np.concatenate(all_images).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = np.concatenate(all_labels)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SketchCNN(len(categories)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f'  Training on {split:,} examples...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(-1) == yb).sum().item()
        acc = correct / (len(X) - split)
        print(f'  Epoch {epoch+1}/{epochs}: loss={total_loss/split:.4f} val_acc={acc:.3f}')

    return model


def score_bitmaps(model, bitmaps, cat_index, device='cpu', batch_size=512):
    X = torch.from_numpy(bitmaps.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            probs = F.softmax(model(xb), dim=-1)
            all_scores.append(probs[:, cat_index].cpu().numpy())
    return np.concatenate(all_scores)


def render_strokes_to_ascii(strokes, canvas_w, canvas_h, line_width=6):
    """Render QuickDraw strokes to ASCII at target canvas size."""
    render_size = 256
    img = Image.new('L', (render_size, render_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        if len(xs) == 1:
            r = max(1, line_width // 2)
            draw.ellipse((xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r), fill=0)
            continue
        draw.line(list(zip(xs, ys)), fill=0, width=line_width)

    arr = np.array(img)
    fg = arr < 128
    if not fg.any():
        return None
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    cw_px, ch_px = crop.size
    target_w = max(1, canvas_w - 4)
    target_h = max(1, canvas_h - 2)

    fit_h = ch_px * target_w / (cw_px * WIDTH_RATIO)
    if fit_h <= target_h:
        new_w, new_h = target_w, max(1, int(round(fit_h)))
    else:
        new_w = max(1, int(round(target_h * cw_px * WIDTH_RATIO / ch_px)))
        new_h = target_h

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    # White canvas, paste resized drawing centered
    canvas = Image.new('L', (canvas_w, canvas_h), 255)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    canvas_arr = np.array(canvas)

    # Dark pixels (low value) → dense chars, light (high) → space
    n = len(RAMP) - 1
    lines = []
    for row in canvas_arr:
        lines.append(''.join(RAMP[int(((255 - v) / 255.0) * n)] for v in row))
    return '\n'.join(lines)


def process_category(category, cat_index, model, threshold, canvas_w, canvas_h,
                     line_width, device):
    """Score and render one category. Returns list of JSONL records."""
    print(f'\n--- Processing "{category}" ---')

    # Score bitmaps
    bitmaps = download_bitmap(category)
    scores = score_bitmaps(model, bitmaps, cat_index, device=device)
    n_pass = (scores >= threshold).sum()
    print(f'  {len(bitmaps):,} total, {n_pass:,} pass threshold >= {threshold}')

    good_indices = set(np.where(scores >= threshold)[0])

    # Stream NDJSON and render the good ones
    ndjson_path = download_ndjson(category)
    records = []
    line_idx = -1
    with open(ndjson_path) as f:
        for line in f:
            line_idx += 1
            if line_idx not in good_indices:
                continue
            try:
                d = json.loads(line)
                ascii_str = render_strokes_to_ascii(
                    d['drawing'], canvas_w, canvas_h, line_width,
                )
                if ascii_str is None:
                    continue
                records.append({
                    'label': category,
                    'ascii': ascii_str,
                    'key_id': d.get('key_id', ''),
                    'confidence': float(scores[line_idx]),
                })
            except Exception:
                continue

            if len(records) % 5000 == 0:
                print(f'    rendered {len(records):,}...')

    print(f'  {len(records):,} rendered successfully')
    return records


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--categories', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--canvas-size', default='64x32')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--line-width', type=int, default=6)
    p.add_argument('--classifier-epochs', type=int, default=5)
    p.add_argument('--max-per-class', type=int, default=10000,
                   help='Max training examples per class for classifier')
    args = p.parse_args(argv)

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Canvas: {cw}x{ch}')
    print(f'Threshold: {args.threshold}')
    print(f'Target categories: {args.categories}')

    # Ensure all target categories are in the classifier's training set
    all_cats = list(CLASSIFIER_CATEGORIES)
    for cat in args.categories:
        if cat not in all_cats:
            all_cats.append(cat)

    model = train_classifier(all_cats, args.max_per_class,
                             args.classifier_epochs, device=device)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    total_written = 0
    with open(args.output, 'w') as out_f:
        for category in args.categories:
            cat_index = all_cats.index(category)
            records = process_category(
                category, cat_index, model, args.threshold,
                cw, ch, args.line_width, device,
            )
            for r in records:
                out_f.write(json.dumps(r) + '\n')
            total_written += len(records)

    print(f'\n=== Done: {total_written:,} examples written to {args.output} ===')


if __name__ == '__main__':
    main()
