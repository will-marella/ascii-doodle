"""Train a small sketch classifier on QuickDraw bitmaps, then rank drawings
by confidence to separate "canonical, well-drawn" from "ambiguous, sloppy."

Downloads 28x28 numpy bitmaps for a set of categories, trains a lightweight
CNN, then scores every drawing in a target category by P(correct_class).
Prints top-K and bottom-K as ASCII so you can eyeball the quality gradient.

Usage:
    python quickdraw_classifier_probe.py --target dog
    python quickdraw_classifier_probe.py --target cat --epochs 5 --top-k 8
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

# ---- ASCII rendering (same as quickdraw_quality_probe.py) ----

RAMP = ' .:-+*#@'
QUICKDRAW_NDJSON_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'
QUICKDRAW_BITMAP_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy'
WIDTH_RATIO = 2.2
CACHE_DIR = 'quickdraw_cache'

# Categories to train the classifier on. More = harder task = more
# discriminating confidence scores. These are our target categories
# plus some confusers.
DEFAULT_CATEGORIES = [
    'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe',
    'fish', 'frog', 'rabbit', 'pig', 'monkey', 'bear', 'lion', 'tiger',
    'car', 'truck', 'bus', 'bicycle', 'motorbike', 'airplane', 'train',
    'flower', 'tree', 'house',
]


def download_bitmap(category):
    """Download the 28x28 numpy bitmap for a category. Returns (N, 784) uint8."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = category.replace(' ', '_')
    path = os.path.join(CACHE_DIR, f'{safe_name}.npy')
    if not os.path.exists(path):
        url = QUICKDRAW_BITMAP_URL.format(urllib.parse.quote(category))
        print(f'  Downloading {category}...')
        urllib.request.urlretrieve(url, path)
    arr = np.load(path)
    print(f'  {category}: {arr.shape[0]:,} drawings')
    return arr


class SketchCNN(nn.Module):
    """Small CNN for 28x28 grayscale sketch classification."""
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
        x = self.pool(F.relu(self.conv1(x)))   # 28->14
        x = self.pool(F.relu(self.conv2(x)))   # 14->7
        x = self.pool(F.relu(self.conv3(x)))   # 7->3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_classifier(categories, max_per_class=10000, epochs=3, batch_size=256, device='cpu'):
    """Train the sketch CNN. Returns (model, category_list)."""
    print(f'\nDownloading bitmaps for {len(categories)} categories...')
    all_images = []
    all_labels = []
    for i, cat in enumerate(categories):
        bitmaps = download_bitmap(cat)
        # Subsample for training speed
        n = min(len(bitmaps), max_per_class)
        idx = np.random.choice(len(bitmaps), n, replace=False)
        all_images.append(bitmaps[idx])
        all_labels.append(np.full(n, i, dtype=np.int64))

    X = np.concatenate(all_images).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = np.concatenate(all_labels)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split 90/10
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SketchCNN(len(categories)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f'\nTraining on {len(X_train):,} examples, validating on {len(X_val):,}...')
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

        # Validate
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
        acc = correct / len(X_val)
        print(f'  Epoch {epoch+1}/{epochs}: loss={total_loss/len(X_train):.4f} val_acc={acc:.3f}')

    return model


def score_category(model, category, cat_index, device='cpu', batch_size=512):
    """Score all drawings in a category by P(correct_class). Returns array of scores."""
    bitmaps = download_bitmap(category)
    X = torch.from_numpy(bitmaps.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0)

    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            scores = probs[:, cat_index].cpu().numpy()
            all_scores.append(scores)
    return np.concatenate(all_scores)


def render_bitmap_to_ascii(bitmap_row, canvas_w, canvas_h):
    """Render a 784-element uint8 vector to ASCII via the standard pipeline."""
    img = Image.fromarray(bitmap_row.reshape(28, 28))

    arr = np.array(img)
    fg = arr > 20  # QuickDraw bitmaps: 0=bg, >0=ink (inverted from our convention)
    if not fg.any():
        return None
    rows_mask = np.any(fg, axis=1)
    cols_mask = np.any(fg, axis=0)
    rmin, rmax = np.where(rows_mask)[0][[0, -1]]
    cmin, cmax = np.where(cols_mask)[0][[0, -1]]
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
    canvas = Image.new('L', (canvas_w, canvas_h), 0)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    canvas_arr = np.array(canvas)

    # QuickDraw bitmaps: 0=background, 255=ink. Our ramp: space=light, @=dark.
    # So ink (high value) should map to dense chars.
    n = len(RAMP) - 1
    lines = []
    for row in canvas_arr:
        lines.append(''.join(RAMP[int((v / 255.0) * n)] for v in row))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='dog', help='Category to rank')
    p.add_argument('--categories', nargs='*', default=None,
                   help='Override classifier categories (default: 26 mixed)')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--max-per-class', type=int, default=10000,
                   help='Max training examples per class')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--canvas-size', default='64x32')
    args = p.parse_args()

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    categories = args.categories or DEFAULT_CATEGORIES
    if args.target not in categories:
        categories = [args.target] + categories

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')

    model = train_classifier(categories, args.max_per_class, args.epochs, device=device)

    target_idx = categories.index(args.target)
    print(f'\nScoring all "{args.target}" drawings...')
    scores = score_category(model, args.target, target_idx, device=device)
    bitmaps = np.load(os.path.join(CACHE_DIR, f'{args.target}.npy'))

    print(f'\nConfidence distribution for "{args.target}":')
    print(f'  min={scores.min():.3f}  p10={np.percentile(scores, 10):.3f}  '
          f'p25={np.percentile(scores, 25):.3f}  median={np.median(scores):.3f}  '
          f'p75={np.percentile(scores, 75):.3f}  p90={np.percentile(scores, 90):.3f}  '
          f'max={scores.max():.3f}')

    # Sort by score
    order = np.argsort(scores)[::-1]
    k = min(args.top_k, len(order))

    print(f'\n{"="*68}')
    print(f'  TOP {k} (highest P({args.target}) → most canonical)')
    print(f'{"="*68}')
    for rank in range(k):
        idx = order[rank]
        ascii_str = render_bitmap_to_ascii(bitmaps[idx], cw, ch)
        if ascii_str is None:
            continue
        print(f'\n--- #{rank+1} confidence={scores[idx]:.4f} ---')
        print(ascii_str)

    print(f'\n{"="*68}')
    print(f'  BOTTOM {k} (lowest P({args.target}) → least recognizable)')
    print(f'{"="*68}')
    for rank in range(k):
        idx = order[-(rank + 1)]
        ascii_str = render_bitmap_to_ascii(bitmaps[idx], cw, ch)
        if ascii_str is None:
            continue
        print(f'\n--- #{rank+1} confidence={scores[idx]:.4f} ---')
        print(ascii_str)

    # Summary
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(f'\n--- Threshold analysis ---')
    for t in thresholds:
        n_above = (scores >= t).sum()
        print(f'  P({args.target}) >= {t:.2f}: {n_above:>6,} / {len(scores):,} '
              f'({100*n_above/len(scores):.1f}%)')


if __name__ == '__main__':
    main()
