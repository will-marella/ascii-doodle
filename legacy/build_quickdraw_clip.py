"""Build a QuickDraw dataset filtered by both classifier confidence AND
CLIP semantic similarity to a target prompt (e.g. "face of a dog").

Combines the two filtering strategies:
  1. Classifier confidence >= threshold (basic quality: "is this a dog?")
  2. CLIP similarity to a semantic prompt (pose/style: "is this a dog face?")

Keeps the top-K by CLIP score (after classifier pre-filter) and renders
from vector strokes to ASCII at the target canvas size.

Usage:
    python build_quickdraw_clip.py \\
        --category dog \\
        --clip-prompt "drawing of a dog face" \\
        --output quickdraw/dog_face_32x16.jsonl \\
        --canvas-size 32x16 \\
        --top-k 20000
"""

import argparse
import json
import os
import time
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

CLASSIFIER_CATEGORIES = [
    'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe',
    'fish', 'frog', 'rabbit', 'pig', 'monkey', 'bear', 'lion', 'tiger',
    'car', 'truck', 'bus', 'bicycle', 'motorbike', 'airplane', 'train',
    'flower', 'tree', 'house',
]


def download_file(url, path):
    if not os.path.exists(path):
        print(f'    Downloading {os.path.basename(path)}...')
        urllib.request.urlretrieve(url, path)


def download_bitmap(category):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{category.replace(" ", "_")}.npy')
    download_file(QUICKDRAW_BITMAP_URL.format(urllib.parse.quote(category)), path)
    return np.load(path)


def download_ndjson(category):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{category.replace(" ", "_")}.ndjson')
    download_file(QUICKDRAW_NDJSON_URL.format(urllib.parse.quote(category)), path)
    return path


# ---- Sketch classifier (same as build_quickdraw_dataset.py) ----

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
    print(f'\n--- Training sketch classifier ({len(categories)} categories) ---')
    all_images, all_labels = [], []
    for i, cat in enumerate(categories):
        bitmaps = download_bitmap(cat)
        n = min(len(bitmaps), max_per_class)
        idx = np.random.choice(len(bitmaps), n, replace=False)
        all_images.append(bitmaps[idx])
        all_labels.append(np.full(n, i, dtype=np.int64))

    X = np.concatenate(all_images).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = np.concatenate(all_labels)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.9 * len(X))

    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SketchCNN(len(categories)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'  Epoch {epoch+1}/{epochs} done')
    model.eval()
    return model


def score_bitmaps_classifier(model, bitmaps, cat_index, device='cpu', batch_size=512):
    X = torch.from_numpy(bitmaps.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            probs = F.softmax(model(xb), dim=-1)
            all_scores.append(probs[:, cat_index].cpu().numpy())
    return np.concatenate(all_scores)


# ---- Stroke rendering ----

def rasterize_strokes(strokes, size=256, line_width=6):
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        if len(xs) == 1:
            r = max(1, line_width // 2)
            draw.ellipse((xs[0]-r, ys[0]-r, xs[0]+r, ys[0]+r), fill=0)
            continue
        draw.line(list(zip(xs, ys)), fill=0, width=line_width)
    return img


def strokes_to_ascii(strokes, canvas_w, canvas_h, line_width=6):
    img = rasterize_strokes(strokes, line_width=line_width)
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
    target_w = max(1, canvas_w - 2)
    target_h = max(1, canvas_h - 1)

    fit_h = ch_px * target_w / (cw_px * WIDTH_RATIO)
    if fit_h <= target_h:
        new_w, new_h = target_w, max(1, int(round(fit_h)))
    else:
        new_w = max(1, int(round(target_h * cw_px * WIDTH_RATIO / ch_px)))
        new_h = target_h

    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('L', (canvas_w, canvas_h), 255)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    canvas_arr = np.array(canvas)

    n = len(RAMP) - 1
    lines = []
    for row in canvas_arr:
        lines.append(''.join(RAMP[int(((255 - v) / 255.0) * n)] for v in row))
    return '\n'.join(lines)


# ---- CLIP scoring ----

def score_with_clip(images_pil, text_prompt, clip_model, clip_processor, device,
                    batch_size=64):
    """Score a list of PIL images against a text prompt. Returns numpy array of scores."""
    # Text embedding (compute once)
    text_inputs = clip_processor(text=[text_prompt], return_tensors='pt', padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    all_scores = []
    for i in range(0, len(images_pil), batch_size):
        batch = [img.convert('RGB') for img in images_pil[i:i + batch_size]]
        img_inputs = clip_processor(images=batch, return_tensors='pt', padding=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            scores = (img_emb @ text_emb.T).squeeze(-1).cpu().numpy()
        all_scores.extend(scores.tolist())
        if (i + batch_size) % 1000 < batch_size:
            print(f'    CLIP scored {min(i + batch_size, len(images_pil)):,}/{len(images_pil):,}')
    return np.array(all_scores)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--category', required=True)
    p.add_argument('--clip-prompt', required=True,
                   help='CLIP text prompt for semantic filtering, e.g. "drawing of a dog face"')
    p.add_argument('--output', required=True)
    p.add_argument('--canvas-size', default='32x16')
    p.add_argument('--classifier-threshold', type=float, default=0.3,
                   help='Minimum classifier confidence for basic quality filter')
    p.add_argument('--top-k', type=int, default=20000,
                   help='Keep top-K drawings by CLIP score (after classifier filter)')
    p.add_argument('--line-width', type=int, default=6)
    p.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    p.add_argument('--classifier-epochs', type=int, default=5)
    args = p.parse_args(argv)

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Canvas: {cw}x{ch}')
    print(f'Category: {args.category}')
    print(f'CLIP prompt: "{args.clip_prompt}"')
    print(f'Classifier threshold: {args.classifier_threshold}')
    print(f'Top-K: {args.top_k}')

    # ---- Step 1: Classifier pre-filter ----
    all_cats = list(CLASSIFIER_CATEGORIES)
    if args.category not in all_cats:
        all_cats.append(args.category)
    cat_index = all_cats.index(args.category)

    classifier = train_classifier(all_cats, device=device, epochs=args.classifier_epochs)
    bitmaps = download_bitmap(args.category)
    print(f'\nScoring {len(bitmaps):,} drawings with classifier...')
    cls_scores = score_bitmaps_classifier(classifier, bitmaps, cat_index, device=device)

    good_mask = cls_scores >= args.classifier_threshold
    good_indices = np.where(good_mask)[0]
    print(f'  {len(good_indices):,} pass classifier threshold >= {args.classifier_threshold}')

    # Free classifier memory
    del classifier
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Step 2: Load strokes for survivors and render to images for CLIP ----
    print(f'\nLoading NDJSON and rendering {len(good_indices):,} drawings...')
    ndjson_path = download_ndjson(args.category)
    good_set = set(good_indices.tolist())

    # Read all lines for good indices
    drawings = {}  # idx -> (strokes, key_id)
    with open(ndjson_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx not in good_set:
                continue
            try:
                d = json.loads(line)
                drawings[line_idx] = (d['drawing'], d.get('key_id', ''))
            except Exception:
                continue
    print(f'  Loaded {len(drawings):,} drawings from NDJSON')

    # Render to 256x256 PIL images for CLIP scoring
    print('  Rasterizing for CLIP...')
    ordered_indices = sorted(drawings.keys())
    pil_images = []
    for idx in ordered_indices:
        strokes, _ = drawings[idx]
        img = rasterize_strokes(strokes, size=256, line_width=args.line_width)
        pil_images.append(img)

    # ---- Step 3: CLIP semantic scoring ----
    print(f'\nLoading CLIP ({args.clip_model})...')
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    print(f'Scoring {len(pil_images):,} drawings against "{args.clip_prompt}"...')
    clip_scores = score_with_clip(pil_images, args.clip_prompt, clip_model,
                                  clip_processor, device)

    print(f'\nCLIP score distribution:')
    print(f'  min={clip_scores.min():.3f}  p25={np.percentile(clip_scores, 25):.3f}  '
          f'median={np.median(clip_scores):.3f}  p75={np.percentile(clip_scores, 75):.3f}  '
          f'max={clip_scores.max():.3f}')

    # Keep top-K by CLIP score
    k = min(args.top_k, len(clip_scores))
    top_k_positions = np.argsort(clip_scores)[::-1][:k]
    clip_threshold = clip_scores[top_k_positions[-1]] if k < len(clip_scores) else 0.0
    print(f'  Keeping top {k:,} (CLIP score >= {clip_threshold:.3f})')

    # Free CLIP memory
    del clip_model, clip_processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Step 4: Render top-K to ASCII and write JSONL ----
    print(f'\nRendering top {k:,} drawings to {cw}x{ch} ASCII...')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    written = 0
    with open(args.output, 'w') as out_f:
        for pos in top_k_positions:
            idx = ordered_indices[pos]
            strokes, key_id = drawings[idx]
            ascii_str = strokes_to_ascii(strokes, cw, ch, args.line_width)
            if ascii_str is None:
                continue
            record = {
                'label': args.category,
                'ascii': ascii_str,
                'key_id': key_id,
                'classifier_score': float(cls_scores[idx]),
                'clip_score': float(clip_scores[pos]),
            }
            out_f.write(json.dumps(record) + '\n')
            written += 1

    print(f'\n=== Done: {written:,} examples written to {args.output} ===')

    # Show a few samples
    print(f'\n--- Top 3 samples (highest CLIP score) ---')
    with open(args.output) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            r = json.loads(line)
            print(f'\n#{i+1} cls={r["classifier_score"]:.3f} clip={r["clip_score"]:.3f}')
            print(r['ascii'])


if __name__ == '__main__':
    main()
