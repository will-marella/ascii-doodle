"""Filter QuickDraw drawings by CLIP quality score and render the best as ASCII.

Downloads N drawings from a category, applies heuristic pre-filters
(recognized, stroke count, bbox coverage), scores survivors with CLIP,
and prints the top-K alongside the bottom-K so you can see the quality
gradient.

Usage:
    python quickdraw_filter_probe.py --category dog --n 500 --top-k 10
    python quickdraw_filter_probe.py --category cat --n 1000 --top-k 15 --canvas-size 48x24
"""

import argparse
import json
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw

RAMP = ' .:-+*#@'
QUICKDRAW_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'
WIDTH_RATIO = 2.2


def rasterize(strokes, size=256, line_width=6):
    """Render strokes to a PIL grayscale image (white bg, black lines)."""
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        if len(xs) == 1:
            r = max(1, line_width // 2)
            draw.ellipse((xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r), fill=0)
            continue
        draw.line(list(zip(xs, ys)), fill=0, width=line_width)
    return img


def bbox_coverage(img):
    """Fraction of the canvas covered by the drawing's bounding box."""
    arr = np.array(img)
    fg = arr < 128
    if not fg.any():
        return 0.0
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
    return bbox_area / (img.size[0] * img.size[1])


def to_ascii(img, canvas_w, canvas_h):
    """Convert a grayscale PIL image to ASCII, cropping to bbox first."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', default='dog')
    p.add_argument('--n', type=int, default=500,
                   help='How many drawings to download and score')
    p.add_argument('--top-k', type=int, default=10,
                   help='Show this many best and worst')
    p.add_argument('--canvas-size', default='64x32')
    p.add_argument('--line-width', type=int, default=6)
    p.add_argument('--min-strokes', type=int, default=3)
    p.add_argument('--max-strokes', type=int, default=20)
    p.add_argument('--min-coverage', type=float, default=0.10)
    p.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    args = p.parse_args()

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    # --- Step 1: Download and pre-filter ---
    url = QUICKDRAW_URL.format(urllib.parse.quote(args.category))
    print(f'Streaming {args.n} drawings from: {url}')

    drawings = []  # list of (key_id, strokes, n_strokes, pil_image)
    skipped_recog = 0
    skipped_strokes = 0
    skipped_coverage = 0
    read = 0

    with urllib.request.urlopen(url) as resp:
        for line in resp:
            if len(drawings) >= args.n:
                break
            read += 1
            try:
                d = json.loads(line)
                if not d.get('recognized', False):
                    skipped_recog += 1
                    continue
                n_strokes = len(d['drawing'])
                if n_strokes < args.min_strokes or n_strokes > args.max_strokes:
                    skipped_strokes += 1
                    continue
                img = rasterize(d['drawing'], line_width=args.line_width)
                cov = bbox_coverage(img)
                if cov < args.min_coverage:
                    skipped_coverage += 1
                    continue
                drawings.append((d.get('key_id', '?'), d['drawing'], n_strokes, img))
            except Exception:
                continue

    print(f'  Read {read:,} lines, kept {len(drawings)} after pre-filters')
    print(f'  Skipped: {skipped_recog} unrecognized, {skipped_strokes} stroke count, '
          f'{skipped_coverage} coverage')

    if not drawings:
        print('No drawings passed filters.')
        return

    # --- Step 2: CLIP scoring ---
    print(f'\nLoading CLIP ({args.clip_model})...')
    from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # Text embedding for the category
    text_prompt = f"a drawing of a {args.category}"
    text_inputs = processor(text=[text_prompt], return_tensors='pt', padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Image embeddings in batches
    print(f'Scoring {len(drawings)} drawings against "{text_prompt}"...')
    batch_size = 64
    all_scores = []
    for i in range(0, len(drawings), batch_size):
        batch_imgs = [d[3].convert('RGB') for d in drawings[i:i + batch_size]]
        img_inputs = processor(images=batch_imgs, return_tensors='pt', padding=True)
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_emb = model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            scores = (img_emb @ text_emb.T).squeeze(-1).cpu().numpy()
        all_scores.extend(scores.tolist())

    # --- Step 3: Rank and display ---
    scored = list(zip(all_scores, drawings))
    scored.sort(key=lambda x: x[0], reverse=True)

    scores_arr = np.array(all_scores)
    print(f'\nCLIP score distribution:')
    print(f'  min={scores_arr.min():.3f}  p25={np.percentile(scores_arr, 25):.3f}  '
          f'median={np.median(scores_arr):.3f}  p75={np.percentile(scores_arr, 75):.3f}  '
          f'max={scores_arr.max():.3f}')

    k = min(args.top_k, len(scored))

    print(f'\n{"="*68}')
    print(f'  TOP {k} (highest CLIP score → most recognizable)')
    print(f'{"="*68}')
    for rank, (score, (key_id, strokes, n_s, img)) in enumerate(scored[:k], 1):
        ascii_str = to_ascii(img, cw, ch)
        if ascii_str is None:
            continue
        print(f'\n--- #{rank} score={score:.3f} strokes={n_s} key={key_id} ---')
        print(ascii_str)

    print(f'\n{"="*68}')
    print(f'  BOTTOM {k} (lowest CLIP score → least recognizable)')
    print(f'{"="*68}')
    for rank, (score, (key_id, strokes, n_s, img)) in enumerate(scored[-k:], 1):
        ascii_str = to_ascii(img, cw, ch)
        if ascii_str is None:
            continue
        print(f'\n--- #{rank} score={score:.3f} strokes={n_s} key={key_id} ---')
        print(ascii_str)

    # Suggested threshold
    p50 = np.median(scores_arr)
    above_median = sum(1 for s in all_scores if s >= p50)
    print(f'\n--- Summary ---')
    print(f'If you keep scores >= median ({p50:.3f}): {above_median}/{len(drawings)} drawings')
    print(f'If you keep scores >= p75 ({np.percentile(scores_arr, 75):.3f}): '
          f'{sum(1 for s in all_scores if s >= np.percentile(scores_arr, 75))}/{len(drawings)} drawings')


if __name__ == '__main__':
    main()
