"""Quality probe for QuickDraw → ASCII rendering.

Streams the first N recognized drawings from a QuickDraw category,
rasterizes the strokes, converts to ASCII at the chosen canvas size,
and prints each one to stdout so you can eyeball whether the dataset
is worth committing to.

QuickDraw simplified NDJSON format (one drawing per line):
    {
      "key_id": "...",
      "word": "dog",
      "recognized": true,
      "drawing": [[[x1,x2,...], [y1,y2,...]], ...]   # strokes in 256x256 space
    }

Usage:
    python quickdraw_quality_probe.py --category dog
    python quickdraw_quality_probe.py --category cat --n 20 --canvas-size 32x16
    python quickdraw_quality_probe.py --category "smiley face" --line-width 8
"""

import argparse
import json
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

RAMP = ' .:-+*#@'
QUICKDRAW_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson'
WIDTH_RATIO = 2.2  # ASCII chars are ~2x taller than wide


def render_drawing(strokes, canvas_w, canvas_h, line_width):
    """Rasterize a QuickDraw drawing to ASCII.

    Strokes come in 256x256 space. We draw them onto a 256x256 canvas with
    the chosen line width, crop to the bbox of the drawing, fit to the
    target canvas preserving aspect (with the ASCII width ratio), then
    convert to characters.
    """
    render_size = 256
    img = Image.new('L', (render_size, render_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        if len(xs) == 1:
            r = max(1, line_width // 2)
            draw.ellipse((xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r), fill=0)
            continue
        points = list(zip(xs, ys))
        draw.line(points, fill=0, width=line_width)

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

    # Fit while preserving aspect ratio with character-cell width ratio
    fit_h_for_full_w = ch_px * target_w / (cw_px * WIDTH_RATIO)
    if fit_h_for_full_w <= target_h:
        new_w, new_h = target_w, max(1, int(round(fit_h_for_full_w)))
    else:
        new_w = max(1, int(round(target_h * cw_px * WIDTH_RATIO / ch_px)))
        new_h = target_h

    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('L', (canvas_w, canvas_h), 255)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    canvas_arr = np.array(canvas)

    # Inverted ramp: dark drawing pixels → dense chars, white bg → space
    n = len(RAMP) - 1
    lines = []
    for row in canvas_arr:
        lines.append(''.join(RAMP[int(((255 - v) / 255.0) * n)] for v in row))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', default='dog')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--canvas-size', default='64x32')
    p.add_argument('--line-width', type=int, default=6,
                   help='Stroke width in the 256x256 render space. Larger = '
                        'thicker lines that survive the downsample better.')
    p.add_argument('--all', action='store_true',
                   help='Include unrecognized drawings (default: recognized only)')
    args = p.parse_args()

    cw, ch = args.canvas_size.split('x')
    cw, ch = int(cw), int(ch)

    url = QUICKDRAW_URL.format(urllib.parse.quote(args.category))
    print(f'Streaming: {url}')
    print(f'Canvas: {cw}x{ch}   line-width: {args.line_width}   '
          f'recognized-only: {not args.all}')
    print()

    found = 0
    skipped = 0
    with urllib.request.urlopen(url) as resp:
        for line in resp:
            if found >= args.n:
                break
            try:
                d = json.loads(line)
                if not args.all and not d.get('recognized', False):
                    skipped += 1
                    continue
                ascii_str = render_drawing(d['drawing'], cw, ch, args.line_width)
                if ascii_str is None:
                    continue
                found += 1
                print(f'--- #{found}: "{args.category}" key_id={d.get("key_id")} '
                      f'strokes={len(d["drawing"])} ---')
                print(ascii_str)
                print()
            except Exception:
                continue

    print(f'Rendered {found} drawings (skipped {skipped} unrecognized).')


if __name__ == '__main__':
    main()
