"""Generate prompt-conditioned ASCII samples and save them as text + SVG.

The SVG output keeps the ASCII as actual text, so it stays crisp at any zoom
level and avoids screenshot artifacts in the README.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from inference import DEFAULT_DIT_CHECKPOINT, DEFAULT_VAE_CHECKPOINT, generate_ascii, load_pipeline


PROMPT_GROUPS = {
    'group1_direct': [
        'bicycle',
        'tree',
        'car',
        'donut',
    ],
    'group2_synonyms': [
        'bike',
        'oak',
        'automobile',
        'doughnut',
    ],
    'group3_near_ood': [
        'storm',
        'flower vase',
        'mushroom cloud',
        'dancer',
    ],
}


def slugify(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')
    return slug or 'sample'


def write_text_sample(path: Path, prompt: str, ascii_art: str) -> None:
    path.write_text(f'prompt: {prompt}\n\n{ascii_art}\n', encoding='utf-8')


def write_svg_sample(
    path: Path,
    prompt: str,
    ascii_art: str,
    background: str,
    title_color: str,
    text_color: str,
) -> None:
    lines = ascii_art.splitlines()
    max_cols = max(len(line) for line in lines) if lines else 0

    font_size = 18
    line_height = 22
    char_width = 10.8
    padding_x = 24
    padding_y = 32
    title_gap = 30

    width = int(padding_x * 2 + max_cols * char_width)
    height = int(padding_y * 2 + title_gap + max(1, len(lines)) * line_height)

    title = escape(prompt)
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_height
        tspans.append(
            f'<tspan x="{padding_x}" dy="{dy}">{escape(line or " ")}</tspan>'
        )
    tspans_str = ''.join(tspans)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" fill="{background}" />
  <text x="{padding_x}" y="{padding_y}" font-family="'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="20" font-weight="700" fill="{title_color}">{title}</text>
  <text x="{padding_x}" y="{padding_y + title_gap}" xml:space="preserve" font-family="'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="{font_size}" fill="{text_color}">{tspans_str}</text>
</svg>
'''
    path.write_text(svg, encoding='utf-8')


def flatten_groups(groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    flat = []
    for group, prompts in groups.items():
        for prompt in prompts:
            flat.append((group, prompt))
    return flat


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='showcase_exports')
    parser.add_argument('--checkpoint', default=DEFAULT_DIT_CHECKPOINT)
    parser.add_argument('--vae-checkpoint', default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument('--device', default=None)
    parser.add_argument('--guidance-scale', type=float, default=10.0)
    parser.add_argument('--sample-steps', type=int, default=50)
    parser.add_argument('--prompts', nargs='*', default=None,
                        help='Optional flat prompt list. If omitted, use built-in grouped prompts.')
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.prompts:
        samples = [('custom', prompt) for prompt in args.prompts]
    else:
        samples = flatten_groups(PROMPT_GROUPS)

    pipeline = load_pipeline(
        checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        device=args.device,
    )

    manifest = {
        'checkpoint': args.checkpoint,
        'vae_checkpoint': args.vae_checkpoint,
        'guidance_scale': args.guidance_scale,
        'sample_steps': args.sample_steps,
        'samples': [],
    }

    for index, (group, prompt) in enumerate(samples, start=1):
        stem = f'{index:02d}_{group}_{slugify(prompt)}'
        ascii_art = generate_ascii(
            pipeline,
            prompt,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
        )
        txt_path = outdir / f'{stem}.txt'
        svg_light_path = outdir / f'{stem}_light.svg'
        svg_dark_path = outdir / f'{stem}_dark.svg'
        write_text_sample(txt_path, prompt, ascii_art)
        write_svg_sample(
            svg_light_path,
            prompt,
            ascii_art,
            background='#f6f2e8',
            title_color='#3a2f1f',
            text_color='#111111',
        )
        write_svg_sample(
            svg_dark_path,
            prompt,
            ascii_art,
            background='#0d1117',
            title_color='#f0d9b5',
            text_color='#f5f5f5',
        )
        manifest['samples'].append({
            'index': index,
            'group': group,
            'prompt': prompt,
            'txt': txt_path.name,
            'svg_light': svg_light_path.name,
            'svg_dark': svg_dark_path.name,
        })
        print(f'[{index:02d}/{len(samples):02d}] saved {prompt!r}')

    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nSaved {len(samples)} samples to {outdir}/')


if __name__ == '__main__':
    main()
