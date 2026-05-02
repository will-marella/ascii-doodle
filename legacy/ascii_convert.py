"""
Convert images to ASCII art using ascii-magic.

Usage:
    python ascii_convert.py <image_path> [--columns N]
    python ascii_convert.py --batch <dir> [--columns N] [--limit N]
"""

import argparse
import glob
import os
from ascii_magic import AsciiArt


def convert(image_path: str, columns: int = 64) -> str:
    art = AsciiArt.from_image(image_path)
    return art.to_ascii(columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Convert images to ASCII art")
    parser.add_argument("image", nargs="?", help="Image path")
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--batch", type=str, help="Directory of images to convert")
    parser.add_argument("--limit", type=int, default=5, help="Max images in batch mode")
    args = parser.parse_args()

    if args.batch:
        files = sorted(glob.glob(os.path.join(args.batch, "*.jpg")))[:args.limit]
        for f in files:
            print(f"\n{'='*args.columns}")
            print(f" {os.path.basename(f)}")
            print(f"{'='*args.columns}")
            print(convert(f, args.columns))
    elif args.image:
        print(convert(args.image, args.columns))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
