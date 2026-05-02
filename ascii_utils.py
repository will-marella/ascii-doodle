"""Helpers for rendering ASCII token grids."""

from data import GRID_W, MASK_TOKEN, RAMP


def decode(token_ids, grid_w: int = None, mask_char: str = '?') -> str:
    """Turn a sequence of token IDs into a multi-line ASCII image."""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    chars = []
    for token in token_ids:
        if token == MASK_TOKEN:
            chars.append(mask_char)
        else:
            chars.append(RAMP[token])

    flat = ''.join(chars)
    width = grid_w or GRID_W
    return '\n'.join(flat[i:i + width] for i in range(0, len(flat), width))
