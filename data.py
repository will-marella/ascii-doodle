"""
Load ASCII dataset from JSONL, pre-tokenize into in-RAM tensors.

Provides AsciiDataset which yields tokens for the autoregressive model.
Loading supports optional label filtering (e.g., filter to just human
classes), and the dataset supports horizontal mirror augmentation on train.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

RAMP = ' "roy48Q'
N_ASCII_TOKENS = len(RAMP)    # 8 real characters
MASK_TOKEN = 8                # [MASK] sentinel for MaskGIT — input only, never a target
VOCAB_SIZE = N_ASCII_TOKENS + 1  # 9 (includes MASK_TOKEN)
GRID_H = 32
GRID_W = 64
SEQ_LEN = GRID_H * GRID_W     # 2048
BG_TOKEN = 7                  # 'Q' — canvas background after rebuild_fixed_canvas


def build_char_lookup() -> np.ndarray:
    """256-entry char->ID lookup. Non-ramp chars mapped to 255 sentinel."""
    table = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate(RAMP):
        table[ord(ch)] = i
    return table


def tokenize_ascii(ascii_str: str, table: np.ndarray) -> np.ndarray:
    """Flatten a 32x64 ASCII block (with newlines) into SEQ_LEN token IDs."""
    flat = ascii_str.replace('\n', '')
    arr = np.frombuffer(flat.encode('ascii'), dtype=np.uint8)
    tokens = table[arr]
    if len(tokens) != SEQ_LEN:
        raise ValueError(f'Expected {SEQ_LEN} tokens, got {len(tokens)}')
    if (tokens == 255).any():
        bad_pos = int(np.argmax(tokens == 255))
        raise ValueError(f'Char {flat[bad_pos]!r} at position {bad_pos} not in ramp')
    return tokens


def load_jsonl_to_tensors(
    path: str,
    filter_labels: set = None,
    limit: int = None,
) -> torch.Tensor:
    """Read JSONL into a [N, SEQ_LEN] uint8 tensor of tokenized ASCII.

    If filter_labels is provided, skip any example whose label is not in it.
    """
    table = build_char_lookup()

    tokens_list = []
    skipped = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if filter_labels is not None and r['label'] not in filter_labels:
                skipped += 1
                continue
            tokens_list.append(tokenize_ascii(r['ascii'], table))
            if limit is not None and len(tokens_list) >= limit:
                break
            if len(tokens_list) % 20000 == 0:
                print(f'  [{path}] tokenized {len(tokens_list)}')

    if filter_labels is not None:
        print(f'  [{path}] kept {len(tokens_list):,}, skipped {skipped:,} non-matching')

    if not tokens_list:
        tokens = np.empty((0, SEQ_LEN), dtype=np.uint8)
    else:
        tokens = np.stack(tokens_list)
    return torch.from_numpy(tokens)


class AsciiDataset(Dataset):
    """Holds pre-tokenized ASCII samples in RAM.

    If augment=True, each __getitem__ returns a horizontally-mirrored copy
    with probability 0.5. Train data typically uses augment=True;
    val data uses augment=False so val loss stays comparable across runs.
    """

    def __init__(self, tokens: torch.Tensor, augment: bool = False):
        self.tokens = tokens
        self.augment = augment

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx].long()
        if self.augment and torch.rand(1).item() < 0.5:
            tokens = tokens.view(GRID_H, GRID_W).flip(dims=[1]).reshape(-1)
        return tokens
