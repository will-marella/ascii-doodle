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

# Standard ASCII-art density ramp: light → dark.
# Background is space (index 0), so generated images are copy-pasteable
# into chat / email / docs without escape sequences. Foreground subjects
# render with denser characters (#, @).
RAMP = ' .:-+*#@'
N_ASCII_TOKENS = len(RAMP)    # 8 real characters
MASK_TOKEN = 8                # [MASK] sentinel for MaskGIT — input only, never a target
VOCAB_SIZE = N_ASCII_TOKENS + 1  # 9 (includes MASK_TOKEN)
GRID_H = 32
GRID_W = 64
SEQ_LEN = GRID_H * GRID_W     # 2048
BG_TOKEN = 0                  # ' ' (space) — canvas background


def build_char_lookup() -> np.ndarray:
    """256-entry char->ID lookup. Non-ramp chars mapped to 255 sentinel."""
    table = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate(RAMP):
        table[ord(ch)] = i
    return table


def tokenize_ascii(ascii_str: str, table: np.ndarray, expected_len: int = None) -> np.ndarray:
    """Flatten an ASCII block (with newlines) into token IDs.

    If expected_len is provided, validate length matches. Otherwise accept any.
    """
    flat = ascii_str.replace('\n', '')
    arr = np.frombuffer(flat.encode('ascii'), dtype=np.uint8)
    tokens = table[arr]
    if expected_len is not None and len(tokens) != expected_len:
        raise ValueError(f'Expected {expected_len} tokens, got {len(tokens)}')
    if (tokens == 255).any():
        bad_pos = int(np.argmax(tokens == 255))
        raise ValueError(f'Char {flat[bad_pos]!r} at position {bad_pos} not in ramp')
    return tokens


def load_jsonl_to_tensors(
    path: str,
    filter_labels: set = None,
    limit: int = None,
) -> torch.Tensor:
    """Read JSONL into a [N, seq_len] uint8 tensor of tokenized ASCII.

    Sequence length is auto-detected from the first example, so this
    works for any canvas resolution (32x64, 64x128, 128x256, etc.).
    If filter_labels is provided, skip any example whose label is not in it.
    """
    table = build_char_lookup()

    tokens_list = []
    detected_len = None
    skipped = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if filter_labels is not None and r['label'] not in filter_labels:
                skipped += 1
                continue
            toks = tokenize_ascii(r['ascii'], table, expected_len=detected_len)
            if detected_len is None:
                detected_len = len(toks)
            tokens_list.append(toks)
            if limit is not None and len(tokens_list) >= limit:
                break
            if len(tokens_list) % 20000 == 0:
                print(f'  [{path}] tokenized {len(tokens_list)}')

    if filter_labels is not None:
        print(f'  [{path}] kept {len(tokens_list):,}, skipped {skipped:,} non-matching')
    if detected_len is not None:
        print(f'  [{path}] seq_len={detected_len}')

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

    grid_h/grid_w are needed for the mirror flip reshape. If not specified,
    defaults to the module-level GRID_H/GRID_W (32x64).

    If clip_embeddings is provided, each item also returns the corresponding
    CLIP embedding (for conditional DiT training).
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        augment: bool = False,
        grid_h: int = None,
        grid_w: int = None,
        clip_embeddings: torch.Tensor = None,
    ):
        self.tokens = tokens
        self.augment = augment
        self.grid_h = grid_h or GRID_H
        self.grid_w = grid_w or GRID_W
        self.clip_embeddings = clip_embeddings
        if clip_embeddings is not None:
            assert len(clip_embeddings) == len(tokens), (
                f'clip_embeddings ({len(clip_embeddings)}) must match tokens ({len(tokens)})'
            )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx].long()
        if self.augment and torch.rand(1).item() < 0.5:
            tokens = tokens.view(self.grid_h, self.grid_w).flip(dims=[1]).reshape(-1)
        if self.clip_embeddings is not None:
            return tokens, self.clip_embeddings[idx]
        return tokens


def load_clip_embeddings(path: str) -> torch.Tensor:
    """Load a parallel CLIP embeddings .npy file aligned with a JSONL."""
    print(f'Loading CLIP embeddings from {path}...')
    arr = np.load(path)
    print(f'  shape={arr.shape}, dtype={arr.dtype}')
    return torch.from_numpy(arr.astype(np.float32))
