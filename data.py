"""
Load ASCII dataset from JSONL, pre-tokenize into in-RAM tensors.

Provides AsciiDataset which yields (tokens, class_id) pairs. The full dataset
is loaded once at startup — ~600MB for 293k examples fits comfortably in RAM,
which eliminates per-step I/O during training.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

RAMP = ' "roy48Q'
VOCAB_SIZE = len(RAMP)        # 8
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


def build_class_map(*paths: str) -> dict:
    """Scan the given JSONL files and build a deterministic label->id mapping."""
    labels = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                labels.add(json.loads(line)['label'])
    return {label: i for i, label in enumerate(sorted(labels))}


def load_jsonl_to_tensors(path: str, class_map: dict, limit: int = None):
    """Read JSONL into (tokens [N, SEQ_LEN] uint8, class_ids [N] int64)."""
    table = build_char_lookup()

    with open(path) as f:
        n_lines = sum(1 for _ in f)
    if limit is not None:
        n_lines = min(n_lines, limit)

    tokens = np.empty((n_lines, SEQ_LEN), dtype=np.uint8)
    class_ids = np.empty(n_lines, dtype=np.int64)

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            r = json.loads(line)
            tokens[i] = tokenize_ascii(r['ascii'], table)
            class_ids[i] = class_map[r['label']]
            if (i + 1) % 50000 == 0:
                print(f'  [{path}] tokenized {i+1}/{n_lines}')

    return torch.from_numpy(tokens), torch.from_numpy(class_ids)


class AsciiDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, class_ids: torch.Tensor):
        self.tokens = tokens
        self.class_ids = class_ids

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx].long(), self.class_ids[idx]
