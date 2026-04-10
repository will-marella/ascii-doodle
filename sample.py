"""Sampling and decoding for the ASCII transformer (unconditional)."""

import torch
import torch.nn.functional as F

from data import BG_TOKEN, GRID_W, RAMP, SEQ_LEN


def decode(token_ids) -> str:
    """Turn a sequence of SEQ_LEN token IDs into a multi-line ASCII image."""
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    chars = ''.join(RAMP[i] for i in token_ids)
    return '\n'.join(chars[i:i + GRID_W] for i in range(0, SEQ_LEN, GRID_W))


@torch.no_grad()
def generate(
    model,
    n_samples: int = None,
    prefix: torch.Tensor = None,
    temperature: float = 0.8,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Autoregressive sampling with temperature + top-k, optionally primed with
    a fixed prefix.

    Two modes:
      - Unconditional: pass `n_samples`. Position 0 is fixed to BG_TOKEN
        (top-left of every canvas is background in the training data).
      - Prefix-primed: pass `prefix` of shape [B, prefix_len]. Those positions
        are locked to the given tokens and sampling proceeds from prefix_len.

    Returns: [B, SEQ_LEN] long tensor of sampled token IDs.

    Note: O(SEQ_LEN^2) compute without a KV cache.
    """
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    if prefix is not None:
        B, prefix_len = prefix.shape
        assert prefix_len >= 1 and prefix_len <= SEQ_LEN
        tokens = torch.full((B, SEQ_LEN), BG_TOKEN, dtype=torch.long, device=device)
        tokens[:, :prefix_len] = prefix.to(device)
        start_t = prefix_len - 1
    else:
        assert n_samples is not None, 'provide either n_samples or prefix'
        B = n_samples
        tokens = torch.full((B, SEQ_LEN), BG_TOKEN, dtype=torch.long, device=device)
        start_t = 0

    for t in range(start_t, SEQ_LEN - 1):
        logits = model(tokens[:, :t + 1])                # [B, t+1, V]
        next_logits = logits[:, -1, :] / temperature      # [B, V]

        if top_k is not None:
            v, _ = torch.topk(next_logits, top_k, dim=-1)
            next_logits = torch.where(
                next_logits < v[:, -1:],
                torch.full_like(next_logits, -float('inf')),
                next_logits,
            )

        probs = F.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
        tokens[:, t + 1] = next_tok

    if was_training:
        model.train()
    return tokens
