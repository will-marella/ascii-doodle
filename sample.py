"""Sampling and decoding for the ASCII transformer."""

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
    class_ids: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Autoregressive sampling with temperature + top-k.

    class_ids: [B] long tensor on the same device as model.
    Returns: [B, SEQ_LEN] long tensor of sampled token IDs.

    Position 0 is fixed to BG_TOKEN — the top-left of every training canvas is
    background, so we skip learning/predicting it. Positions 1..SEQ_LEN-1 are
    sampled autoregressively.

    Note: O(SEQ_LEN^2) compute without a KV cache. For SEQ_LEN=2048 and small
    batch sizes (8-16) this takes ~30-60s on an A100 — acceptable as a
    training-time callback. Add KV caching if it becomes a bottleneck.
    """
    was_training = model.training
    model.eval()

    device = class_ids.device
    B = class_ids.shape[0]
    tokens = torch.full((B, SEQ_LEN), BG_TOKEN, dtype=torch.long, device=device)

    for t in range(SEQ_LEN - 1):
        logits = model(tokens[:, :t + 1], class_ids)        # [B, t+1, V]
        next_logits = logits[:, -1, :] / temperature         # [B, V]

        if top_k is not None:
            v, _ = torch.topk(next_logits, top_k, dim=-1)
            next_logits = torch.where(
                next_logits < v[:, -1:],
                torch.full_like(next_logits, -float('inf')),
                next_logits,
            )

        probs = F.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
        tokens[:, t + 1] = next_tok

    if was_training:
        model.train()
    return tokens
