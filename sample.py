"""MaskGIT sampling and decoding for the ASCII transformer (v3).

Generation is iterative: start from all-[MASK], run the model, keep the
highest-confidence predictions, re-mask the rest, repeat. After ~12 iterations
the sequence is fully committed. Each iteration is a single forward pass on
the full 2048-token sequence with bidirectional attention.
"""

import math

import torch
import torch.nn.functional as F

from data import BG_TOKEN, GRID_W, MASK_TOKEN, RAMP, SEQ_LEN


def decode(token_ids) -> str:
    """Turn a sequence of SEQ_LEN token IDs into a multi-line ASCII image.

    Any residual MASK tokens are rendered as '?' for visibility.
    """
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    chars = []
    for t in token_ids:
        if t == MASK_TOKEN:
            chars.append('?')
        else:
            chars.append(RAMP[t])
    flat = ''.join(chars)
    return '\n'.join(flat[i:i + GRID_W] for i in range(0, SEQ_LEN, GRID_W))


@torch.no_grad()
def generate(
    model,
    n_samples: int = None,
    initial_tokens: torch.Tensor = None,
    n_steps: int = 12,
    temperature: float = 1.0,
    noise_temperature: float = 4.5,
) -> torch.Tensor:
    """
    MaskGIT iterative unmasking.

    Two modes:
      - Unconditional: pass `n_samples`. Starts from [B, SEQ_LEN] of MASK_TOKEN.
      - Partial-primed: pass `initial_tokens` of shape [B, SEQ_LEN] where
        positions with MASK_TOKEN will be filled and all other positions stay
        fixed. This supports arbitrary conditioning (not just prefixes).

    Args:
        n_steps: number of iterative unmasking passes. Cosine schedule.
        temperature: sampling temperature for per-position multinomial draws.
        noise_temperature: scale of Gumbel noise added to confidence scores
            during early iterations. Decays linearly to 0 over n_steps.

    Returns: [B, SEQ_LEN] long tensor with all positions unmasked.
    """
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    if initial_tokens is not None:
        tokens = initial_tokens.clone().to(device)
        B = tokens.size(0)
    else:
        assert n_samples is not None, 'provide either n_samples or initial_tokens'
        B = n_samples
        tokens = torch.full((B, SEQ_LEN), MASK_TOKEN, dtype=torch.long, device=device)

    for step in range(n_steps):
        is_masked = (tokens == MASK_TOKEN)
        if not is_masked.any():
            break

        # Forward pass on the full current state
        logits = model(tokens)                           # [B, T, V]

        # Never predict MASK itself as an output token
        logits[..., MASK_TOKEN] = -float('inf')

        # Sample a candidate token for every position
        probs = F.softmax(logits / temperature, dim=-1)   # [B, T, V]
        flat_probs = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).view(B, -1)  # [B, T]

        # Confidence of each sampled candidate
        sampled_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [B, T]
        confidence = torch.log(sampled_probs.clamp(min=1e-10))               # [B, T]

        # Already-unmasked positions should be preserved: give them inf confidence
        confidence = torch.where(
            is_masked, confidence, torch.full_like(confidence, float('inf'))
        )

        # Add Gumbel noise decaying over iterations (encourages diversity early)
        frac_done = (step + 1) / n_steps
        noise_scale = noise_temperature * (1 - frac_done)
        u = torch.rand_like(confidence).clamp(min=1e-10, max=1 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        confidence = confidence + noise_scale * gumbel

        # Cosine schedule: how many positions should remain masked AFTER this step
        next_mask_ratio = math.cos(math.pi * (step + 1) / (2 * n_steps))
        n_masked_target = int(next_mask_ratio * SEQ_LEN)
        n_unmask_target = SEQ_LEN - n_masked_target
        n_unmask_target = max(1, min(SEQ_LEN, n_unmask_target))

        # Top-n_unmask_target positions (by confidence) stay unmasked this round
        sorted_conf, _ = torch.sort(confidence, dim=-1, descending=True)
        threshold = sorted_conf[:, n_unmask_target - 1:n_unmask_target]       # [B, 1]
        keep = confidence >= threshold                                        # [B, T]

        # For currently-masked positions that are "kept", adopt the sampled value.
        # Everything else (already unmasked, or kept-masked) stays as-is.
        positions_to_unmask = is_masked & keep
        tokens = torch.where(positions_to_unmask, sampled, tokens)

    if was_training:
        model.train()
    return tokens
