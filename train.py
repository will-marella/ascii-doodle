"""Train the ASCII transformer (v3: MaskGIT, unconditional, humans, mirror-augmented).

Training objective: for each batch, sample a masking ratio r, randomly mask
that fraction of positions, and compute cross-entropy loss on the masked
positions only. The model learns to predict any hidden position from all
visible positions via bidirectional attention.

Sampling happens via iterative unmasking (see sample.generate) rather than
left-to-right autoregression, which sidesteps the compounding-error problem
observed in v1/v2.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    AsciiDataset,
    BG_TOKEN,
    MASK_TOKEN,
    VOCAB_SIZE,
    load_jsonl_to_tensors,
)
from model import AsciiTransformer
from sample import decode, generate


# v3 data filter: four coherent "single human figure" classes from Open Images.
FILTER_LABELS = frozenset({'Girl', 'Woman', 'Boy', 'Man'})

# Mask-ratio sampling range. Avoids degenerate 0% and 100% masking.
MASK_RATIO_MIN = 0.15
MASK_RATIO_MAX = 1.00


# Optional callback invoked after every checkpoint save. Used by Modal to flush
# the persistent volume so checkpoints survive crashes.
POST_CHECKPOINT_HOOK = None


def lr_schedule(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def get_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = 'emb' in name
        is_norm = 'ln' in name or 'norm' in name
        if p.dim() < 2 or is_embedding or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def apply_random_masking(tokens: torch.Tensor) -> tuple:
    """Replace a random fraction of positions with MASK_TOKEN.

    Returns:
        masked: [B, T] with some positions set to MASK_TOKEN
        mask:   [B, T] bool, True where a position was masked (target for loss)
    """
    B, T = tokens.shape
    device = tokens.device

    # One mask ratio per sequence
    ratios = torch.rand(B, 1, device=device)
    ratios = ratios * (MASK_RATIO_MAX - MASK_RATIO_MIN) + MASK_RATIO_MIN

    # Independent per-position mask
    u = torch.rand(B, T, device=device)
    mask = u < ratios

    # Guarantee at least one masked position per sequence
    no_mask_rows = ~mask.any(dim=-1)
    if no_mask_rows.any():
        idx = torch.randint(0, T, (no_mask_rows.sum(),), device=device)
        mask[no_mask_rows, idx] = True

    masked_tokens = torch.where(mask, torch.full_like(tokens, MASK_TOKEN), tokens)
    return masked_tokens, mask


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy over masked positions only.

    If bg_weight != 1.0, positions whose target is BG_TOKEN get down/up-weighted.
    """
    B, T, V = logits.shape
    per_pos = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        reduction='none',
    ).view(B, T)

    weight = mask.float()
    if bg_weight != 1.0:
        bg = (targets == BG_TOKEN)
        weight = weight * torch.where(bg, torch.full_like(weight, bg_weight), torch.ones_like(weight))

    total = (per_pos * weight).sum()
    denom = weight.sum().clamp(min=1.0)
    return total / denom


@torch.no_grad()
def evaluate(model, val_loader, device, bg_weight: float) -> float:
    """Compute average masked CE loss over the validation set.

    Uses a fixed mask ratio of 0.5 on every sample for a stable comparable metric.
    """
    model.eval()
    total = 0.0
    count = 0
    for tokens in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        B, T = tokens.shape

        # Fixed 50% random mask for eval comparability
        u = torch.rand(B, T, device=device)
        mask = u < 0.5
        no_mask_rows = ~mask.any(dim=-1)
        if no_mask_rows.any():
            idx = torch.randint(0, T, (no_mask_rows.sum(),), device=device)
            mask[no_mask_rows, idx] = True
        masked = torch.where(mask, torch.full_like(tokens, MASK_TOKEN), tokens)

        logits = model(masked)
        loss = masked_ce_loss(logits, tokens, mask, bg_weight=bg_weight)
        total += loss.item() * B
        count += B
    model.train()
    return total / count


def render_samples(model, n_samples, n_steps=12, temperature=1.0):
    sampled = generate(model, n_samples=n_samples, n_steps=n_steps, temperature=temperature)
    for i in range(n_samples):
        print(f'  --- sample {i+1}/{n_samples} ---')
        for line in decode(sampled[i]).split('\n'):
            print(f'  {line}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train-path', default='openimages/train_ascii_64x32_cc.jsonl')
    p.add_argument('--val-path', default='openimages/validation_ascii_64x32_cc.jsonl')
    p.add_argument('--checkpoint-dir', default='checkpoints/v3_maskgit')
    p.add_argument('--resume', default=None)
    # model
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--n-layers', type=int, default=8)
    p.add_argument('--n-heads', type=int, default=6)
    # training
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--peak-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--warmup-frac', type=float, default=0.02)
    p.add_argument('--total-steps', type=int, default=20000)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--bg-loss-weight', type=float, default=1.0,
                   help='Down-weight for BG-target positions. Default 1.0 (no weighting).')
    p.add_argument('--no-augment', action='store_true')
    # sampling (used at eval time)
    p.add_argument('--sample-steps', type=int, default=12)
    p.add_argument('--sample-temperature', type=float, default=1.0)
    # logging / eval
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--n-samples', type=int, default=4)
    # misc
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f'Filter labels: {sorted(FILTER_LABELS)}')
    print(f'Augmentation: {"mirror flip (p=0.5)" if not args.no_augment else "none"}')
    print(f'Mode: MaskGIT (bidirectional, iterative unmask sampling)')

    # ---- data ----
    print(f'Loading train data from {args.train_path}...')
    train_tokens = load_jsonl_to_tensors(
        args.train_path, filter_labels=FILTER_LABELS, limit=args.limit_train,
    )
    print(f'  {len(train_tokens):,} training examples')

    print(f'Loading val data from {args.val_path}...')
    val_tokens = load_jsonl_to_tensors(args.val_path, filter_labels=FILTER_LABELS)
    print(f'  {len(val_tokens):,} validation examples')

    train_ds = AsciiDataset(train_tokens, augment=not args.no_augment)
    val_ds = AsciiDataset(val_tokens, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- model ----
    model = AsciiTransformer(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    print(f'Model: {model.num_params():,} params (vocab_size={VOCAB_SIZE} incl. MASK)')

    optimizer = torch.optim.AdamW(
        get_param_groups(model, args.weight_decay),
        lr=args.peak_lr, betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(args.total_steps * args.warmup_frac))

    # ---- resume ----
    start_step = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']

    # ---- training loop ----
    model.train()
    step = start_step
    t_start = time.time()
    train_iter = iter(train_loader)
    print(f'Training from step {step} to {args.total_steps}')

    while step < args.total_steps:
        try:
            tokens = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            tokens = next(train_iter)

        tokens = tokens.to(device, non_blocking=True)

        lr = lr_schedule(step, args.peak_lr, warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        masked_tokens, mask = apply_random_masking(tokens)

        with torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')
        ):
            logits = model(masked_tokens)
            loss = masked_ce_loss(logits, tokens, mask, bg_weight=args.bg_loss_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            its = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f'step {step:>6}  loss {loss.item():.4f}  lr {lr:.2e}  '
                f'gn {grad_norm:.2f}  {its:.2f} it/s'
            )

        if step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, bg_weight=args.bg_loss_weight)
            print(f'  [eval] val_loss {val_loss:.4f}')

        if step > 0 and step % args.sample_every == 0:
            print(f'  [samples @ step {step}]')
            render_samples(
                model, args.n_samples,
                n_steps=args.sample_steps,
                temperature=args.sample_temperature,
            )

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
                'version': 'v3_maskgit',
            }, ckpt_path)
            print(f'  [ckpt] {ckpt_path}')
            existing = sorted(
                [f for f in os.listdir(args.checkpoint_dir)
                 if f.startswith('step_') and f.endswith('.pt')],
                key=lambda x: int(x.split('_')[1].split('.')[0]),
            )
            for old in existing[:-3]:
                os.remove(os.path.join(args.checkpoint_dir, old))
            if POST_CHECKPOINT_HOOK is not None:
                POST_CHECKPOINT_HOOK()

        step += 1

    print('Training complete.')


if __name__ == '__main__':
    main()
