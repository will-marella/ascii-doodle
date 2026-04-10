"""Train the ASCII transformer (v2: unconditional, humans, mirror-augmented)."""

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
    VOCAB_SIZE,
    load_jsonl_to_tensors,
)
from model import AsciiTransformer
from sample import decode, generate


# v2 data filter: four coherent "single human figure" classes from Open Images.
# Excludes Person (often multi-figure) and Human body (scale inconsistency).
FILTER_LABELS = frozenset({'Girl', 'Woman', 'Boy', 'Man'})


# Optional callback invoked after every checkpoint save. Used by Modal to flush
# the persistent volume so checkpoints survive crashes. None in local runs.
POST_CHECKPOINT_HOOK = None


def lr_schedule(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.1,
) -> float:
    """Linear warmup -> cosine decay down to min_ratio * peak_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def get_param_groups(model: torch.nn.Module, weight_decay: float):
    """Weight decay only on 2D Linear weights — skip biases, LN, and embeddings."""
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


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Shifted cross-entropy. logits[:, :-1] predicts tokens[:, 1:]."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    V = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        weight=loss_weight,
    )


@torch.no_grad()
def evaluate(model, val_loader, loss_weight, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for tokens in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        logits = model(tokens)
        loss = compute_loss(logits, tokens, loss_weight)
        total += loss.item() * tokens.size(0)
        count += tokens.size(0)
    model.train()
    return total / count


def render_samples(model, n_samples, temperature=0.8, top_k=3):
    sampled = generate(model, n_samples, temperature=temperature, top_k=top_k)
    for i in range(n_samples):
        print(f'  --- sample {i+1}/{n_samples} ---')
        for line in decode(sampled[i]).split('\n'):
            print(f'  {line}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train-path', default='openimages/train_ascii_64x32_cc.jsonl')
    p.add_argument('--val-path', default='openimages/validation_ascii_64x32_cc.jsonl')
    p.add_argument('--checkpoint-dir', default='checkpoints/v2_humans')
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
    p.add_argument('--bg-loss-weight', type=float, default=0.15)
    p.add_argument('--no-augment', action='store_true', help='disable mirror augmentation')
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
    print(f'Model: {model.num_params():,} params')

    loss_weight = torch.ones(VOCAB_SIZE, device=device)
    loss_weight[BG_TOKEN] = args.bg_loss_weight

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

        with torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')
        ):
            logits = model(tokens)
            loss = compute_loss(logits, tokens, loss_weight)

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
            val_loss = evaluate(model, val_loader, loss_weight, device)
            print(f'  [eval] val_loss {val_loss:.4f}')

        if step > 0 and step % args.sample_every == 0:
            print(f'  [samples @ step {step}]')
            render_samples(model, args.n_samples)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
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
