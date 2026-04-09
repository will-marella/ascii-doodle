"""Train the ASCII transformer."""

import argparse
import json
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
    build_class_map,
    load_jsonl_to_tensors,
)
from model import AsciiTransformer
from sample import decode, generate


# Curated sample classes for qualitative eval — chosen for visual diversity and
# sample-count coverage. Each has >=1k training examples in the post-cc dataset.
# Any name not in the loaded class_map is silently skipped, so this list is
# robust to dataset regeneration.
SAMPLE_CLASS_NAMES = [
    'Girl',                 # 33,065 — human figure
    'Car',                  # 15,527 — angular vehicle
    'Flower',               #  5,515 — radial organic
    'Dog',                  #  5,070 — quadruped
    'Bird',                 #  3,450 — winged animal
    'Guitar',               #  1,951 — elongated instrument
    'Fixed-wing aircraft',  #  1,835 — elongated vehicle
    'Cake',                 #  1,089 — compact food
]

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
    for tokens, class_ids in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)
        logits = model(tokens, class_ids)
        loss = compute_loss(logits, tokens, loss_weight)
        total += loss.item() * tokens.size(0)
        count += tokens.size(0)
    model.train()
    return total / count


def render_samples(model, class_ids, class_names, temperature=0.8, top_k=3):
    sampled = generate(model, class_ids, temperature=temperature, top_k=top_k)
    for i, name in enumerate(class_names):
        print(f'  --- {name} ---')
        for line in decode(sampled[i]).split('\n'):
            print(f'  {line}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train-path', default='openimages/train_ascii_64x32_cc.jsonl')
    p.add_argument('--val-path', default='openimages/validation_ascii_64x32_cc.jsonl')
    p.add_argument('--class-map', default='class_map.json')
    p.add_argument('--checkpoint-dir', default='checkpoints')
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
    # logging / eval
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--n-samples', type=int, default=8)
    # misc
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- class map ----
    if os.path.exists(args.class_map):
        with open(args.class_map) as f:
            class_map = json.load(f)
        print(f'Loaded class map from {args.class_map}: {len(class_map)} classes')
    else:
        print('Building class map from train + val...')
        class_map = build_class_map(args.train_path, args.val_path)
        with open(args.class_map, 'w') as f:
            json.dump(class_map, f, indent=2)
        print(f'Saved {len(class_map)} classes to {args.class_map}')

    # ---- data ----
    print(f'Loading train data from {args.train_path}...')
    train_tokens, train_class_ids = load_jsonl_to_tensors(
        args.train_path, class_map, limit=args.limit_train
    )
    print(f'  {len(train_tokens):,} examples')

    print(f'Loading val data from {args.val_path}...')
    val_tokens, val_class_ids = load_jsonl_to_tensors(args.val_path, class_map)
    print(f'  {len(val_tokens):,} examples')

    train_loader = DataLoader(
        AsciiDataset(train_tokens, train_class_ids),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        AsciiDataset(val_tokens, val_class_ids),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- model ----
    model = AsciiTransformer(
        vocab_size=VOCAB_SIZE,
        n_classes=len(class_map),
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

    # Fixed eval class IDs for comparable samples across checkpoints.
    # Prefer the curated diverse list; fall back to first-N-by-ID if curated
    # names aren't in this class_map (e.g., after regenerating the dataset).
    curated_ids = [class_map[name] for name in SAMPLE_CLASS_NAMES if name in class_map]
    missing = [name for name in SAMPLE_CLASS_NAMES if name not in class_map]
    if missing:
        print(f'Warning: curated classes not in class_map: {missing}')
    if len(curated_ids) >= args.n_samples:
        picked_ids = curated_ids[: args.n_samples]
        picked_names = [name for name in SAMPLE_CLASS_NAMES if name in class_map][: args.n_samples]
    else:
        print(f'Falling back to first {args.n_samples} classes by ID')
        id_to_class = {v: k for k, v in class_map.items()}
        picked_ids = sorted(class_map.values())[: args.n_samples]
        picked_names = [id_to_class[i] for i in picked_ids]
    sample_class_ids = torch.tensor(picked_ids, dtype=torch.long, device=device)
    sample_class_names = picked_names
    print(f'Sample classes: {sample_class_names}')

    # ---- training loop ----
    model.train()
    step = start_step
    t_start = time.time()
    train_iter = iter(train_loader)
    print(f'Training from step {step} to {args.total_steps}')

    while step < args.total_steps:
        try:
            tokens, class_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            tokens, class_ids = next(train_iter)

        tokens = tokens.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        lr = lr_schedule(step, args.peak_lr, warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        with torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')
        ):
            logits = model(tokens, class_ids)
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
            render_samples(model, sample_class_ids, sample_class_names)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'class_map': class_map,
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
