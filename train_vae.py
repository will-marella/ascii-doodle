"""Train the ASCII VAE.

Stage 1 of the v5 latent-diffusion stack. The VAE learns a compressed
continuous latent space for ASCII canvases. Downstream, a diffusion model
will operate in this latent space.

Loss: CE reconstruction + beta * KL (with per-channel free bits).
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
    GRID_H as DEFAULT_GRID_H,
    GRID_W as DEFAULT_GRID_W,
    N_ASCII_TOKENS,
    load_jsonl_to_tensors,
)
from ascii_utils import decode
from vae import AsciiVAE


# Override via --filter-labels.
FILTER_LABELS = None


# Optional callback invoked after every checkpoint save (Modal volume commit).
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
    """Weight decay only on conv/linear weights, not biases/norms/embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = 'emb' in name
        is_norm = 'norm' in name
        if p.dim() < 2 or is_embedding or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
):
    """Returns (total, recon_nats_per_image, kl_raw_nats_per_image, kl_used_in_loss).

    - Reconstruction: cross-entropy summed over spatial, mean over batch
    - KL: per-element analytic KL against N(0, I). Free bits applied per channel
      (each channel gets at least `free_bits` nats of KL budget before it starts
      contributing to the loss).
    """
    # Reconstruction: CE per position, sum over spatial, mean over batch
    ce = F.cross_entropy(logits, targets, reduction='none')   # [B, H, W]
    recon = ce.sum(dim=[1, 2]).mean()                          # scalar, nats/image

    # KL per element
    kl_per_elem = 0.5 * (mean.pow(2) + log_var.exp() - 1.0 - log_var)  # [B, C, H', W']

    # Raw KL for monitoring (no free bits, no clamping)
    kl_raw = kl_per_elem.sum(dim=[1, 2, 3]).mean()             # scalar, nats/image

    # Per-channel KL for loss with free bits: sum over spatial, mean over batch -> [C]
    kl_per_channel = kl_per_elem.sum(dim=[2, 3]).mean(dim=0)   # [C]
    if free_bits > 0:
        kl_per_channel = torch.clamp(kl_per_channel, min=free_bits)
    kl_loss = kl_per_channel.sum()                             # scalar

    total = recon + beta * kl_loss
    return total, recon, kl_raw, kl_loss


@torch.no_grad()
def evaluate(model, val_loader, device, beta, free_bits, grid_h, grid_w):
    model.eval()
    n_samples = 0
    total_recon = 0.0
    total_kl = 0.0
    total_correct = 0
    total_positions = 0
    for tokens in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        tokens_2d = tokens.view(-1, grid_h, grid_w)
        logits, mean, log_var = model(tokens_2d)
        _, recon, kl_raw, _ = vae_loss(logits, tokens_2d, mean, log_var, beta=beta, free_bits=free_bits)
        B = tokens_2d.size(0)
        total_recon += recon.item() * B
        total_kl += kl_raw.item() * B
        pred = logits.argmax(dim=1)                            # [B, H, W]
        total_correct += (pred == tokens_2d).sum().item()
        total_positions += tokens_2d.numel()
        n_samples += B
    model.train()
    return {
        'recon_per_image': total_recon / n_samples,
        'kl_raw_per_image': total_kl / n_samples,
        'accuracy': total_correct / total_positions,
    }


def render_reconstructions(model, val_tokens: torch.Tensor, n_samples: int = 4, grid_h: int = 32, grid_w: int = 64):
    """Encode + decode a few real samples, print original vs reconstruction."""
    device = next(model.parameters()).device
    samples = val_tokens[:n_samples].long().to(device)
    samples_2d = samples.view(-1, grid_h, grid_w)

    with torch.no_grad():
        logits, mean, log_var = model(samples_2d)
        pred = logits.argmax(dim=1)                            # [B, H, W]

    for i in range(n_samples):
        print(f'  --- sample {i+1}/{n_samples} ---')
        print('  [original]')
        orig_flat = samples_2d[i].reshape(-1).tolist()
        for line in decode(orig_flat, grid_w=grid_w).split('\n'):
            print(f'    {line}')
        print('  [reconstruction]')
        recon_flat = pred[i].reshape(-1).tolist()
        for line in decode(recon_flat, grid_w=grid_w).split('\n'):
            print(f'    {line}')


def main(argv=None):
    p = argparse.ArgumentParser()
    # paths
    p.add_argument('--train-path', default='openimages/train_ascii_64x32_cc.jsonl',
                   help='JSONL training data, or ignored if --train-npy is set.')
    p.add_argument('--val-path', default='openimages/validation_ascii_64x32_cc.jsonl',
                   help='JSONL val data, or ignored if --train-npy is set.')
    p.add_argument('--train-npy', default=None,
                   help='Directory with tokens.npy + labels.npy (from build_quickdraw_full.py). '
                        'Overrides --train-path/--val-path.')
    p.add_argument('--checkpoint-dir', default='checkpoints/vae_v1')
    p.add_argument('--resume', default=None)
    # grid (auto-detected from data if not specified)
    p.add_argument('--grid-h', type=int, default=None,
                   help='Grid height. Auto-detected from data if omitted.')
    p.add_argument('--grid-w', type=int, default=None,
                   help='Grid width. Auto-detected from data if omitted.')
    # model
    p.add_argument('--embed-dim', type=int, default=32)
    p.add_argument('--base-channels', type=int, default=64)
    p.add_argument('--latent-channels', type=int, default=16)
    p.add_argument('--downsample-stages', type=int, default=2,
                   help='2 = 4x spatial compression (default), 3 = 8x, etc.')
    # training
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--peak-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-frac', type=float, default=0.02)
    p.add_argument('--total-steps', type=int, default=10000)
    p.add_argument('--grad-clip', type=float, default=1.0)
    # VAE loss
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--free-bits', type=float, default=1.0,
                   help='Per-channel KL free bits (nats). 0 disables.')
    p.add_argument('--no-augment', action='store_true')
    # logging / eval
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--n-samples', type=int, default=4)
    # misc
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--filter-labels', default='',
                   help='Comma-separated list of labels to keep. Empty = all.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    filter_labels = None
    if args.filter_labels:
        filter_labels = frozenset(l.strip() for l in args.filter_labels.split(',') if l.strip())

    print(f'Mode: ASCII VAE training')
    print(f'Augmentation: {"mirror flip (p=0.5)" if not args.no_augment else "none"}')

    # ---- data ----
    if args.train_npy:
        import numpy as np
        from torch.utils.data import Dataset as _Dataset

        class _NpyDataset(_Dataset):
            def __init__(self, tokens_npy, augment=False):
                self.tokens = np.load(tokens_npy, mmap_mode='r')
                self.grid_h = self.tokens.shape[1]
                self.grid_w = self.tokens.shape[2]
                self.augment = augment
            def __len__(self):
                return len(self.tokens)
            def __getitem__(self, idx):
                t = torch.from_numpy(self.tokens[idx].copy()).long()
                if self.augment and torch.rand(1).item() < 0.5:
                    t = t.flip(dims=[1])
                return t.reshape(-1)

        tokens_path = os.path.join(args.train_npy, 'tokens.npy')
        print(f'Loading numpy dataset from {args.train_npy}/')
        train_ds = _NpyDataset(tokens_path, augment=not args.no_augment)
        grid_h, grid_w = train_ds.grid_h, train_ds.grid_w
        # Use a small subset for validation (not the full 50M)
        val_size = min(5000, len(train_ds))
        val_tokens_full = torch.from_numpy(train_ds.tokens[:val_size].copy()).long().reshape(val_size, -1)
        val_ds = AsciiDataset(val_tokens_full, augment=False, grid_h=grid_h, grid_w=grid_w)
        print(f'  {len(train_ds):,} training examples')
        print(f'  grid: {grid_h}x{grid_w}')
    else:
        print(f'Filter labels: {sorted(filter_labels) if filter_labels else "(all classes)"}')
        print(f'Loading train data from {args.train_path}...')
        train_tokens = load_jsonl_to_tensors(
            args.train_path, filter_labels=filter_labels, limit=args.limit_train,
        )
        print(f'  {len(train_tokens):,} training examples')

        print(f'Loading val data from {args.val_path}...')
        val_tokens_full = load_jsonl_to_tensors(args.val_path, filter_labels=filter_labels)
        print(f'  {len(val_tokens_full):,} validation examples')

        seq_len = train_tokens.shape[1]
        if args.grid_h and args.grid_w:
            grid_h, grid_w = args.grid_h, args.grid_w
        elif seq_len == 2048:
            grid_h, grid_w = 32, 64
        elif seq_len == 8192:
            grid_h, grid_w = 64, 128
        elif seq_len == 32768:
            grid_h, grid_w = 128, 256
        else:
            raise ValueError(
                f'Cannot infer grid dims from seq_len={seq_len}. '
                f'Pass --grid-h and --grid-w explicitly.'
            )
        print(f'  grid: {grid_h}x{grid_w} (seq_len={seq_len})')
        train_ds = AsciiDataset(train_tokens, augment=not args.no_augment, grid_h=grid_h, grid_w=grid_w)
        val_ds = AsciiDataset(val_tokens_full, augment=False, grid_h=grid_h, grid_w=grid_w)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- model ----
    model = AsciiVAE(
        vocab_size=N_ASCII_TOKENS,              # 8 (no MASK — VAE never sees MASK)
        grid_h=grid_h,
        grid_w=grid_w,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        downsample_stages=args.downsample_stages,
    ).to(device)

    print(
        f'Model: {model.num_params():,} params | '
        f'latent shape [{args.latent_channels}, {model.latent_h}, {model.latent_w}] | '
        f'{2 ** args.downsample_stages}x spatial compression'
    )

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
        tokens_2d = tokens.view(-1, grid_h, grid_w)

        lr = lr_schedule(step, args.peak_lr, warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        with torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')
        ):
            logits, mean, log_var = model(tokens_2d)
            total, recon, kl_raw, kl_used = vae_loss(
                logits, tokens_2d, mean, log_var,
                beta=args.beta, free_bits=args.free_bits,
            )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            its = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f'step {step:>6}  total {total.item():.3f}  recon {recon.item():.3f}  '
                f'kl {kl_raw.item():.2f}  lr {lr:.2e}  gn {grad_norm:.2f}  '
                f'{its:.2f} it/s'
            )

        if step > 0 and step % args.eval_every == 0:
            m = evaluate(model, val_loader, device, args.beta, args.free_bits, grid_h, grid_w)
            print(
                f'  [eval] recon {m["recon_per_image"]:.3f}  '
                f'kl {m["kl_raw_per_image"]:.2f}  '
                f'acc {m["accuracy"]:.4f}'
            )

        if step > 0 and step % args.sample_every == 0:
            print(f'  [samples @ step {step}]')
            render_reconstructions(model, val_tokens_full, n_samples=args.n_samples, grid_h=grid_h, grid_w=grid_w)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
                'grid': (grid_h, grid_w),
                'version': 'vae_v1',
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

    # Always save final checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': vars(args),
        'grid': (grid_h, grid_w),
        'version': 'vae_v1',
    }, ckpt_path)
    print(f'  [final ckpt] {ckpt_path}')
    if POST_CHECKPOINT_HOOK is not None:
        POST_CHECKPOINT_HOOK()

    print('VAE training complete.')


if __name__ == '__main__':
    main()
