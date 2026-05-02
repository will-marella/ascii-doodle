"""Train a flow-matching DiT directly on ASCII token grids (no VAE).

Tokens (0-7, ordinal brightness) are cast to float and normalized to
~N(0,1). The DiT learns the velocity field in this normalized space.
At inference: denormalize -> round -> clip to [0,7] -> map to RAMP chars.

Supports two data formats:
  - JSONL (legacy): --train-path foo.jsonl
  - Numpy (full QuickDraw): --train-npy dir/ (expects tokens.npy, labels.npy,
    categories.json from build_quickdraw_full.py)

When --train-npy is used with categories.json, CLIP text embeddings are
computed automatically for conditioning (one per category name).

Usage:
    # Unconditional (JSONL)
    python train_dit_direct.py \\
        --train-path quickdraw/dog_face_32x16.jsonl \\
        --grid-h 16 --grid-w 32

    # CLIP-conditioned (numpy, full dataset)
    python train_dit_direct.py \\
        --train-npy quickdraw/full_32x16 \\
        --grid-h 16 --grid-w 32 --dim 256 --n-layers 10

    # On Modal:
    modal run modal_app.py::train_dit_direct --gpu a10g --args "..."
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dit import DiT

# Optional callback for Modal volume commit
POST_CHECKPOINT_HOOK = None

RAMP = ' .:-+*#@'
N_TOKENS = 8


# ---- Dataset classes ----

class TokenDataset(Dataset):
    """Dataset from a flat token tensor (JSONL-loaded)."""
    def __init__(self, tokens, grid_h, grid_w, augment=False):
        self.tokens = tokens
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.augment = augment

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        t = self.tokens[idx].clone()
        if self.augment and torch.rand(1).item() < 0.5:
            t = t.view(self.grid_h, self.grid_w).flip(dims=[1]).reshape(-1)
        return t, -1  # no label


class NumpyTokenDataset(Dataset):
    """Memory-mapped dataset from tokens.npy + labels.npy."""
    def __init__(self, tokens_path, labels_path, augment=False):
        self.tokens = np.load(tokens_path, mmap_mode='r')  # [N, H, W]
        self.labels = np.load(labels_path, mmap_mode='r')   # [N]
        self.grid_h = self.tokens.shape[1]
        self.grid_w = self.tokens.shape[2]
        self.augment = augment

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        t = torch.from_numpy(self.tokens[idx].copy()).long()
        label = int(self.labels[idx])
        if self.augment and torch.rand(1).item() < 0.5:
            t = t.flip(dims=[1])
        return t.reshape(-1), label


# ---- Helpers ----

def lr_schedule(step, peak_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


def compute_token_stats(loader, grid_h, grid_w, n_batches=100):
    """Compute mean and std of token values (as floats) across the dataset."""
    all_vals = []
    for i, (tokens, _) in enumerate(loader):
        if i >= n_batches:
            break
        all_vals.append(tokens.float().view(-1, grid_h, grid_w))
    all_vals = torch.cat(all_vals)
    mean = all_vals.mean()
    std = all_vals.std()
    print(f'  Token stats: mean={mean:.4f} std={std:.4f}')
    return mean, std


def compute_clip_embeddings(categories, device='cpu'):
    """Compute CLIP text embeddings for category names."""
    from transformers import CLIPModel, CLIPTokenizer
    print(f'Computing CLIP text embeddings for {len(categories)} categories...')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    prompts = [f'a drawing of a {cat}' for cat in categories]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embs = model.get_text_features(**inputs)
        embs = embs.float().cpu()  # [n_cats, 512]
    clip_dim = embs.shape[1]
    print(f'  CLIP dim: {clip_dim}, embeddings shape: {embs.shape}')
    del model, tokenizer
    return embs, clip_dim


def flow_matching_loss(model, x_1, clip_emb=None, cfg_dropout=0.1):
    """Flow matching loss on normalized token grids."""
    B = x_1.shape[0]
    device = x_1.device
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_b) * x_0 + t_b * x_1
    v_target = x_1 - x_0

    # CFG dropout
    if clip_emb is not None and cfg_dropout > 0:
        keep = torch.rand(B, device=device) > cfg_dropout
        keep = keep.view(B, 1).expand_as(clip_emb)
        null = model.get_null_clip(B)
        clip_emb = torch.where(keep, clip_emb, null)

    v_pred = model(x_t, t, clip_emb=clip_emb)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_flow(model, n_samples, n_steps=50, clip_emb=None,
                guidance_scale=0.0, device='cpu'):
    """Euler-integrate from noise to data."""
    model.eval()
    shape = (n_samples, model.latent_channels, model.latent_h, model.latent_w)
    x = torch.randn(*shape, device=device)
    dt = 1.0 / n_steps

    use_cfg = (clip_emb is not None and guidance_scale > 1.0
               and model.clip_proj is not None)
    null_emb = model.get_null_clip(n_samples) if model.clip_proj is not None else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        if use_cfg:
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            emb_double = torch.cat([clip_emb, null_emb], dim=0)
            v_both = model(x_double, t_double, clip_emb=emb_double)
            v_cond, v_uncond = v_both.chunk(2, dim=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        elif clip_emb is not None:
            v = model(x, t, clip_emb=clip_emb)
        else:
            v = model(x, t)
        x = x + v * dt
    model.train()
    return x


def render_sample(token_grid):
    """Convert a [H, W] int tensor to ASCII string."""
    lines = []
    for row in token_grid:
        lines.append(''.join(RAMP[min(max(int(v), 0), len(RAMP)-1)] for v in row))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    # data (one of these required)
    p.add_argument('--train-path', default=None,
                   help='JSONL training data (legacy single-category)')
    p.add_argument('--train-npy', default=None,
                   help='Directory with tokens.npy, labels.npy, categories.json')
    p.add_argument('--checkpoint-dir', default='checkpoints/dit_direct_v1')
    p.add_argument('--resume', default=None)
    # grid
    p.add_argument('--grid-h', type=int, required=True)
    p.add_argument('--grid-w', type=int, required=True)
    # DiT model
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--n-layers', type=int, default=6)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--ffn-mult', type=int, default=4)
    # conditioning
    p.add_argument('--cfg-dropout', type=float, default=0.1)
    p.add_argument('--guidance-scale', type=float, default=3.0)
    # training
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--peak-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-frac', type=float, default=0.02)
    p.add_argument('--total-steps', type=int, default=50000)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--no-augment', action='store_true')
    # sampling
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--sample-prompts', nargs='*',
                   default=['dog', 'cat', 'car', 'bird', 'house',
                            'fish', 'bicycle', 'tree', 'airplane', 'guitar'],
                   help='Category names to sample at eval time')
    # logging
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--sample-every', type=int, default=2000)
    p.add_argument('--ckpt-every', type=int, default=5000)
    p.add_argument('--n-samples', type=int, default=4)
    # misc
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    grid_h, grid_w = args.grid_h, args.grid_w

    # ---- data ----
    categories = None
    clip_embs = None  # [n_cats, clip_dim]
    clip_dim = 0

    if args.train_npy:
        print(f'Loading numpy dataset from {args.train_npy}/')
        tokens_path = os.path.join(args.train_npy, 'tokens.npy')
        labels_path = os.path.join(args.train_npy, 'labels.npy')
        cats_path = os.path.join(args.train_npy, 'categories.json')

        train_ds = NumpyTokenDataset(tokens_path, labels_path,
                                     augment=not args.no_augment)
        print(f'  {len(train_ds):,} training examples, grid {train_ds.grid_h}x{train_ds.grid_w}')
        grid_h, grid_w = train_ds.grid_h, train_ds.grid_w

        if os.path.exists(cats_path):
            with open(cats_path) as f:
                categories = json.load(f)
            print(f'  {len(categories)} categories')

            # Load pre-computed CLIP embeddings if available, else compute
            clip_path = os.path.join(args.train_npy, 'clip_embeddings.npy')
            if os.path.exists(clip_path):
                clip_embs = torch.from_numpy(np.load(clip_path)).float().to(device)
                clip_dim = clip_embs.shape[1]
                print(f'  Loaded pre-computed CLIP embeddings: {clip_embs.shape}')
            else:
                clip_embs, clip_dim = compute_clip_embeddings(categories, device=device)
                clip_embs = clip_embs.to(device)

    elif args.train_path:
        print(f'Loading JSONL from {args.train_path}...')
        from data import load_jsonl_to_tensors
        train_tokens = load_jsonl_to_tensors(args.train_path, limit=args.limit_train)
        print(f'  {len(train_tokens):,} training examples')
        train_ds = TokenDataset(train_tokens, grid_h, grid_w,
                                augment=not args.no_augment)
    else:
        raise ValueError('Provide --train-path (JSONL) or --train-npy (numpy dir)')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    print(f'Mode: Direct DiT flow matching (no VAE)')
    print(f'Grid: {grid_h}x{grid_w} = {grid_h * grid_w} tokens')
    print(f'CLIP conditioning: {"yes (dim=" + str(clip_dim) + ")" if clip_dim > 0 else "no (unconditional)"}')
    print(f'Device: {device}')

    # Compute normalization stats
    print('Computing token stats...')
    tok_mean, tok_std = compute_token_stats(train_loader, grid_h, grid_w)
    tok_mean = tok_mean.to(device)
    tok_std = tok_std.to(device)

    # ---- model ----
    model = DiT(
        latent_channels=1,
        latent_h=grid_h,
        latent_w=grid_w,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        clip_dim=clip_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'DiT: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}')
    print(f'  {n_params:,} parameters')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print(f'  Resumed at step {start_step}')

    warmup_steps = int(args.total_steps * args.warmup_frac)
    print(f'Training for {args.total_steps} steps (warmup={warmup_steps})')

    # Pre-compute sample prompt embeddings
    sample_clip_embs = None
    if clip_dim > 0 and categories:
        sample_prompts = args.sample_prompts
        prompt_indices = []
        valid_prompts = []
        for sp in sample_prompts:
            if sp in categories:
                prompt_indices.append(categories.index(sp))
                valid_prompts.append(sp)
        if prompt_indices:
            sample_clip_embs = clip_embs[prompt_indices]  # [n_prompts, clip_dim]
            print(f'Sample prompts: {valid_prompts}')

    print()

    # ---- training loop ----
    model.train()
    data_iter = iter(train_loader)
    t_start = time.time()
    running_loss = 0.0

    step = start_step
    while step < args.total_steps:
        try:
            batch_tokens, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_tokens, batch_labels = next(data_iter)

        x = batch_tokens.float().to(device).view(-1, grid_h, grid_w)
        x = (x - tok_mean) / tok_std
        x = x.unsqueeze(1)  # [B, 1, H, W]

        # Look up CLIP embeddings for this batch's labels
        batch_clip = None
        if clip_embs is not None:
            batch_labels_dev = batch_labels.to(device)
            batch_clip = clip_embs[batch_labels_dev]  # [B, clip_dim]

        lr = lr_schedule(step, args.peak_lr, warmup_steps, args.total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = flow_matching_loss(model, x, clip_emb=batch_clip,
                                  cfg_dropout=args.cfg_dropout)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if step > 0 and step % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1)
            eta = (args.total_steps - step) / max(steps_per_sec, 1e-6)
            print(f'  step {step:>6d} | loss={avg:.4f} | lr={lr:.2e} | '
                  f'{steps_per_sec:.1f} step/s | eta={eta/60:.0f}m')
            running_loss = 0.0

        if step > 0 and step % args.sample_every == 0:
            print(f'\n  [samples @ step {step}]')

            if sample_clip_embs is not None:
                # Conditioned samples
                n_prompts = len(sample_clip_embs)
                n_per = min(args.n_samples, 2)
                for pi in range(n_prompts):
                    emb = sample_clip_embs[pi:pi+1].expand(n_per, -1).to(device)
                    raw = sample_flow(model, n_per, args.sample_steps,
                                      clip_emb=emb,
                                      guidance_scale=args.guidance_scale,
                                      device=device)
                    raw = raw.squeeze(1) * tok_std.cpu() + tok_mean.cpu()
                    toks = raw.round().clamp(0, N_TOKENS - 1).long()
                    print(f'  --- {valid_prompts[pi]} (cfg={args.guidance_scale}) ---')
                    print(render_sample(toks[0]))
                    print()
            else:
                # Unconditional samples
                raw = sample_flow(model, args.n_samples, args.sample_steps,
                                  device=device)
                raw = raw.squeeze(1) * tok_std.cpu() + tok_mean.cpu()
                toks = raw.round().clamp(0, N_TOKENS - 1).long()
                for i in range(min(args.n_samples, 4)):
                    print(f'  --- sample {i+1} ---')
                    print(render_sample(toks[i]))
                    print()
            model.train()

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
                'grid': (grid_h, grid_w),
                'tok_mean': tok_mean.cpu(),
                'tok_std': tok_std.cpu(),
                'clip_dim': clip_dim,
                'categories': categories,
                'version': 'dit_direct_v2_clip' if clip_dim > 0 else 'dit_direct_v1',
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

    # Final checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': vars(args),
        'grid': (grid_h, grid_w),
        'tok_mean': tok_mean.cpu(),
        'tok_std': tok_std.cpu(),
        'clip_dim': clip_dim,
        'categories': categories,
        'version': 'dit_direct_v2_clip' if clip_dim > 0 else 'dit_direct_v1',
    }, ckpt_path)
    print(f'  [final ckpt] {ckpt_path}')
    if POST_CHECKPOINT_HOOK is not None:
        POST_CHECKPOINT_HOOK()

    print('Training complete.')


if __name__ == '__main__':
    main()
