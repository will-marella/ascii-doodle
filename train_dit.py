"""Train a DiT in the VAE's latent space using flow matching.

Pipeline:
  1. Load a frozen, trained VAE
  2. Encode training batches into continuous latents [B, C, H', W']
  3. Normalize latents to ~unit variance per channel (computed once at start)
  4. Train a DiT to predict the flow-matching velocity field x_1 - x_0
     where x_0 ~ N(0, I), x_1 = real_latent, x_t = (1-t)x_0 + t*x_1
  5. Periodically sample new latents from N(0, I) via Euler integration,
     unnormalize, VAE-decode, and print as ASCII

Loss:
    L = E[|| model(x_t, t) - (x_1 - x_0) ||^2]
"""

import argparse
import math
import os
import time

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    AsciiDataset,
    N_ASCII_TOKENS,
    load_clip_embeddings,
    load_jsonl_to_tensors,
)
from ascii_utils import decode
from clip_utils import text_to_clip_embeddings
from dit import DiT
from vae import AsciiVAE


# Default label filter — set to None to use ALL classes (diversity experiment).
# Override via --filter-labels.
FILTER_LABELS = None

# Hand-picked diverse prompts for text-to-ASCII progress tracking during
# training. Encoded once via CLIP's text encoder at startup and re-used at
# every sample render step.
SAMPLE_PROMPTS = [
    "dog", "cat", "car", "bird", "house", "fish",
    "bicycle", "tree", "airplane", "guitar",
]

POST_CHECKPOINT_HOOK = None


def lr_schedule(step, peak_lr, warmup_steps, total_steps, min_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = 'emb' in name
        is_norm = 'norm' in name or 'ln' in name
        if p.dim() < 2 or is_embedding or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def load_vae(path: str, device: str):
    """Load a trained VAE checkpoint and return (model, grid_h, grid_w)."""
    print(f'Loading VAE from {path}...')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    grid_h, grid_w = ckpt.get('grid', (config.get('grid_h', 32), config.get('grid_w', 64)))

    vae = AsciiVAE(
        vocab_size=N_ASCII_TOKENS,
        grid_h=grid_h,
        grid_w=grid_w,
        embed_dim=config['embed_dim'],
        base_channels=config['base_channels'],
        latent_channels=config['latent_channels'],
        downsample_stages=config['downsample_stages'],
    ).to(device)
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print(
        f'  step {ckpt["step"]}, {vae.num_params():,} params, '
        f'grid {grid_h}x{grid_w}, latent [{vae.latent_channels}, {vae.latent_h}, {vae.latent_w}]'
    )
    return vae, grid_h, grid_w


@torch.no_grad()
def compute_latent_stats(vae, loader, device, grid_h, grid_w, n_batches=20):
    """Compute per-channel mean and std of VAE latents on a sample of training data.

    Used to normalize latents into ~unit variance before flow matching.
    Returns (mean [C], std [C]) — both should be broadcast as [1, C, 1, 1].
    """
    print(f'Computing latent statistics on {n_batches} batches...')
    sums = []
    sq_sums = []
    n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Dataset may return tokens alone or (tokens, clip_emb) — unpack either way
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        tokens = tokens.to(device).view(-1, grid_h, grid_w)
        mean, _ = vae.encode(tokens)        # use the mean (deterministic)
        # mean: [B, C, H', W'] — collapse over batch and spatial
        flat = mean.permute(1, 0, 2, 3).reshape(mean.size(1), -1)  # [C, B*H*W]
        sums.append(flat.sum(dim=1))
        sq_sums.append((flat ** 2).sum(dim=1))
        n += flat.size(1)
    total_sum = torch.stack(sums).sum(dim=0)
    total_sq = torch.stack(sq_sums).sum(dim=0)
    mean = total_sum / n
    var = (total_sq / n) - mean ** 2
    std = var.clamp(min=1e-8).sqrt()
    print(f'  mean per channel:  min={mean.min():.3f} max={mean.max():.3f} avg={mean.mean():.3f}')
    print(f'  std per channel:   min={std.min():.3f} max={std.max():.3f} avg={std.mean():.3f}')
    return mean, std


def normalize_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (z - mean[None, :, None, None]) / std[None, :, None, None]


def unnormalize_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return z * std[None, :, None, None] + mean[None, :, None, None]


def flow_matching_loss(
    model: DiT,
    x_1: torch.Tensor,
    clip_emb: torch.Tensor = None,
    cfg_dropout: float = 0.0,
) -> torch.Tensor:
    """Linear-interpolant flow matching loss.

    Sample x_0 ~ N(0,I), t ~ U[0,1]. Define x_t = (1-t) x_0 + t x_1.
    The target velocity is constant along the path: v* = x_1 - x_0.

    If clip_emb is provided and cfg_dropout > 0, randomly replace some samples'
    CLIP embeddings with the model's null embedding (per-sample, with the given
    probability). This trains the model to handle both conditional and
    unconditional generation in one pass, enabling CFG at inference time.
    """
    B = x_1.size(0)
    device = x_1.device
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_b) * x_0 + t_b * x_1
    v_target = x_1 - x_0

    # CFG dropout: replace some samples' clip_emb with the null embedding
    if clip_emb is not None and cfg_dropout > 0:
        keep = (torch.rand(B, device=device) > cfg_dropout).unsqueeze(-1)  # [B, 1]
        null = model.get_null_clip(B)
        clip_emb = torch.where(keep, clip_emb, null)

    v_pred = model(x_t, t, clip_emb=clip_emb)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_flow(
    model: DiT,
    n_samples: int,
    n_steps: int = 50,
    clip_emb: torch.Tensor = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Euler-integrate the velocity field from N(0, I) to data.

    If clip_emb is None: unconditional sampling (model must have no clip_dim or
    the caller is responsible for substituting null).
    If clip_emb is provided and guidance_scale > 1: classifier-free guidance.
    Each step does TWO forward passes (cond + null) and combines:
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    shape = (n_samples, model.latent_channels, model.latent_h, model.latent_w)
    x = torch.randn(*shape, device=device)
    dt = 1.0 / n_steps

    use_cfg = (clip_emb is not None) and (guidance_scale > 1.0) and (model.clip_proj is not None)
    null_emb = model.get_null_clip(n_samples) if model.clip_proj is not None else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        if use_cfg:
            # Single batched forward: [conditional; unconditional]
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            emb_double = torch.cat([clip_emb, null_emb], dim=0)
            v_both = model(x_double, t_double, clip_emb=emb_double)
            v_cond, v_uncond = v_both.chunk(2, dim=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, clip_emb=clip_emb)
        x = x + dt * v

    if was_training:
        model.train()
    return x


def render_dit_samples(
    model, vae, latent_mean, latent_std, n_samples, n_steps, grid_w,
    clip_emb=None, guidance_scale=0.0, tags=None, header=None,
):
    """Sample from DiT, decode through VAE, and print as ASCII.

    Args:
        tags: optional list of strings (e.g. labels or prompts) printed
              next to each sample index for readability.
        header: optional section header printed before the samples.
    """
    if header:
        print(f'  {header}')
    z_norm = sample_flow(
        model, n_samples, n_steps=n_steps,
        clip_emb=clip_emb, guidance_scale=guidance_scale,
    )
    z = unnormalize_latent(z_norm, latent_mean, latent_std)
    with torch.no_grad():
        logits = vae.decode(z)            # [B, V, H, W]
        tokens = logits.argmax(dim=1)     # [B, H, W]
    for i in range(n_samples):
        tag = f' — {tags[i]}' if (tags and i < len(tags)) else ''
        print(f'  --- sample {i+1}/{n_samples}{tag} ---')
        flat = tokens[i].reshape(-1).tolist()
        for line in decode(flat, grid_w=grid_w).split('\n'):
            print(f'    {line}')


def main(argv=None):
    p = argparse.ArgumentParser()
    # paths
    p.add_argument('--vae-checkpoint', required=True,
                   help='Path to a trained VAE checkpoint (frozen).')
    p.add_argument('--train-path', default='openimages/train_ascii_128x64_cc.jsonl')
    p.add_argument('--val-path', default='openimages/validation_ascii_128x64_cc.jsonl')
    p.add_argument('--train-npy', default=None,
                   help='Directory with tokens.npy, labels.npy, clip_embeddings.npy '
                        '(from build_quickdraw_full.py). Overrides --train-path.')
    p.add_argument('--clip-embeddings', default=None,
                   help='Path to a parallel .npy file of CLIP image embeddings '
                        'aligned with the train JSONL. If provided, the DiT is '
                        'trained as CLIP-conditional with CFG dropout.')
    p.add_argument('--checkpoint-dir', default='checkpoints/dit_v1')
    p.add_argument('--resume', default=None)
    # DiT model
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--n-layers', type=int, default=12)
    p.add_argument('--n-heads', type=int, default=6)
    p.add_argument('--ffn-mult', type=int, default=4)
    p.add_argument('--cfg-dropout', type=float, default=0.1,
                   help='Probability of replacing CLIP embedding with null '
                        'during training. Enables CFG at inference time.')
    p.add_argument('--guidance-scale', type=float, default=3.0,
                   help='CFG scale for sample-time generation. >1 amplifies '
                        'conditioning. Used during render_dit_samples.')
    # training
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--peak-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-frac', type=float, default=0.02)
    p.add_argument('--total-steps', type=int, default=20000)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--no-augment', action='store_true')
    # sampling
    p.add_argument('--sample-steps', type=int, default=50,
                   help='Number of Euler steps for flow matching sampling.')
    # logging / eval
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=2000,
                   help='Compute val loss every N steps (0 to disable)')
    p.add_argument('--eval-batches', type=int, default=50,
                   help='Number of val batches per eval')
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--ckpt-every', type=int, default=1000)
    p.add_argument('--n-samples', type=int, default=4)
    p.add_argument('--latent-stats-batches', type=int, default=20)
    # misc
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--filter-labels', default='',
                   help='Comma-separated list of class labels to keep. '
                        'Empty (default) = use all classes.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    # Build label filter from CLI arg
    filter_labels = None
    if args.filter_labels:
        filter_labels = frozenset(l.strip() for l in args.filter_labels.split(',') if l.strip())

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print('Mode: DiT flow-matching training in VAE latent space')
    print(f'CLIP conditioning: {"yes" if (args.clip_embeddings or args.train_npy) else "no (unconditional)"}')

    # ---- VAE (frozen) ----
    vae, grid_h, grid_w = load_vae(args.vae_checkpoint, device)

    # ---- data ----
    train_clip = None
    clip_dim = 0

    if args.train_npy:
        import json as _json
        from torch.utils.data import Dataset as _Dataset

        class _NpyClipDataset(_Dataset):
            """Numpy dataset that returns (tokens_flat, clip_emb) tuples."""
            def __init__(self, tokens_npy, labels_npy, clip_embs, augment=False):
                self.tokens = np.load(tokens_npy, mmap_mode='r')
                self.labels = np.load(labels_npy, mmap_mode='r')
                self.clip_embs = clip_embs  # [n_cats, clip_dim] tensor
                self.grid_h = self.tokens.shape[1]
                self.grid_w = self.tokens.shape[2]
                self.augment = augment
            def __len__(self):
                return len(self.tokens)
            def __getitem__(self, idx):
                t = torch.from_numpy(self.tokens[idx].copy()).long()
                if self.augment and torch.rand(1).item() < 0.5:
                    t = t.flip(dims=[1])
                clip = self.clip_embs[int(self.labels[idx])]
                return t.reshape(-1), clip

        tokens_path = os.path.join(args.train_npy, 'tokens.npy')
        labels_path = os.path.join(args.train_npy, 'labels.npy')
        clip_path = os.path.join(args.train_npy, 'clip_embeddings.npy')
        cats_path = os.path.join(args.train_npy, 'categories.json')

        print(f'Loading numpy dataset from {args.train_npy}/')
        clip_embs_np = np.load(clip_path)
        clip_embs_t = torch.from_numpy(clip_embs_np).float()
        clip_dim = clip_embs_t.shape[1]

        if os.path.exists(cats_path):
            with open(cats_path) as f:
                categories = _json.load(f)
            print(f'  {len(categories)} categories, clip_dim={clip_dim}')

        train_ds = _NpyClipDataset(tokens_path, labels_path, clip_embs_t,
                                   augment=not args.no_augment)
        grid_h, grid_w = train_ds.grid_h, train_ds.grid_w
        print(f'  {len(train_ds):,} training examples, grid {grid_h}x{grid_w}')

    else:
        print(f'Filter labels: {sorted(filter_labels) if filter_labels else "(all classes)"}')
        print(f'Loading train data from {args.train_path}...')
        train_tokens = load_jsonl_to_tensors(
            args.train_path, filter_labels=filter_labels, limit=args.limit_train,
        )
        print(f'  {len(train_tokens):,} training examples')

        if args.clip_embeddings:
            train_clip = load_clip_embeddings(args.clip_embeddings)
            if args.limit_train:
                train_clip = train_clip[:args.limit_train]
            assert len(train_clip) == len(train_tokens), (
                f'CLIP embeddings ({len(train_clip)}) do not match tokens ({len(train_tokens)}).'
            )
            clip_dim = train_clip.shape[1]
            print(f'  clip_dim = {clip_dim}')

        train_ds = AsciiDataset(
            train_tokens,
            augment=not args.no_augment,
            grid_h=grid_h, grid_w=grid_w,
            clip_embeddings=train_clip,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # ---- val data (small held-out subset for loss tracking) ----
    val_loader = None
    if args.train_npy and args.eval_every > 0:
        # Use the tail end of the dataset as val
        val_size = min(10000, len(train_ds) // 10)
        val_start = len(train_ds) - val_size
        tokens_all = np.load(os.path.join(args.train_npy, 'tokens.npy'), mmap_mode='r')
        labels_all = np.load(os.path.join(args.train_npy, 'labels.npy'), mmap_mode='r')
        clip_path = os.path.join(args.train_npy, 'clip_embeddings.npy')
        _clip_for_val = torch.from_numpy(np.load(clip_path)).float()

        class _ValDataset(torch.utils.data.Dataset):
            def __init__(self, tokens, labels, clip_embs, start, size):
                self.tokens = tokens
                self.labels = labels
                self.clip_embs = clip_embs
                self.start = start
                self.size = size
                self.grid_h = tokens.shape[1]
                self.grid_w = tokens.shape[2]
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                i = self.start + idx
                t = torch.from_numpy(self.tokens[i].copy()).long().reshape(-1)
                clip = self.clip_embs[int(self.labels[i])]
                return t, clip

        val_ds = _ValDataset(tokens_all, labels_all, _clip_for_val, val_start, val_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        print(f'Val set: {val_size:,} examples (tail of dataset)')

    # ---- latent normalization stats ----
    latent_mean, latent_std = compute_latent_stats(
        vae, train_loader, device, grid_h, grid_w, n_batches=args.latent_stats_batches,
    )

    # ---- DiT ----
    dit = DiT(
        latent_channels=vae.latent_channels,
        latent_h=vae.latent_h,
        latent_w=vae.latent_w,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        clip_dim=clip_dim,
    ).to(device)
    print(
        f'DiT: {dit.num_params():,} params | '
        f'{args.n_layers}L x {args.dim}d x {args.n_heads}h | '
        f'seq_len={dit.seq_len} | clip_dim={clip_dim}'
    )

    optimizer = torch.optim.AdamW(
        get_param_groups(dit, args.weight_decay),
        lr=args.peak_lr, betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(args.total_steps * args.warmup_frac))

    # ---- resume ----
    start_step = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        dit.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']

    # ---- training loop ----
    dit.train()
    step = start_step
    t_start = time.time()
    train_iter = iter(train_loader)
    print(f'Training from step {step} to {args.total_steps}')

    # If we have CLIP conditioning, capture a few sample embeddings up front
    # so we can render the same "subjects" across the run for visual comparison.
    sample_clip = None
    sample_labels = None
    sample_text_clip = None

    if clip_dim > 0:
        # For numpy path: use the pre-computed per-category CLIP embeddings
        if args.train_npy:
            clip_path = os.path.join(args.train_npy, 'clip_embeddings.npy')
            cats_path = os.path.join(args.train_npy, 'categories.json')
            _clip_embs = torch.from_numpy(np.load(clip_path)).float().to(device)

            if os.path.exists(cats_path):
                import json as _json
                with open(cats_path) as f:
                    _categories = _json.load(f)
                # Use category names as sample prompts
                sample_prompts = [c for c in SAMPLE_PROMPTS if c in _categories]
                if not sample_prompts:
                    sample_prompts = _categories[:min(8, len(_categories))]
                sample_labels = sample_prompts
                sample_text_clip = torch.stack([
                    _clip_embs[_categories.index(c)] for c in sample_prompts
                ]).to(device)
                print(f'Sample prompts (from categories): {sample_prompts}')

        # For JSONL path: use train_clip directly
        elif train_clip is not None:
            with torch.no_grad():
                sample_clip = train_clip[:args.n_samples].to(device)

            import json as _json
            sample_labels = []
            with open(args.train_path) as f:
                for i, line in enumerate(f):
                    if i >= args.n_samples:
                        break
                    r = _json.loads(line)
                    sample_labels.append(r.get('label', '?'))
            print(f'Image-embedding sample labels: {sample_labels}')

        # Also try encoding text prompts via CLIP text encoder
        if sample_text_clip is None:
            try:
                print(f'Encoding {len(SAMPLE_PROMPTS)} text prompts via CLIP...')
                sample_text_clip = text_to_clip_embeddings(
                    SAMPLE_PROMPTS,
                    device=torch.device(device),
                ).to(device)
                sample_labels = list(SAMPLE_PROMPTS)
                print(f'  shape: {tuple(sample_text_clip.shape)}')
                print(f'  prompts: {SAMPLE_PROMPTS}')
            except Exception as e:
                print(f'Warning: text prompt encoding failed ({e}); '
                      f'skipping text-prompt samples.')
                sample_text_clip = None

    while step < args.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if isinstance(batch, (tuple, list)):
            tokens, clip_emb = batch
            tokens = tokens.to(device, non_blocking=True)
            clip_emb = clip_emb.to(device, non_blocking=True)
        else:
            tokens = batch.to(device, non_blocking=True)
            clip_emb = None

        tokens_2d = tokens.view(-1, grid_h, grid_w)

        lr = lr_schedule(step, args.peak_lr, warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Encode through frozen VAE (no autocast — use the VAE's natural dtype)
        with torch.no_grad():
            mean, _ = vae.encode(tokens_2d)
        z = normalize_latent(mean, latent_mean, latent_std)

        with torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')
        ):
            loss = flow_matching_loss(
                dit, z,
                clip_emb=clip_emb,
                cfg_dropout=args.cfg_dropout if clip_emb is not None else 0.0,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            its = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f'step {step:>6}  loss {loss.item():.4f}  lr {lr:.2e}  '
                f'gn {grad_norm:.2f}  {its:.2f} it/s'
            )

        if step > 0 and args.eval_every > 0 and step % args.eval_every == 0 and val_loader is not None:
            dit.eval()
            val_losses = []
            val_iter = iter(val_loader)
            with torch.no_grad():
                for vi in range(args.eval_batches):
                    try:
                        vb_tokens, vb_clip = next(val_iter)
                    except StopIteration:
                        break
                    vb_tokens = vb_tokens.to(device)
                    vb_clip = vb_clip.to(device)
                    vb_2d = vb_tokens.view(-1, grid_h, grid_w)
                    vb_mean, _ = vae.encode(vb_2d)
                    vb_z = normalize_latent(vb_mean, latent_mean, latent_std)
                    vl = flow_matching_loss(dit, vb_z, clip_emb=vb_clip, cfg_dropout=0.0)
                    val_losses.append(vl.item())
            val_avg = sum(val_losses) / len(val_losses) if val_losses else 0
            print(f'  [val @ step {step}] val_loss={val_avg:.4f}')
            dit.train()

        if step > 0 and step % args.sample_every == 0:
            print(f'  [samples @ step {step}]')
            # Image-embedding targets (JSONL path only)
            if sample_clip is not None:
                render_dit_samples(
                    dit, vae, latent_mean, latent_std,
                    n_samples=args.n_samples,
                    n_steps=args.sample_steps,
                    grid_w=grid_w,
                    clip_emb=sample_clip,
                    guidance_scale=args.guidance_scale,
                    tags=sample_labels,
                    header='[image-embedding targets]',
                )
            # Text-prompt / category targets
            if sample_text_clip is not None:
                render_dit_samples(
                    dit, vae, latent_mean, latent_std,
                    n_samples=len(sample_text_clip),
                    n_steps=args.sample_steps,
                    grid_w=grid_w,
                    clip_emb=sample_text_clip,
                    guidance_scale=args.guidance_scale,
                    tags=sample_labels or SAMPLE_PROMPTS,
                    header='[text-prompt targets]',
                )

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model': dit.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(args),
                'grid': (grid_h, grid_w),
                'latent_mean': latent_mean.cpu(),
                'latent_std': latent_std.cpu(),
                'clip_dim': clip_dim,
                'version': 'dit_v2_clip' if clip_dim > 0 else 'dit_v1',
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
        'model': dit.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': vars(args),
        'grid': (grid_h, grid_w),
        'latent_mean': latent_mean.cpu(),
        'latent_std': latent_std.cpu(),
        'clip_dim': clip_dim,
        'version': 'dit_v2_clip' if clip_dim > 0 else 'dit_v1',
    }, ckpt_path)
    print(f'  [final ckpt] {ckpt_path}')
    if POST_CHECKPOINT_HOOK is not None:
        POST_CHECKPOINT_HOOK()

    print('DiT training complete.')


if __name__ == '__main__':
    main()
