"""Modal app for training the ASCII transformer.

---- One-time setup ----
Upload the dataset to a persistent Modal volume (only needed once, or when
you regenerate the JSONLs):

    modal volume create ascii-generator-data
    modal volume put ascii-generator-data openimages/train_ascii_64x32_cc.jsonl /openimages/train_ascii_64x32_cc.jsonl
    modal volume put ascii-generator-data openimages/validation_ascii_64x32_cc.jsonl /openimages/validation_ascii_64x32_cc.jsonl

Verify:
    modal volume ls ascii-generator-data /openimages

---- Training ----
    # Smoke test (A10G, ~2 min, tiny subset, fast eval cadence)
    modal run modal_app.py::train --args "--limit-train 2000 --total-steps 100 --eval-every 50 --sample-every 50 --ckpt-every 100 --batch-size 16"

    # Full run on A10G (cheapest, ~3-4 hours)
    modal run modal_app.py::train

    # Full run on A100 (~1-1.5 hours, 3x the cost)
    modal run modal_app.py::train --gpu a100

    # Custom args passed through to train.py
    modal run modal_app.py::train --args "--total-steps 30000 --peak-lr 2e-4"

---- After training ----
    # List checkpoints in the volume
    modal volume ls ascii-generator-data /checkpoints

    # Pull a checkpoint back locally
    modal volume get ascii-generator-data /checkpoints/step_20000.pt .
"""

import os

import modal

APP_NAME = "ascii-generator"
VOLUME_NAME = "ascii-generator-data"
VOLUME_MOUNT = "/data"

app = modal.App(APP_NAME)

# The training code only needs torch + numpy — scipy is for the offline data
# rebuild script and isn't used at train time.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "scipy",
        "Pillow",
        "transformers==4.44.0",
    )
    .add_local_python_source(
        "data", "model", "sample", "train", "inspect_samples", "probe_prefix",
        "vae", "train_vae", "dit", "train_dit", "train_dit_direct",
        "rebuild_fixed_canvas", "compute_clip_embeddings",
        "compute_clip_text_embeddings", "sample_dit", "sample_prompts",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_train(extra_args):
    """Shared training entry point used by both GPU-specific wrappers."""
    import train as train_mod
    os.chdir(VOLUME_MOUNT)
    os.makedirs("checkpoints", exist_ok=True)

    # Flush the volume after every checkpoint save so long runs are crash-safe.
    # volume.commit() takes ~10-30s but runs only every ckpt_every steps.
    def _commit():
        print("  [volume] committing...")
        volume.commit()

    train_mod.POST_CHECKPOINT_HOOK = _commit
    train_mod.main(extra_args or [])
    volume.commit()
    print("Final volume commit done.")


# Modal pins GPU type at function-decoration time, so we define one function
# per GPU choice and dispatch from the local entrypoint.

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 60 * 6,  # 6h — generous cap for a POC run
)
def train_a10g(extra_args: list = None):
    _run_train(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A100",
    timeout=60 * 60 * 6,
)
def train_a100(extra_args: list = None):
    _run_train(extra_args)


@app.local_entrypoint()
def train(gpu: str = "a10g", args: str = ""):
    """Launch a MaskGIT training run on Modal.

    Args:
        gpu: 'a10g' (default, cheap) or 'a100' (faster).
        args: passthrough string, quoted, forwarded to train.py's argparse.
              e.g. --args "--total-steps 500 --limit-train 2000"
    """
    extra_args = args.split() if args else []
    gpu_key = gpu.lower()
    if gpu_key == "a10g":
        train_a10g.remote(extra_args)
    elif gpu_key == "a100":
        train_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


# ---- VAE training ----

def _run_vae(extra_args):
    """Shared VAE training entry point."""
    import train_vae as train_vae_mod
    os.chdir(VOLUME_MOUNT)
    os.makedirs("checkpoints", exist_ok=True)

    def _commit():
        print("  [volume] committing...")
        volume.commit()

    train_vae_mod.POST_CHECKPOINT_HOOK = _commit
    train_vae_mod.main(extra_args or [])
    volume.commit()
    print("Final volume commit done.")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
)
def train_vae_a10g(extra_args: list = None):
    _run_vae(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A100",
    timeout=60 * 60 * 6,
)
def train_vae_a100(extra_args: list = None):
    _run_vae(extra_args)


@app.local_entrypoint()
def train_vae(gpu: str = "a10g", args: str = ""):
    """Train the ASCII VAE on Modal.

    Args:
        gpu: 'a10g' (default, cheap) or 'a100' (faster).
        args: passthrough string forwarded to train_vae.py's argparse.
              e.g. --args "--total-steps 5000 --beta 0.5"
    """
    extra_args = args.split() if args else []
    gpu_key = gpu.lower()
    if gpu_key == "a10g":
        train_vae_a10g.remote(extra_args)
    elif gpu_key == "a100":
        train_vae_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


# ---- DiT (latent flow matching) training ----

def _run_dit(extra_args):
    """Shared DiT training entry point."""
    import train_dit as train_dit_mod
    os.chdir(VOLUME_MOUNT)
    os.makedirs("checkpoints", exist_ok=True)

    def _commit():
        print("  [volume] committing...")
        volume.commit()

    train_dit_mod.POST_CHECKPOINT_HOOK = _commit
    train_dit_mod.main(extra_args or [])
    volume.commit()
    print("Final volume commit done.")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
)
def train_dit_a10g(extra_args: list = None):
    _run_dit(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A100",
    timeout=60 * 60 * 6,
)
def train_dit_a100(extra_args: list = None):
    _run_dit(extra_args)


@app.local_entrypoint()
def train_dit(gpu: str = "a10g", args: str = ""):
    """Train a DiT in the VAE latent space (flow matching) on Modal.

    Required: --vae-checkpoint pointing to a trained VAE checkpoint.

    Args:
        gpu: 'a10g' (default, cheap) or 'a100' (faster).
        args: passthrough forwarded to train_dit.py's argparse.
              e.g. --args "--vae-checkpoint /data/checkpoints/vae_128x64_8x/step_10000.pt"
    """
    extra_args = args.split() if args else []
    gpu_key = gpu.lower()
    if gpu_key == "a10g":
        train_dit_a10g.remote(extra_args)
    elif gpu_key == "a100":
        train_dit_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


# ---- Direct DiT (no VAE) training ----

def _run_dit_direct(extra_args):
    """Shared direct DiT training entry point."""
    import train_dit_direct as tdd
    os.chdir(VOLUME_MOUNT)
    os.makedirs("checkpoints", exist_ok=True)

    def _commit():
        print("  [volume] committing...")
        volume.commit()

    tdd.POST_CHECKPOINT_HOOK = _commit
    tdd.main(extra_args or [])
    volume.commit()
    print("Final volume commit done.")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 60 * 6,
)
def train_dit_direct_a10g(extra_args: list = None):
    _run_dit_direct(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A100",
    timeout=60 * 60 * 6,
)
def train_dit_direct_a100(extra_args: list = None):
    _run_dit_direct(extra_args)


@app.local_entrypoint()
def train_dit_direct(gpu: str = "a10g", args: str = ""):
    """Train a DiT directly on ASCII tokens (no VAE) with flow matching.

    Args:
        gpu: 'a10g' (default) or 'a100'.
        args: passthrough forwarded to train_dit_direct.py's argparse.
    """
    extra_args = args.split() if args else []
    gpu_key = gpu.lower()
    if gpu_key == "a10g":
        train_dit_direct_a10g.remote(extra_args)
    elif gpu_key == "a100":
        train_dit_direct_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


# ---- One-shot helpers for getting bulk data onto the Modal volume ----

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=4.0,
    timeout=60 * 60 * 2,
)
def _untar_remote(tar_path: str, extract_to: str = ""):
    """Untar a tarball that's already been uploaded to the Modal volume.

    tar_path: path inside the volume (e.g. "openimages/train-images.tar")
    extract_to: directory inside the volume to extract into. Default: same
                directory as the tarball.
    """
    import subprocess
    import os
    os.chdir(VOLUME_MOUNT)
    if not extract_to:
        extract_to = os.path.dirname(tar_path) or "."
    os.makedirs(extract_to, exist_ok=True)
    print(f'Untarring {tar_path} -> {extract_to} ...')
    subprocess.run(['tar', '-xf', tar_path, '-C', extract_to], check=True)
    print('Done. Committing volume...')
    volume.commit()
    print('Volume committed.')


@app.local_entrypoint()
def untar(tar_path: str, extract_to: str = ""):
    """Untar a file already uploaded to the Modal volume.

    Example:
        modal run modal_app.py::untar --tar-path openimages/train-images.tar
    """
    _untar_remote.remote(tar_path, extract_to)


# ---- Data prep on Modal: rebuild ASCII + compute CLIP embeddings ----
#
# Both run in the same container with the source images mounted from the
# volume at /data/openimages. Upload images once via `modal volume put`,
# then all data prep runs in the cloud at cloud speeds.

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=16.0,
    memory=16384,
    timeout=60 * 60 * 4,
)
def _rebuild_remote(
    annotations: str,
    masks_dir: str,
    images_dir: str,
    output: str,
    canvas_size: str,
    workers: int,
):
    import rebuild_fixed_canvas as rfc
    os.chdir(VOLUME_MOUNT)
    argv = [
        '--annotations', annotations,
        '--masks-dir', masks_dir,
        '--images-dir', images_dir,
        '--output', output,
        '--workers', str(workers),
    ]
    if canvas_size:
        argv += ['--canvas-size', canvas_size]
    rfc.main(argv)
    volume.commit()
    print('Volume committed.')


@app.local_entrypoint()
def rebuild_data(
    annotations: str = "openimages/train-annotations-object-segmentation.csv",
    masks_dir: str = "openimages/train-masks",
    images_dir: str = "openimages/train-images",
    output: str = "openimages/train_ascii_128x64_relaxed.jsonl",
    canvas_size: str = "128x64",
    workers: int = 16,
):
    """Run rebuild_fixed_canvas.py on Modal.

    Source images and annotation CSVs must already be uploaded to the
    /data/openimages/ volume. Paths are relative to /data.
    """
    _rebuild_remote.remote(annotations, masks_dir, images_dir, output, canvas_size, workers)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    cpu=8.0,
    memory=16384,
    timeout=60 * 60 * 4,
)
def _compute_clip_remote(
    jsonl: str,
    images_dir: str,
    output: str,
    batch_size: int,
    model: str,
):
    import compute_clip_embeddings as cce
    os.chdir(VOLUME_MOUNT)
    argv = [
        '--jsonl', jsonl,
        '--images-dir', images_dir,
        '--output', output,
        '--batch-size', str(batch_size),
        '--model', model,
        '--device', 'cuda',
    ]
    cce.main(argv)
    volume.commit()
    print('Volume committed.')


@app.local_entrypoint()
def compute_clip(
    jsonl: str = "openimages/train_ascii_128x64_relaxed.jsonl",
    images_dir: str = "openimages/train-images",
    output: str = "openimages/train_clip_128x64.npy",
    batch_size: int = 256,
    model: str = "openai/clip-vit-base-patch32",
):
    """Run compute_clip_embeddings.py on Modal with GPU acceleration.

    Inputs are paths inside the Modal volume (relative to /data). Source
    images must be uploaded to /data/openimages/train-images first.
    """
    _compute_clip_remote.remote(jsonl, images_dir, output, batch_size, model)


# ---- CLIP TEXT embeddings (synthetic captions) ----

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    cpu=4.0,
    memory=8192,
    timeout=60 * 60,
)
def _compute_clip_text_remote(
    jsonl: str,
    output: str,
    batch_size: int,
    model: str,
    template: str,
):
    import compute_clip_text_embeddings as ccte
    os.chdir(VOLUME_MOUNT)
    argv = [
        '--jsonl', jsonl,
        '--output', output,
        '--batch-size', str(batch_size),
        '--model', model,
        '--device', 'cuda',
        '--template', template,
    ]
    ccte.main(argv)
    volume.commit()
    print('Volume committed.')


@app.local_entrypoint()
def compute_clip_text(
    jsonl: str = "openimages/train_ascii_128x64_relaxed.jsonl",
    output: str = "openimages/train_clip_text_128x64.npy",
    batch_size: int = 512,
    model: str = "openai/clip-vit-base-patch32",
    template: str = "a photo of a {label}",
):
    """Compute CLIP TEXT embeddings from synthetic captions on Modal A10G.

    No source images needed — only the JSONL with labels. The script
    synthesizes "a photo of a {label}" for each example, runs CLIP's text
    encoder, and saves a parallel .npy aligned line-by-line with the JSONL.
    """
    _compute_clip_text_remote.remote(jsonl, output, batch_size, model, template)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 30,
)
def _inspect_remote(checkpoint: str):
    import inspect_samples
    inspect_samples.main(['--checkpoint', checkpoint, '--device', 'cuda'])


@app.local_entrypoint()
def inspect(checkpoint: str = "/data/checkpoints/v3_maskgit/step_19000.pt"):
    """Generate samples from a trained checkpoint under multiple decoding configs.

    Runs on a Modal A10G, which is ~10-20x faster than local CPU for sampling.

    Args:
        checkpoint: path inside the Modal volume. Default: latest v3 run's final ckpt.
    """
    _inspect_remote.remote(checkpoint)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 30,
)
def _probe_remote(checkpoint: str, n_samples: int):
    import os
    import probe_prefix
    os.chdir(VOLUME_MOUNT)
    probe_prefix.main([
        '--checkpoint', checkpoint,
        '--device', 'cuda',
        '--n-samples', str(n_samples),
    ])


@app.local_entrypoint()
def probe(
    checkpoint: str = "/data/checkpoints/v3_maskgit/step_19000.pt",
    n_samples: int = 3,
):
    """Partial-visibility diagnostic: does the MaskGIT model fill in missing regions?

    For each of `n_samples` real training examples, runs several visibility
    scenarios (top/bottom/left/random) and prints the infilled results.

    Args:
        checkpoint: path inside the Modal volume.
        n_samples: number of real samples to test (default 3).
    """
    _probe_remote.remote(checkpoint, n_samples)


# ---- DiT sampling ----

@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 30,
)
def _sample_dit_remote(checkpoint: str, vae_checkpoint: str, n_categories: int,
                       guidance_scale: float, seed: int):
    import os
    import sample_dit
    os.chdir(VOLUME_MOUNT)
    argv = [
        '--checkpoint', checkpoint,
        '--vae-checkpoint', vae_checkpoint,
        '--n-categories', str(n_categories),
        '--guidance-scale', str(guidance_scale),
        '--seed', str(seed),
        '--device', 'cuda',
    ]
    sample_dit.main(argv)


@app.local_entrypoint()
def sample_categories(
    checkpoint: str = "checkpoints/dit_vae_full_250m/step_60000.pt",
    vae_checkpoint: str = "checkpoints/vae_qd_full_b01/step_10000.pt",
    n_categories: int = 50,
    guidance_scale: float = 3.0,
    seed: int = 42,
):
    """Sample one ASCII art per category from a trained DiT.

    Args:
        checkpoint: DiT checkpoint path inside the Modal volume.
        vae_checkpoint: VAE checkpoint path inside the Modal volume.
        n_categories: number of random categories to sample (default 50).
        guidance_scale: CFG guidance scale (default 3.0).
        seed: random seed for category selection and sampling.
    """
    _sample_dit_remote.remote(checkpoint, vae_checkpoint, n_categories,
                              guidance_scale, seed)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 30,
)
def _sample_prompts_remote(checkpoint: str, vae_checkpoint: str, prompts: list,
                           guidance_scale: float, seed: int):
    import os
    import sample_prompts
    os.chdir(VOLUME_MOUNT)
    argv = [
        '--checkpoint', checkpoint,
        '--vae-checkpoint', vae_checkpoint,
        '--guidance-scale', str(guidance_scale),
        '--seed', str(seed),
        '--device', 'cuda',
        '--prompts', *prompts,
    ]
    sample_prompts.main(argv)


@app.local_entrypoint()
def sample_text(
    checkpoint: str = "checkpoints/dit_vae_full_250m/step_60000.pt",
    vae_checkpoint: str = "checkpoints/vae_qd_full_b01/step_10000.pt",
    guidance_scale: float = 10.0,
    seed: int = 42,
):
    """Sample ASCII art from free-text prompts.

    Prompts are hardcoded below — edit to change.
    """
    prompts = [
        # Out-of-distribution / generalization probes
        "abstract art",
        "cloudy day",
        "smile",
        "robot",
        "pirate ship",
        "cityscape",
        "old man",
        "dancing person",
        "spaceship",
        "haunted house",
        # Near-distribution but not exact categories
        "puppy",
        "sports car",
        "electric guitar",
        "Christmas tree",
        "baby bird",
    ]
    _sample_prompts_remote.remote(checkpoint, vae_checkpoint, prompts,
                                  guidance_scale, seed)
