"""Minimal Modal entrypoints for the current QuickDraw ASCII pipeline.

Supported workflows:
  - Train / resume the VAE
  - Train / resume the latent DiT
  - Sample prompt-conditioned outputs from the pretrained model
"""

import os

import modal

APP_NAME = 'ascii-generator'
VOLUME_NAME = 'ascii-generator-data'
VOLUME_MOUNT = '/data'

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.4.0',
        'numpy',
        'Pillow',
        'transformers==4.44.0',
    )
    .add_local_python_source(
        'ascii_utils',
        'clip_utils',
        'data',
        'dit',
        'inference',
        'sample_prompts',
        'train_dit',
        'train_vae',
        'vae',
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _commit_volume():
    print('  [volume] committing...')
    volume.commit()


def _run_train_vae(extra_args: list[str] | None):
    import train_vae as train_vae_mod

    os.chdir(VOLUME_MOUNT)
    os.makedirs('checkpoints', exist_ok=True)
    train_vae_mod.POST_CHECKPOINT_HOOK = _commit_volume
    train_vae_mod.main(extra_args or [])
    volume.commit()


def _run_train_dit(extra_args: list[str] | None):
    import train_dit as train_dit_mod

    os.chdir(VOLUME_MOUNT)
    os.makedirs('checkpoints', exist_ok=True)
    train_dit_mod.POST_CHECKPOINT_HOOK = _commit_volume
    train_dit_mod.main(extra_args or [])
    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu='A10G',
    timeout=60 * 60 * 6,
)
def train_vae_a10g(extra_args: list[str] | None = None):
    _run_train_vae(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu='A100',
    timeout=60 * 60 * 6,
)
def train_vae_a100(extra_args: list[str] | None = None):
    _run_train_vae(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu='A10G',
    timeout=60 * 60 * 6,
)
def train_dit_a10g(extra_args: list[str] | None = None):
    _run_train_dit(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu='A100',
    timeout=60 * 60 * 6,
)
def train_dit_a100(extra_args: list[str] | None = None):
    _run_train_dit(extra_args)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu='A10G',
    timeout=60 * 30,
)
def sample_prompts_remote(argv: list[str]):
    import sample_prompts as sample_prompts_mod

    os.chdir(VOLUME_MOUNT)
    sample_prompts_mod.main(argv)


@app.local_entrypoint()
def train_vae(gpu: str = 'a10g', args: str = ''):
    extra_args = args.split() if args else []
    if gpu.lower() == 'a10g':
        train_vae_a10g.remote(extra_args)
    elif gpu.lower() == 'a100':
        train_vae_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


@app.local_entrypoint()
def train_dit(gpu: str = 'a10g', args: str = ''):
    extra_args = args.split() if args else []
    if gpu.lower() == 'a10g':
        train_dit_a10g.remote(extra_args)
    elif gpu.lower() == 'a100':
        train_dit_a100.remote(extra_args)
    else:
        raise ValueError(f"Unknown GPU: {gpu!r}. Use 'a10g' or 'a100'.")


@app.local_entrypoint()
def sample_prompts(
    prompts: str,
    checkpoint: str = 'local_data/models/checkpoints/dit_vae_full_250m/step_60000.pt',
    vae_checkpoint: str = 'local_data/models/checkpoints/vae_qd_full_b01/step_10000.pt',
    guidance_scale: float = 10.0,
    sample_steps: int = 50,
):
    prompt_list = [part.strip() for part in prompts.split('|') if part.strip()]
    if not prompt_list:
        raise ValueError('Provide one or more prompts separated by "|"')

    argv = [
        '--checkpoint', checkpoint,
        '--vae-checkpoint', vae_checkpoint,
        '--guidance-scale', str(guidance_scale),
        '--sample-steps', str(sample_steps),
        '--prompts',
        *prompt_list,
    ]
    sample_prompts_remote.remote(argv)
