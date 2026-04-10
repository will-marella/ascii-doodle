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
    .pip_install("torch==2.4.0", "numpy")
    .add_local_python_source(
        "data", "model", "sample", "train", "inspect_samples", "probe_prefix",
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
    """Launch a training run on Modal.

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
def inspect(checkpoint: str = "/data/checkpoints/v2_humans/step_19000.pt"):
    """Generate samples from a trained checkpoint under multiple decoding configs.

    Runs on a Modal A10G, which is ~10-20x faster than local CPU for sampling.

    Args:
        checkpoint: path inside the Modal volume. Default: latest v2 run's final ckpt.
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
    checkpoint: str = "/data/checkpoints/v2_humans/step_19000.pt",
    n_samples: int = 3,
):
    """Prefix-priming diagnostic: does real context help the model produce better output?

    For each of `n_samples` real training examples, runs generation at multiple
    prefix lengths (0, 4, 8, 16, 24 rows) and prints all continuations.

    Args:
        checkpoint: path inside the Modal volume.
        n_samples: number of real samples to test (default 3).
    """
    _probe_remote.remote(checkpoint, n_samples)
