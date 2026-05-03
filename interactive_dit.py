"""Interactive ASCII art generation on Modal using the current 130m model.

Usage:
    modal run interactive_dit.py
    modal run interactive_dit.py --guidance-scale 10.0
"""

import os

import modal

APP_NAME = "ascii-generator-interactive"
VOLUME_NAME = "ascii-generator-data"
VOLUME_MOUNT = "/data"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "transformers==4.44.0",
    )
    .add_local_python_source(
        "ascii_utils", "clip_utils", "data", "dit", "inference", "train_dit", "vae",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME)


@app.cls(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=60 * 30,
)
class Generator:
    def __init__(
        self,
        checkpoint: str = "local_data/models/checkpoints/dit_vae_full_250m/step_60000.pt",
        vae_checkpoint: str = "local_data/models/checkpoints/vae_qd_full_b01/step_10000.pt",
    ):
        self.checkpoint = checkpoint
        self.vae_checkpoint = vae_checkpoint

    @modal.enter()
    def load_model(self):
        from inference import load_pipeline

        os.chdir(VOLUME_MOUNT)
        self.pipeline = load_pipeline(
            checkpoint=self.checkpoint,
            vae_checkpoint=self.vae_checkpoint,
            device='cuda',
        )
        print(f"Model loaded: {self.pipeline.dit.num_params():,} params")

    @modal.method()
    def generate(self, prompt: str, guidance_scale: float = 10.0,
                 sample_steps: int = 50) -> str:
        from inference import generate_ascii

        return generate_ascii(
            self.pipeline,
            prompt,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
        )


@app.local_entrypoint()
def main(guidance_scale: float = 10.0, sample_steps: int = 50):
    gen = Generator()

    print("\nASCII Art Generator")
    print(f"guidance_scale={guidance_scale}, sample_steps={sample_steps}")
    print('Type a prompt and press Enter. Type "quit" to exit.\n')

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = gen.generate.remote(prompt, guidance_scale, sample_steps)
        print()
        for line in result.split("\n"):
            print(f"  {line}")
        print()
