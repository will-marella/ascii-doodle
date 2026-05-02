"""Interactive ASCII art generation via a deployed DiT on Modal.

Spins up a Modal container with the model loaded, then accepts text
prompts in a loop locally. Each prompt is sent to the GPU container
for CLIP encoding + DiT sampling + VAE decoding.

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
        "data", "sample", "vae", "dit", "train_dit",
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
        checkpoint: str = "checkpoints/dit_vae_full_250m/step_60000.pt",
        vae_checkpoint: str = "checkpoints/vae_qd_full_b01/step_10000.pt",
    ):
        self.checkpoint = checkpoint
        self.vae_checkpoint = vae_checkpoint

    @modal.enter()
    def load_model(self):
        import torch
        from dit import DiT
        from train_dit import load_vae
        from sample import text_to_clip_embeddings

        os.chdir(VOLUME_MOUNT)
        device = "cuda"

        ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        clip_dim = ckpt.get("clip_dim", 0)

        vae, grid_h, grid_w = load_vae(self.vae_checkpoint, device)
        self.vae = vae
        self.grid_h = grid_h
        self.grid_w = grid_w

        dit = DiT(
            latent_channels=vae.latent_channels,
            latent_h=vae.latent_h,
            latent_w=vae.latent_w,
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            ffn_mult=config.get("ffn_mult", 4),
            clip_dim=clip_dim,
        ).to(device)
        dit.load_state_dict(ckpt["model"])
        dit.eval()
        self.dit = dit
        self.device = device
        self.latent_mean = ckpt["latent_mean"].to(device)
        self.latent_std = ckpt["latent_std"].to(device)

        print(f"Model loaded: {dit.num_params():,} params, step {ckpt['step']}")

    @modal.method()
    def generate(self, prompt: str, guidance_scale: float = 10.0,
                 sample_steps: int = 50) -> str:
        import torch
        from sample import decode, text_to_clip_embeddings
        from train_dit import sample_flow, unnormalize_latent

        clip_emb = text_to_clip_embeddings(
            [f"a drawing of a {prompt}"],
            device=torch.device(self.device),
        )

        z_norm = sample_flow(
            self.dit, 1, n_steps=sample_steps,
            clip_emb=clip_emb, guidance_scale=guidance_scale,
        )
        z = unnormalize_latent(z_norm, self.latent_mean, self.latent_std)
        with torch.no_grad():
            logits = self.vae.decode(z)
            tokens = logits.argmax(dim=1)
        flat = tokens[0].reshape(-1).tolist()
        return decode(flat, grid_w=self.grid_w)


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
