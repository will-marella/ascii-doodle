"""CLIP helpers for prompt-conditioned QuickDraw ASCII generation."""

import torch


_clip_text_cache = {}


def text_to_clip_embeddings(
    prompts: list[str],
    device: torch.device | None = None,
    model_name: str = 'openai/clip-vit-base-patch32',
) -> torch.Tensor:
    """Encode text prompts via CLIP's text encoder."""
    cache_key = (model_name, str(device))
    if cache_key not in _clip_text_cache:
        from transformers import CLIPModel, CLIPTokenizer

        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        _clip_text_cache[cache_key] = (model, tokenizer)

    model, tokenizer = _clip_text_cache[cache_key]
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.float()
