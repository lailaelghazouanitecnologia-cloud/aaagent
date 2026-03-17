"""Sample generation quality (qualitative evaluation)."""

from __future__ import annotations

import torch

from diffusion.sampling import DiffusionSampler


def generate_samples(
    model,
    tokenizer,
    prompts: list[str],
    seq_len: int = 512,
    sampling_steps: int = 64,
    temperature: float = 1.0,
    device: str = "cuda",
) -> list[dict[str, str]]:
    """Generate text samples from the model.

    Args:
        model: The HCLM-D model.
        tokenizer: The tokenizer instance.
        prompts: List of prompt strings.
        seq_len: Total sequence length to generate.
        sampling_steps: Number of diffusion sampling steps.
        temperature: Sampling temperature.
        device: Device to run on.

    Returns:
        List of dicts with 'prompt' and 'generated' keys.
    """
    model.eval()
    sampler = DiffusionSampler(
        mask_token_id=0,
        sampling_steps=sampling_steps,
        temperature=temperature,
    )

    results = []
    for prompt in prompts:
        # Tokenize prompt
        encoding = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

        # Generate
        output_ids = sampler.sample(model, prompt_ids=prompt_ids, seq_len=seq_len, device=device)

        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        results.append({"prompt": prompt, "generated": generated_text})

    return results
