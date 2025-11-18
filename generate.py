"""Image generation stubs for supported models."""

from __future__ import annotations

import hashlib
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def _seed_rng(seed: int) -> None:
    random.seed(seed)


def _draw_placeholder(prompt: str, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    wrapped = "\n".join(prompt[:700].split("\n"))
    draw.text((10, 10), wrapped, fill=(0, 0, 0), font=font)
    return img


def generate_image(model_name: str, prompt: str, seed: int) -> Image.Image:
    """Generate an image using the requested model.

    This implementation is a deterministic placeholder that renders the
    prompt text into an RGB canvas. In a production environment, this
    function would route to the appropriate model client (Qwen-VL, Gemini,
    GPT-4o) and return the decoded image output.
    """

    # Derive a deterministic seed from model and prompt for reproducibility
    # across runs.
    digest = hashlib.sha256(f"{model_name}-{prompt}".encode("utf-8")).hexdigest()
    derived_seed = int(digest[:8], 16) ^ seed
    _seed_rng(derived_seed)

    return _draw_placeholder(prompt)

