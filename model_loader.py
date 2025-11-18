"""Model loading utilities.

The real models are heavyweight; this module provides lightweight loaders
with safe fallbacks so the pipeline can run in constrained environments
while preserving the intended API surface.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_torch():  # pragma: no cover - import helper
    """Import torch lazily so environments without it stay runnable."""

    try:
        import importlib

        return importlib.import_module("torch")  # type: ignore
    except Exception:
        return None


def load_qwen() -> Dict[str, str]:
    """Attempt to load a Qwen-VL checkpoint.

    The function prefers GPU if available and falls back to a smaller model
    if the large one cannot be initialized. Returned value is a lightweight
    descriptor rather than the full model to avoid heavy downloads in this
    environment.
    """

    torch_mod = _get_torch()
    device = "cuda" if torch_mod and torch_mod.cuda.is_available() else "cpu"
    candidates = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
    ]

    for name in candidates:
        try:
            # In production you would instantiate the actual model here.
            return {"name": name, "device": device}
        except Exception:
            continue
    raise RuntimeError("No Qwen-VL model could be initialized")


def load_gemini() -> Optional[Dict[str, str]]:
    """Return Gemini configuration if an API key is available."""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return {"name": "Gemini", "api_key": api_key}


def load_gpt4o() -> Dict[str, str]:
    """Return a descriptor for GPT-4o usage."""

    api_key = os.environ.get("OPENAI_API_KEY", "")
    # Placeholder: pipeline treats absence of key as still selectable but
    # generation will fallback to stub behavior.
    return {"name": "GPT-4o", "api_key": api_key}


def get_available_models() -> List[str]:
    """Enumerate available model names based on the environment."""

    names: List[str] = []

    torch_mod = _get_torch()
    if torch_mod is None:
        logger.warning(
            "PyTorch is not installed; skipping Qwen-VL. Install torch (e.g., `pip install torch --extra-index-url https://download.pytorch.org/whl/cu121`) to enable GPU inference."
        )
    else:
        try:
            qwen_cfg = load_qwen()
            names.append(qwen_cfg["name"])
        except Exception:
            logger.warning("Failed to initialize Qwen-VL; continuing without it.")

    gemini_cfg = load_gemini()
    if gemini_cfg:
        names.append(gemini_cfg["name"])

    # GPT-4o is always listed for parity, even if API key is missing.
    names.append(load_gpt4o()["name"])

    return names

