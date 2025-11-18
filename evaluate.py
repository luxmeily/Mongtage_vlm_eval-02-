"""Evaluation stubs and helpers for montage experiments."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter


def preprocess_face(image: Image.Image) -> Image.Image:
    """Basic preprocessing placeholder: resize and blur to stabilize metrics."""

    return image.convert("RGB").resize((256, 256)).filter(ImageFilter.SMOOTH)


def _pseudo_score(name: str, image: Image.Image) -> float:
    """Deterministic pseudo score based on image content hash."""

    data = image.tobytes()
    digest = hashlib.sha256(name.encode("utf-8") + data).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def arcface_similarity(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("arcface", preprocess_face(generated))


def vqa_accuracy(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("vqa", preprocess_face(generated))


def clip_score(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("clip", preprocess_face(generated))


def attribute_f1(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("attribute", preprocess_face(generated))


def ssim_heatmap(generated: Image.Image, reference: Image.Image) -> Tuple[Image.Image, float]:
    """Compute a simple per-pixel difference heatmap as an SSIM placeholder."""

    gen = preprocess_face(generated)
    ref = preprocess_face(reference) if reference else gen

    diff = ImageChops.difference(gen, ref)
    heatmap = diff.convert("L").filter(ImageFilter.GaussianBlur(radius=2))

    # Normalize pseudo-SSIM between 0 and 1 using inverted mean difference.
    arr = np.asarray(heatmap, dtype=np.float32)
    score = 1.0 - min(arr.mean() / 255.0, 1.0)
    return heatmap.convert("RGB"), float(score)


def evaluate_all(generated: Image.Image, references: Dict[str, Image.Image]) -> Dict[str, float]:
    """Compute all metrics with stubs and return a flat dictionary."""

    ref_image = references.get("montage") or references.get("sketch") or generated

    heatmap, ssim_score = ssim_heatmap(generated, ref_image)

    return {
        "arcface": arcface_similarity(generated, ref_image),
        "vqa": vqa_accuracy(generated, ref_image),
        "clip": clip_score(generated, ref_image),
        "attribute_f1": attribute_f1(generated, ref_image),
        "ssim": ssim_score,
        "heatmap": heatmap,
    }

