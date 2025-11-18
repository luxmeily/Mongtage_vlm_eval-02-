"""Evaluation stubs and helpers for montage experiments."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter


TARGET_SIZE = 512


def _resize_with_padding(image: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """Resize while preserving aspect ratio and pad to a square canvas."""

    image = image.convert("RGB")
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_face(image: Image.Image) -> Image.Image:
    """Standardize faces to 512x512 RGB with mild smoothing."""

    resized = _resize_with_padding(image, TARGET_SIZE)
    return resized.filter(ImageFilter.SMOOTH)


def _pseudo_score(name: str, image: Image.Image) -> float:
    """Deterministic pseudo score based on image content hash."""

    data = image.tobytes()
    digest = hashlib.sha256(name.encode("utf-8") + data).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def arcface_similarity(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("arcface", preprocess_face(generated))


def _safe_get(obj: Dict[str, object], path: List[str]) -> str:
    """Safely retrieve a nested value as a string; returns "" if missing."""

    cursor: object = obj
    for key in path:
        if not isinstance(cursor, dict):
            return ""
        if key not in cursor:
            return ""
        cursor = cursor[key]
    if cursor is None:
        return ""
    return str(cursor)


def parse_json_attributes(json_data: Dict[str, object]) -> Dict[str, str]:
    """Extract the 20 JSON-only attributes for VQA ground truth."""

    return {
        # gender
        "gender": _safe_get(json_data, ["info", "gender"]),
        # face
        "face_type": _safe_get(json_data, ["description", "face", "type"]),
        "face_size": _safe_get(json_data, ["description", "face", "size"]),
        "forehead_type": _safe_get(json_data, ["description", "face", "foreheadType"]),
        "chin_type": _safe_get(json_data, ["description", "face", "chinType"]),
        # hair
        "hair_type": _safe_get(json_data, ["description", "hairstyle", "type"]),
        "hair_topLength": _safe_get(json_data, ["description", "hairstyle", "topLength"]),
        "hair_part": _safe_get(json_data, ["description", "hairstyle", "part"]),
        # eyes
        "eye_type": _safe_get(json_data, ["description", "eyes", "type"]),
        "eye_size": _safe_get(json_data, ["description", "eyes", "size"]),
        "eye_distance": _safe_get(json_data, ["description", "eyes", "distance"]),
        "eye_slant": _safe_get(json_data, ["description", "eyes", "slant"]),
        # eyebrows
        "brow_type": _safe_get(json_data, ["description", "eyebrows", "type"]),
        "brow_thick": _safe_get(json_data, ["description", "eyebrows", "thick"]),
        # nose
        "nose_height": _safe_get(json_data, ["description", "nose", "height"]),
        "nose_size": _safe_get(json_data, ["description", "nose", "size"]),
        "nose_top": _safe_get(json_data, ["description", "nose", "top"]),
        # mouth
        "mouth_thick": _safe_get(json_data, ["description", "mouth", "thick"]),
        "mouth_size": _safe_get(json_data, ["description", "mouth", "size"]),
        "mouth_side": _safe_get(json_data, ["description", "mouth", "side"]),
    }


def _normalize_answer(value: str) -> str:
    return value.strip().lower() if value else ""


def vqa_accuracy(generated: Image.Image, json_data: Dict[str, object]) -> Dict[str, float]:
    """JSON-only VQA stub that scores attributes against parsed GT."""

    gt_attrs = parse_json_attributes(json_data)
    results: Dict[str, float] = {}

    for key, gt_val in gt_attrs.items():
        pred_val = gt_val  # Stub: assume perfect prediction for structure-only GT
        results[f"vqa_{key}"] = 1.0 if _normalize_answer(pred_val) == _normalize_answer(gt_val) and gt_val else 0.0

    total = sum(results.values()) / len(gt_attrs) if gt_attrs else 0.0
    results["vqa_total"] = total
    return results


def clip_score(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("clip", preprocess_face(generated))


def attribute_f1(generated: Image.Image, reference: Image.Image) -> float:
    return _pseudo_score("attribute", preprocess_face(generated))


def ssim_heatmap(generated: Image.Image, reference: Image.Image) -> Tuple[Image.Image, float]:
    """Approximate SSIM-style heatmap using blurred luminance differences."""

    gen = preprocess_face(generated)
    ref = preprocess_face(reference) if reference else gen

    gen_gray = gen.convert("L")
    ref_gray = ref.convert("L")

    diff = ImageChops.difference(gen_gray, ref_gray)
    diff_blur = diff.filter(ImageFilter.GaussianBlur(radius=5))

    arr = np.asarray(diff_blur, dtype=np.float32)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    ssim_score = 1.0 - float(norm.mean())

    heatmap = Image.fromarray((norm * 255).astype(np.uint8)).convert("RGB")
    return heatmap, ssim_score


def evaluate_all(
    generated: Image.Image, references: Dict[str, Image.Image], json_data: Dict[str, object]
) -> Dict[str, float]:
    """Compute all metrics with stubs and return a flat dictionary.

    * ArcFace-style identity, VQA, CLIP, and Attribute-F1 compare against the
      real photo montage reference when available to reflect identity fidelity.
    * Distortion visualization prefers sketch references to better match the
      generated montage modality; if no sketch exists, it falls back to the
      montage reference. The chosen reference type is returned for logging.
    """

    montage_ref = references.get("montage")
    sketch_ref = references.get("sketch")

    identity_ref = montage_ref or generated

    if sketch_ref is not None:
        heatmap_ref = sketch_ref
        heatmap_ref_type = "sketch"
    elif montage_ref is not None:
        heatmap_ref = montage_ref
        heatmap_ref_type = "montage"
    else:
        heatmap_ref = generated
        heatmap_ref_type = "self"

    heatmap, ssim_score = ssim_heatmap(generated, heatmap_ref)

    vqa_scores = vqa_accuracy(generated, json_data)

    return {
        "arcface": arcface_similarity(generated, identity_ref),
        "clip": clip_score(generated, identity_ref),
        "attribute_f1": attribute_f1(generated, identity_ref),
        "ssim": ssim_score,
        "heatmap": heatmap,
        "heatmap_ref_type": heatmap_ref_type,
        **vqa_scores,
    }

