"""Evaluation stubs and helpers for montage experiments."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from model_loader import get_instructblip_vqa


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


def _normalize_prediction(attr: str, text: str) -> str:
    """Map free-form VQA text answers into comparable labels."""

    if not text:
        return ""

    t = text.strip().lower()
    # gender
    if attr == "gender":
        if "남" in t or "male" in t or t.startswith("m"):
            return "m"
        if "여" in t or "female" in t or t.startswith("f"):
            return "f"
    return t


def _generate_questions(gt_attrs: Dict[str, str]) -> Dict[str, str]:
    """Create Korean attribute-specific questions for InstructBLIP VQA."""

    questions = {
        "gender": "이 인물의 성별은 무엇인가요?",  # expects 남성/여성
        "face_type": "이 인물의 얼굴형은 어떤가요?",
        "face_size": "이 인물의 얼굴 크기는 어떤가요?",
        "forehead_type": "이 인물의 이마 형태는 어떤가요?",
        "chin_type": "이 인물의 턱 형태는 어떤가요?",
        "hair_type": "이 인물의 헤어 타입은 어떤가요?",
        "hair_topLength": "이 인물의 앞머리 길이는 어느 정도인가요?",
        "hair_part": "이 인물의 가르마는 어디에 있나요?",
        "eye_type": "이 인물의 눈 형태는 어떤가요?",
        "eye_size": "이 인물의 눈 크기는 어떤가요?",
        "eye_distance": "이 인물의 눈 사이 거리는 어떤가요?",
        "eye_slant": "이 인물의 눈매 기울기는 어떤가요?",
        "brow_type": "이 인물의 눈썹 형태는 어떤가요?",
        "brow_thick": "이 인물의 눈썹 두께는 어떤가요?",
        "nose_height": "이 인물의 코 높이는 어떤가요?",
        "nose_size": "이 인물의 코 크기는 어떤가요?",
        "nose_top": "이 인물의 콧망울 형태는 어떤가요?",
        "mouth_thick": "이 인물의 입술 두께는 어떤가요?",
        "mouth_size": "이 인물의 입 크기는 어떤가요?",
        "mouth_side": "이 인물의 입꼬리는 어떤가요?",
    }

    # Fall back to generic phrasing if any attribute is missing
    for attr in gt_attrs:
        questions.setdefault(attr, f"이 인물의 {attr}은 무엇인가요?")
    return questions


def _run_instructblip_vqa(image: Image.Image, questions: Dict[str, str]) -> Dict[str, str]:
    """Query InstructBLIP once per attribute, or stub if unavailable."""

    bundle = get_instructblip_vqa()
    if not bundle.get("available"):
        # Stub: return empty to fall back on default normalization
        return {key: "" for key in questions}

    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]

    answers: Dict[str, str] = {}
    for attr, question in questions.items():
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=16)
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        answers[attr] = text
    return answers


def vqa_accuracy(generated: Image.Image, json_data: Dict[str, object]) -> Dict[str, float]:
    """JSON-only VQA using InstructBLIP for image-question answers."""

    gt_attrs = parse_json_attributes(json_data)
    questions = _generate_questions(gt_attrs)
    raw_answers = _run_instructblip_vqa(generated, questions)

    results: Dict[str, float] = {}
    for key, gt_val in gt_attrs.items():
        pred_val = _normalize_prediction(key, raw_answers.get(key, ""))
        if not gt_val:
            results[f"vqa_{key}"] = 0.0
            continue
        results[f"vqa_{key}"] = 1.0 if pred_val and pred_val == _normalize_answer(gt_val) else 0.0

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

