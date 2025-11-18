"""Dataset utilities for loading and preparing AI Hub montage inputs.

This module focuses on two core responsibilities:
1. Loading JSON metadata for each synthetic person.
2. Producing prompt-ready artifacts by removing or extracting specific
   description fields according to the project specification.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON dictionary.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _remove_nested_description(obj: Any) -> Any:
    """Recursively remove description-style natural language fields.

    The finalized spec requires that JSON prompts keep the structural keys
    intact while stripping natural-language description fields. Specifically:

    - Remove any ``description`` string values (e.g., ``description.face.description``).
    - Within ``feature`` blocks, also drop ``mustache`` and ``sideburns`` keys.
    - Preserve all other keys such as ``info.*``, ``sketch_info``, and
      ``org_sketch_info``.
    """

    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "feature" and isinstance(value, dict):
                feature_cleaned = {
                    k: v
                    for k, v in value.items()
                    if k not in {"description", "mustache", "sideburns"}
                }
                cleaned[key] = _remove_nested_description(feature_cleaned)
                continue

            if key == "description" and isinstance(value, str):
                continue

            cleaned[key] = _remove_nested_description(value)
        return cleaned

    if isinstance(obj, list):
        return [_remove_nested_description(item) for item in obj]

    return obj


def clean_json(original: Dict[str, Any]) -> Dict[str, Any]:
    """Return a cleaned JSON object for JSON-style prompting.

    Only description text fields are removed; all other structural
    attributes remain intact.
    """

    # Work on a deep copy-like transformation to avoid mutating input.
    return _remove_nested_description(original)


def extract_descriptions(original: Dict[str, Any]) -> List[str]:
    """Extract natural-language description strings.

    Returns the ordered list of description.*.description fields excluding
    feature.description.
    """

    description_paths = [
        ["description", "face", "description"],
        ["description", "hairstyle", "description"],
        ["description", "eyebrows", "description"],
        ["description", "eyes", "description"],
        ["description", "nose", "description"],
        ["description", "mouth", "description"],
        ["description", "neck", "description"],
        ["description", "wrinkle", "description"],
    ]

    collected: List[str] = []

    for path in description_paths:
        cursor: Any = original
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, str) and cursor.strip():
            collected.append(cursor.strip())
    return collected


def _find_image(path: str) -> Optional[Image.Image]:
    if not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _try_load_with_extensions(base_path: str, exts: List[str]) -> Optional[Image.Image]:
    """Load an image by trying multiple extensions in order."""

    for ext in exts:
        candidate = f"{base_path}{ext}"
        img = _find_image(candidate)
        if img is not None:
            return img
    return None


def load_gt_images(person_id: str, level: Optional[str] = None) -> Dict[str, Optional[Image.Image]]:
    """Load ground-truth montage and sketch images if available.

    Args:
        person_id: Identifier matching the JSON/GT filenames.
        level: Optional level (L/M/H) to search level-specific sketch images.
    """

    montage_base = os.path.join("data", "images", "montage", f"{person_id}")
    org_sketch_base = os.path.join("data", "images", "org_sketch", f"{person_id}")
    level_sketch_base = (
        os.path.join("data", "images", "sketch", level, f"{person_id}")
        if level
        else None
    )

    sketch_image = None
    if level_sketch_base:
        sketch_image = _try_load_with_extensions(level_sketch_base, [".png", ".jpg", ".jpeg"])
    sketch_image = sketch_image or _try_load_with_extensions(org_sketch_base, [".png", ".jpg", ".jpeg"])

    return {
        "montage": _try_load_with_extensions(montage_base, [".png", ".jpg", ".jpeg"]),
        "sketch": sketch_image,
    }

