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
    """Recursively remove description fields according to the spec.

    Rules (finalized):
    - Drop description.<part>.description for face/hairstyle/eyebrows/eyes/
      nose/mouth/neck/wrinkle.
    - Drop description.feature.description, description.feature.mustache,
      and description.feature.sideburns.
    - Preserve every other field, including info.*, sketch_info, and
      org_sketch_info.
    """

    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            # Handle the feature sub-structure separately.
            if key == "feature" and isinstance(value, dict):
                feature_cleaned = {
                    k: v
                    for k, v in value.items()
                    if k not in {"description", "mustache", "sideburns"}
                }
                cleaned[key] = _remove_nested_description(feature_cleaned)
                continue

            # Remove description field inside known parts.
            if key == "description" and isinstance(value, str):
                # Skip any plain string description fields.
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


def load_gt_images(person_id: str) -> Dict[str, Optional[Image.Image]]:
    """Load ground-truth montage and sketch images if available."""

    montage_path = os.path.join("data", "images", "montage", f"{person_id}.png")
    sketch_path = os.path.join("data", "images", "org_sketch", f"{person_id}.png")
    return {
        "montage": _find_image(montage_path),
        "sketch": _find_image(sketch_path),
    }

