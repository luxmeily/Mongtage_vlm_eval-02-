"""Prompt construction helpers for montage generation."""

from __future__ import annotations

import json
from typing import List


def build_common_prefix(age: str, gender: str) -> str:
    gender_kr = {"M": "남성", "F": "여성"}.get(gender, gender)
    return (
        f"{age}대 한국인 {gender_kr}의 몽타주를 그려라.\n"
        "설명에 포함되지 않은 세부 특징은 생성하지 마라."
    )


def build_json_prompt(common_prefix: str, cleaned_json: dict) -> str:
    body = json.dumps(cleaned_json, ensure_ascii=False, indent=2)
    return (
        f"{common_prefix}\n\n"
        "아래 JSON 설명만 기반으로 묘사된 인물을 그려라.\n\n"
        f"{body}"
    )


def build_natural_prompt(common_prefix: str, description_list: List[str]) -> str:
    natural_block = "\n".join(description_list)
    return (
        f"{common_prefix}\n\n"
        "아래 자연어 설명만 기반으로 묘사된 인물을 그려라.\n\n"
        f"{natural_block}"
    )

