"""Orchestrate the full montage evaluation pipeline."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List

from dataset import clean_json, extract_descriptions, load_gt_images, load_json
from evaluate import evaluate_all
from generate import generate_image
from model_loader import get_available_models
from prompt_builder import build_common_prefix, build_json_prompt, build_natural_prompt


DATA_DIR = Path(os.environ.get("DATA_ROOT", "data"))
OUTPUT_DIR = Path("outputs")


def _ensure_dirs() -> None:
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)


def _list_persons() -> List[str]:
    persons: set[str] = set()
    for level in ["L", "M", "H"]:
        level_dir = DATA_DIR / "json" / level
        if not level_dir.exists():
            continue
        for file in level_dir.glob("*.json"):
            persons.add(file.stem)
    return sorted(persons)


def _append_csv(path: Path, headers: List[str], row: Dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_pipeline(seed: int = 1234) -> None:
    logging.basicConfig(level=logging.INFO)

    _ensure_dirs()
    # Load InstructBLIP VQA model once at startup as required.
    from model_loader import get_instructblip_vqa

    vqa_bundle = get_instructblip_vqa()
    checkpoint = vqa_bundle.get("checkpoint", "Salesforce/instructblip-vicuna-7b-small")
    if vqa_bundle.get("available"):
        logging.info(
            "InstructBLIP VQA loaded (%s) on device %s",
            checkpoint,
            vqa_bundle.get("device"),
        )
    else:
        logging.info(
            "InstructBLIP VQA unavailable; continuing with stubbed VQA answers so the pipeline remains runnable."
        )
    persons = _list_persons()
    available_models = get_available_models()

    metrics_headers = [
        "person",
        "level",
        "prompt_type",
        "model",
        "run",
        "arcface",
        "clip",
        "attribute_f1",
        "ssim",
        "vqa_gender",
        "vqa_face_type",
        "vqa_face_size",
        "vqa_forehead_type",
        "vqa_chin_type",
        "vqa_hair_type",
        "vqa_hair_topLength",
        "vqa_hair_part",
        "vqa_eye_type",
        "vqa_eye_size",
        "vqa_eye_distance",
        "vqa_eye_slant",
        "vqa_brow_type",
        "vqa_brow_thick",
        "vqa_nose_height",
        "vqa_nose_size",
        "vqa_nose_top",
        "vqa_mouth_thick",
        "vqa_mouth_size",
        "vqa_mouth_side",
        "vqa_total",
        "image_path",
        "heatmap_path",
        "heatmap_ref_type",
    ]

    prompts_headers = ["person", "level", "prompt_type", "model", "run", "prompt"]

    for person in persons:
        for level in ["L", "M", "H"]:
            json_path = DATA_DIR / "json" / level / f"{person}.json"
            if not json_path.exists():
                continue

            original = load_json(str(json_path))
            age = str(original.get("info", {}).get("age", ""))
            gender = str(original.get("info", {}).get("gender", ""))

            cleaned = clean_json(original)
            descriptions = extract_descriptions(original)
            common_prefix = build_common_prefix(age, gender)

            for prompt_type in ["json", "natural"]:
                for model_name in available_models:
                    for run in [1, 2]:
                        if prompt_type == "json":
                            prompt = build_json_prompt(common_prefix, cleaned)
                        else:
                            prompt = build_natural_prompt(common_prefix, descriptions)

                        image = generate_image(model_name, prompt, seed + run)
                        references = load_gt_images(person, level)
                        metrics = evaluate_all(image, references, original)

                        image_filename = f"{person}_{level}_{prompt_type}_{model_name.replace('/', '_')}_{run}.png"
                        heatmap_filename = f"{person}_{level}_{prompt_type}_{model_name.replace('/', '_')}_{run}_heatmap.png"

                        image_path = OUTPUT_DIR / "images" / image_filename
                        heatmap_path = OUTPUT_DIR / "heatmaps" / heatmap_filename

                        image.save(image_path)
                        metrics.get("heatmap").save(heatmap_path)  # type: ignore[arg-type]

                        metric_row = {
                            "person": person,
                            "level": level,
                            "prompt_type": prompt_type,
                            "model": model_name,
                            "run": run,
                            "arcface": metrics["arcface"],
                            "clip": metrics["clip"],
                            "attribute_f1": metrics["attribute_f1"],
                            "ssim": metrics["ssim"],
                            "vqa_gender": metrics.get("vqa_gender"),
                            "vqa_face_type": metrics.get("vqa_face_type"),
                            "vqa_face_size": metrics.get("vqa_face_size"),
                            "vqa_forehead_type": metrics.get("vqa_forehead_type"),
                            "vqa_chin_type": metrics.get("vqa_chin_type"),
                            "vqa_hair_type": metrics.get("vqa_hair_type"),
                            "vqa_hair_topLength": metrics.get("vqa_hair_topLength"),
                            "vqa_hair_part": metrics.get("vqa_hair_part"),
                            "vqa_eye_type": metrics.get("vqa_eye_type"),
                            "vqa_eye_size": metrics.get("vqa_eye_size"),
                            "vqa_eye_distance": metrics.get("vqa_eye_distance"),
                            "vqa_eye_slant": metrics.get("vqa_eye_slant"),
                            "vqa_brow_type": metrics.get("vqa_brow_type"),
                            "vqa_brow_thick": metrics.get("vqa_brow_thick"),
                            "vqa_nose_height": metrics.get("vqa_nose_height"),
                            "vqa_nose_size": metrics.get("vqa_nose_size"),
                            "vqa_nose_top": metrics.get("vqa_nose_top"),
                            "vqa_mouth_thick": metrics.get("vqa_mouth_thick"),
                            "vqa_mouth_size": metrics.get("vqa_mouth_size"),
                            "vqa_mouth_side": metrics.get("vqa_mouth_side"),
                            "vqa_total": metrics.get("vqa_total"),
                            "image_path": str(image_path),
                            "heatmap_path": str(heatmap_path),
                            "heatmap_ref_type": metrics.get("heatmap_ref_type"),
                        }

                        _append_csv(OUTPUT_DIR / "metrics.csv", metrics_headers, metric_row)

                        prompt_row = {
                            "person": person,
                            "level": level,
                            "prompt_type": prompt_type,
                            "model": model_name,
                            "run": run,
                            "prompt": prompt,
                        }
                        _append_csv(OUTPUT_DIR / "prompts.csv", prompts_headers, prompt_row)


if __name__ == "__main__":
    run_pipeline()

