# Montage VLM Evaluation (stub pipeline)

## Dataset root
Set `DATA_ROOT` to the base directory that contains `images/` and `json/` as laid out in your Windows dataset (e.g., `C:\\Users\\...\\TL`). By default the code expects `data/` under the repo.

```bash
export DATA_ROOT="C:/Users/206/Desktop/050.페르소나 기반의 가상 인물 몽타주 데이터/01.데이터/1.Training/라벨링데이터_221205_add/TL"
```

## Installation
Install the lightweight dependencies first:

```bash
pip install -r requirements.txt
```

To enable the VQA path with InstructBLIP you also need PyTorch; pick the wheel for your platform (examples are in `requirements.txt`).

## Model selection
- Qwen-VL (requires PyTorch) and Gemini (needs `GEMINI_API_KEY`) are discovered automatically.
- GPT-4o is **disabled by default** per request; set `ENABLE_GPT4O=1` if you later want to include it.

## Running
```bash
python main.py
```
Generated images, heatmaps, prompts, and metrics are written under `outputs/`.
