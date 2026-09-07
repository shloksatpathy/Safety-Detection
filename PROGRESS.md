# Progress

Tracking the state of the Safety Detector project: a YOLOv8n hard-hat detector taken
through ONNX and TensorRT (FP32 → FP16 → INT8) to measure runtime efficiency on
compute-limited edge devices.

## Goal

Train an FP32 PyTorch baseline, export the equivalent ONNX and TensorRT engines, then
compare them on: throughput, latency percentiles, VRAM utilization, and accuracy delta.

## Status

### Done

- **Dataset pipeline** (`training.py`) — converts the Hard Hat Workers dataset from
  Pascal VOC XML to YOLO-normalized `.txt` labels, does a seeded 80/20 train/val split,
  writes the `images/{train,val}` + `labels/{train,val}` layout and `data.yaml`.
  - Source: `data/archive/` — 5000 PNGs (~416x416) + 5000 VOC XML files.
  - Classes: `helmet` (18966 boxes), `head` (5785), `person` (751).
  - Converted dataset written to `data/yolo_dataset/` (gitignored).
- **FP32 baseline training** — 50 epochs of `yolov8n.pt` fine-tuning completed.
  Results in `runs/detect/train/` (gitignored). Final validation metrics:

  | metric | value |
  |---|---|
  | precision(B) | 0.60 |
  | recall(B) | 0.61 |
  | mAP50(B) | 0.632 |
  | mAP50-95(B) | 0.411 |

  Best checkpoint: `runs/detect/train/weights/best.pt`.
- **Environment** — `requirements.txt` updated (adds `ultralytics`, PyTorch cu126
  index). `.gitignore` covers `data/`, `runs/`, `venv/`, `*.pt`, `__pycache__/`.

### In progress / next

1. Export the FP32 baseline to ONNX and validate parity against the PyTorch model.
2. Build TensorRT engines: FP32, FP16, and INT8 (INT8 needs a calibration set from the
   val split).
3. Benchmark harness — measure throughput, latency percentiles (p50/p90/p99), VRAM
   usage, and mAP delta vs. the FP32 baseline for each runtime.
4. Collect results into a comparison table and write up findings.

## Notes

- Model weights and datasets are not tracked in git — regenerate the dataset with
  `python training.py` (it skips conversion if `data/yolo_dataset/data.yaml` exists).
- `runs/detect/train/args.yaml` records the exact training config (imgsz 416, batch 16,
  seed 42, 50 epochs).
