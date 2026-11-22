# Construction Safety AI Project
# Part 2 - YOLO26 Object Detection
#
# Dataset:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1
#
# Fine-tunes YOLO26 Nano on 25 construction safety classes.
# Saves model weights, metrics, and detection examples.

from pathlib import Path
from ultralytics import YOLO
import torch

# 1. PROJECT SETTINGS

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_CONFIG = (
    PROJECT_DIR / "dataset" / "data_yolo26.yaml"
)

TEST_IMAGES_DIR = (
    PROJECT_DIR / "dataset" / "test" / "images"
)

RESULTS_DIR = (
    PROJECT_DIR / "results" / "part2_yolo"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo26n.pt"

EPOCHS = 10
IMAGE_SIZE = 640
BATCH_SIZE = 4
DEVICE = "cpu"

print("=" * 55)
print("CUDA available:", torch.cuda.is_available())
print("Dataset:", DATASET_CONFIG)
print("Model:", MODEL_NAME)

if not DATASET_CONFIG.exists():
    raise FileNotFoundError(DATASET_CONFIG)

# 2. LOAD PRETRAINED MODEL

print("\nLOADING YOLO MODEL")

model = YOLO(MODEL_NAME)

print("YOLO26 Nano loaded successfully.")

# 3. TRAIN MODEL

print("\nSTARTING YOLO26 TRAINING")

model.train(
    data=str(DATASET_CONFIG),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=0,
    project=str(RESULTS_DIR),
    name="training",
    exist_ok=True,
    plots=True,
    seed=42
)

print("\nYOLO26 TRAINING COMPLETE")

# 4. LOAD BEST TRAINED WEIGHTS

BEST_WEIGHTS = (
    RESULTS_DIR / "training" / "weights" / "best.pt"
)

if not BEST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Best model weights not found: {BEST_WEIGHTS}"
    )

trained_model = YOLO(str(BEST_WEIGHTS))

print("\nBest trained model loaded:")
print(BEST_WEIGHTS)

# 5. EVALUATE ON TEST DATASET

print("\nEVALUATING YOLO26 ON TEST DATASET")

metrics = trained_model.val(
    data=str(DATASET_CONFIG),
    split="test",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=0,
    project=str(RESULTS_DIR),
    name="test_evaluation",
    exist_ok=True,
    plots=True
)

print("\nTEST METRICS")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

print("\n" + "=" * 55)
print("YOLO26 TRAINING AND EVALUATION COMPLETE")
print("=" * 55)

print("Results saved to:", RESULTS_DIR)

