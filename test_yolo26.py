# Construction Safety AI Project
# Part 2 - Verify YOLO26 Installation

from pathlib import Path
import ultralytics
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

print("=" * 50)
print("YOLO26 INSTALLATION TEST")
print("=" * 50)

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Load the pretrained YOLO26 Nano model
# Model weights download automatically the first time

model = YOLO("yolo26n.pt")

print("\nYOLO26 model loaded successfully.")

# Find one image from the original Roboflow dataset

images_dir = PROJECT_DIR / "dataset" / "test" / "images"

image_files = sorted(
    path for path in images_dir.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_files:
    raise RuntimeError("No test images found.")

test_image = image_files[0]

print("\nTesting image:", test_image.name)

# Run object detection on one image.
# The pretrained model has not yet been fine-tuned
# on our construction safety dataset.

results = model.predict(
    source=str(test_image),
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    project=str(PROJECT_DIR / "results" / "part2_yolo26"),
    name="installation_test",
    exist_ok=True
)

print("\nDetections:", len(results[0].boxes))
print("YOLO26 installation test complete.")
