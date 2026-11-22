# Construction Safety AI Project
# Part 2 - YOLO26 Test Image Predictions
#
# Uses the previously trained YOLO26 model.
# Does NOT retrain the model.
#
# Dataset:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1

from pathlib import Path
from ultralytics import YOLO
from test.test_deque import fail

# 1. PROJECT DIRECTORIES

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results" / "part2_yolo"

BEST_WEIGHTS = (
    RESULTS_DIR / "training" / "weights" / "best.pt"
)

TEST_IMAGES_DIR = (
    PROJECT_DIR / "dataset" / "test" / "images"
)

# 2. LOAD TRAINED MODEL

print("=" * 55)
print("YOLO26 TEST IMAGE PREDICTIONS")
print("=" * 55)

if not BEST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Trained model not found: {BEST_WEIGHTS}"
    )
    
model = YOLO(str(BEST_WEIGHTS))

print("\nTrained YOLO26 model loaded successfully.")
print("Model:", BEST_WEIGHTS)

# 3. SELECT THREE TEST IMAGES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

test_images = sorted(
    image for image in TEST_IMAGES_DIR.iterdir()
    if image.suffix.lower() in IMAGE_EXTENSIONS
)

if len(test_images) < 3:
    raise RuntimeError(
        "At least three test images are required."
    )

selected_images = test_images[:3]

# 4. GENERATE OBJECT DETECTIONS

for index, image_path in enumerate(
    selected_images, start=1
):
    
    print(f"\nProcessing image {index}/3")
    print("Filename:", image_path.name)
    
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(RESULTS_DIR),
        name="test_predictions",
        exist_ok=True,
        verbose=True
    )
    
    print(
        "Objects detected:",
        len(results[0].boxes)
    )
    
# 5. COMPLETION

print("\n" + "=" * 55)
print("YOLO26 PREDICTIONS COMPLETE")
print("=" * 55)

print(
    "Prediction images daved to:",
    RESULTS_DIR / "test_predictions"
)
