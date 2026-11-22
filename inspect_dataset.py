# Construction Safety AI Project
# Part 1 - Dataset Inspection
#
# Dataset source:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1
#
# This program checks the dataset folders, reads the
# class names, and counts the images and annotations.

from pathlib import Path
import yaml

# Locate the dataset relative to this Python file
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Dataset configuration file
YAML_FILE = DATASET_DIR / "data.yaml"

print("CONSTRUCTION SAFETY DATASET INSPECTION")
print("=" * 45)

print("Dataset locations:", DATASET_DIR)

if not YAML_FILE.exists():
    print("\nERROR: data.yaml was not found.")
    print("Check that the dataset was extracted correctly.")
    raise SystemExit(1)

# Read the YOLO dataset configuration
with open(YAML_FILE, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("\nDATASET CLASSES")
print("-" * 45)

classes = config.get("names", [])

# Support list and dictionary class definitions
if isinstance(classes, dict):
    class_items = sorted(
        classes.items, key=lambda item: int(item[0])
    )
else:
    class_items = enumerate(classes)
    
for class_id, class_name in class_items:
    print(f"{class_id}: {class_name}")

print("\nDATASET FOLDERS")
print("-" * 45)

# Check common Roboflow split names
for split in ["train", "valid", "test"]:
    
    split_dir = DATASET_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    if not images_dir.exists():
        print(f"{split}: Folder not found")
        continue
    
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    images = [
        path for path in images_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]
    
    labels = (
        list(labels_dir.glob("*.txt"))
        if labels_dir.exists() else []
    )
    
    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Label files:", len(labels))
    
print("\nDataset inspection complete.")
