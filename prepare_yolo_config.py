# Construction Safety AI Project
# Part 2 - Prepare YOLO26 Dataset Configuration
#
# Dataset source:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1

from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

SOURCE_YAML = DATASET_DIR / "data.yaml"
OUTPUT_YAML = DATASET_DIR / "data_yolo26.yaml"

print("=" * 50)
print("PREPARING YOLO26 CONFIGURATION")
print("=" * 50)

# Read original Roboflow configuration

with open(SOURCE_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
    
# Verify required folders exist

for split in ["train", "valid", "test"]:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)

# Preserve all original class definitions
names = config["names"]

# Normalize class names into an ordered list
if isinstance(names, dict):
    names = [
        name for _, name in sorted(
            names.items(),
            key=lambda item: int(item[0])
        )
    ]

# Use absolute dataset root and relative image folders
new_config = {
    "path": str(DATASET_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(names),
    "names": names
}

# Write new configuration
with open(OUTPUT_YAML, "w", encoding="utf-8") as file:
    yaml.safe_dump(
        new_config,
        file,
        sort_keys=False,
        allow_unicode=True
    )

print("\nClasses:", len(names))
print("Training images:", DATASET_DIR / "train" / "images")
print("Validation images:", DATASET_DIR / "valid" / "images")
print("Testing images:", DATASET_DIR / "test" / "images")

print("\nYOLO26 configuration saved:")
print(OUTPUT_YAML)

print("\nConfiguration preparation complete.")
