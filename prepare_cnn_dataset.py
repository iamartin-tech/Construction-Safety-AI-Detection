# Construction Safety AI Project
# Part 1 - Prepare CNN Dataset
#
# Dataset source:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1
#
# Creates image-level labels using YOLO annotations.
# 0 = Hardhat Only
# 1 = No-Hardhat Present
#
# Original YOLO images and labels are not modified.

from pathlib import Path
from collections import Counter
import csv
import random
import shutil


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "cnn_dataset"

HARDHAT_ID = 2
NO_HARDHAT_ID = 5

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def classify_image(label_file):
    """
    Assign an image-level classification using
    the YOLO annotation file.

    0 = Hardhat Only
    1 = No-Hardhat Present
    """

    if not label_file.exists():
        return None

    class_ids = set()

    # Read the YOLO annotation file
    with open(label_file, "r", encoding="utf-8") as file:

        for line in file:

            # Separate the annotation into values
            parts = line.strip().split()

            # Skip empty lines
            if not parts:
                continue

            # Extract the class ID
            class_id = int(parts[0])

            # Store the detected class
            class_ids.add(class_id)

    # Class 5 = NO-Hardhat
    if NO_HARDHAT_ID in class_ids:
        return 1

    # Class 2 = Hardhat
    if HARDHAT_ID in class_ids:
        return 0

    # Image does not contain either class
    return None


def collect_images(source_splits):
    """Collect labeled iimages from selected source splits."""
    
    records = []
    
    for split in source_splits:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"
        
        if not images_dir.exists():
            print("Missing folder:", images_dir)
            continue
        
        for image_path in sorted(images_dir.iterdir()):
            
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            label_path = labels_dir / (
                image_path.stem + ".txt"
            )
            
            class_label = classify_image(label_path)
            
            if class_label is None:
                continue
            
            records.append(
                (split, image_path, class_label)
                )
    
    return records

def save_records(records, destination):
    """Copy images into classification folders."""
    
    class_names = {
        0: "Hardhat_Only",
        1: "No_Hardhat_Present"
    }
    
    saved = []
    
    for source_split, image_path, label in records:
        
        class_name = class_names[label]
        
        target_dir = OUTPUT_DIR / destination / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Include source split to avoid filename collisions
        new_name = source_split + "_" + image_path.name
        target_path = target_dir / new_name
        
        shutil.copy2(image_path, target_path)
        
        saved.append([
            destination,
            class_name,
            str(image_path),
            str(target_path)
        ])
    
    return saved

def main():
    
    print("PREPARING CNN DATASET")
    print("=" * 45)
    
    # Combine the original train and validation images
    development = collect_images(["train", "valid"])
    
    # Preserve original test images for final evaluation
    test_records = collect_images(["test"])
    
    print("Eligible development images:", len(development))
    print("Eligible test images:", len(test_records))
    
    # Stratified 80/20 split of development images
    train_records = []
    valid_records = []
    
    for class_label in [0, 1]:
        
        group = [
            record for record in development
            if record[2] == class_label
        ]
        
        random.shuffle(group)
        
        split_index = int(len(group) * 0.80)
        
        train_records.extend(group[:split_index])
        valid_records.extend(group[split_index:])
        
        # Rebuild only the generated CNN dataset
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        
        all_saved = []
        
        for name, records in [
            ("train", train_records),
            ("valid", valid_records),
            ("test", test_records)
        ]:
        
            saved = save_records(records, name)
            all_saved.extend(saved)
            
            counts = Counter(record[2] for record in records)
            
            print(f"\n{name.upper()}")
            print("Total:", len(records))
            print("Hardhat Only:", counts[0])
            print("No-Hardhat Present:", counts[1])
        
        # Save an audit trail for the generated dataset
        csv_path = OUTPUT_DIR / "image_manifest.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            
            writer = csv.writer(file)
            
            writer.writerow([
                "split",
                "class",
                "original_image",
                "copied_image"
            ])
            
            writer.writerows(all_saved)
        
        print("\nCnn dataset preparation complete.")
        print("Output location:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
    
    
        