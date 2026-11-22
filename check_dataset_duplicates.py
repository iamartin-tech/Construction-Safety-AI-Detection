# Construction Safety AI Project
# Part 1 - Check Dataset Duplicates
#
# Uses SHA-256 hashes to identify files
# with identical contents across splits.

from pathlib import Path
from collections import defaultdict
import hashlib


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "cnn_dataset"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def calculate_hash(image_path):
    """Calculate the SHA-256 hash of an image file."""
    
    sha256 = hashlib.sha256()
    
    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            sha256.update(chunk)
            
    return sha256.hexdigest()

def main():
    print("=" * 55)
    print("CNN DATASET DUPLICATE IMAGE CHECK")
    print("=" * 55)
    
    hashes = defaultdict(list)
    
    for split in SPLITS:
        split_dir = DATASET_DIR / split
        count = 0
        
        for image_path in split_dir.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            image_hash = calculate_hash(image_path)
            hashes[image_hash].append((split, image_path))
            count += 1
            
        print(f"{split.upper()}: {count} images checked")
        
    duplicates = []
    
    for image_hash, records in hashes.items():
        split_names = {record[0] for record in records}
        
        # Report duplicates appearing in different splits
        if len(split_names) > 1:
            duplicates.append(records)
    
    print("\nRESULTS")
    print("-" * 55)
    
    if not duplicates:
        print("PASS: No exact duplicate images across splits.")
    else:
        print(
            "WARNING:",
            len(duplicates),
            "duplicate image groups across splits."
        )
        
        for group_number, records in enumerate(
            duplicates, start=1
        ):
            
            print(f"\nDuplicate group {group_number}")
            
            for split, path in records:
                print(f" {split}: {path}")
    
    print("\nDuplicate check complete.")

if __name__ == "__main__":
    main()
    