# Construction Safety AI Project
# Part 3 - Gemini Flash Object Detection
#
# Dataset:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1
#
# Uses the same three test images as Part 2 (YOLO26).
# No model training is required.

import os
import json
import time
from pathlib import Path

from google import genai
from google.genai import types, errors
from PIL import Image, ImageDraw, ImageFont

# 1. PROJECT SETTINGS
PROJECT_DIR = Path(__file__).resolve().parent

TEST_IMAGES_DIR = (
    PROJECT_DIR / "dataset" / "test" / "images"
)

RESULTS_DIR = (
    PROJECT_DIR / "results" / "part3_gemini"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-3.8-flash"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Gemini will return bounding boxes on a 0-1000 scle
BOX_SCALE = 1000

MAX_ATTEMPTS = 3

# 2. CONNECT TO GEMINI
print("=" * 55)
print("PART 3 - GEMINI FLASH OBJECT DETECTION")
print("=" * 55)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not found."
    )

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)

print("\nGemini API client initialized.")
print("Model:", MODEL_NAME)
# 3. SELECT THREE TEST IMAGES
test_images = sorted(
    image for image in TEST_IMAGES_DIR.iterdir()
    if image.suffix.lower() in IMAGE_EXTENSIONS
)

if len(test_images) < 3:
    raise RuntimeError(
        "At least three test images are required."
    )

selected_images = test_images[:3]

print("\nSelected test images:")

for index, image_path in enumerate(selected_images, 1):
    print(f"{index}. {image_path.name}")

# 4. OBJECT DETECTION PROMPT
PROMPT = """
Analyze this construction site image and identify visible
objects relevant to construction site safety.

Look for:
- Person
- Hardhat
- NO-Hardhat (person visibly without a hardhat)
- Safety Vest
- NO-Safety Vest (person visibly without a safety vest)
- Gloves
- Ladder
- Safety Cone
- Machinery
- Vehicles
- Other clearly identifiable construction safety objects

For each detected object, provide:
1. A short class label.
2. A bounding box.

Bounding box format:
[ymin, xmin, ymax, xmax]

All coordinates must be integers normalized from 0 to 1000.

The top-left corner of the image is (0, 0).
The bottom-right corner is (1000, 1000).

Return a JSON object with this exact structure:

{
  "detections": [
    {
      "label": "Person",
      "box": [100, 200, 800, 600]
    }
  ]
}

Return only visible objects. Do not invent detections.
If no relevant objects are visible, return an empty
detections array.

Avoid duplicating the same physical object.
"""


# 5. HELPER FUNCTION: VALIDATE BOXES
def convert_box(box, width, height):
    """
    Convert normalized [ymin, xmin, ymax, xmax]
    coordinates into image pixel coordinates.
    """

    if not isinstance(box, list) or len(box) != 4:
        return None

    try:
        ymin, xmin, ymax, xmax = [
            float(value) for value in box
        ]
    except (TypeError, ValueError):
        return None

    # Clamp values to the expected normalized range.
    ymin = max(0, min(BOX_SCALE, ymin))
    xmin = max(0, min(BOX_SCALE, xmin))
    ymax = max(0, min(BOX_SCALE, ymax))
    xmax = max(0, min(BOX_SCALE, xmax))

    if ymin >= ymax or xmin >= xmax:
        return None

    x1 = round(xmin / BOX_SCALE * width)
    y1 = round(ymin / BOX_SCALE * height)
    x2 = round(xmax / BOX_SCALE * width)
    y2 = round(ymax / BOX_SCALE * height)

    return (x1, y1, x2, y2)


# 6. HELPER FUNCTION: DRAW DETECTIONS
def draw_detections(image, detections):
    """
    Draw labeled bounding boxes on a copy of the image.
    """

    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    width, height = annotated.size

    font = ImageFont.load_default()

    valid_count = 0

    for detection in detections:

        if not isinstance(detection, dict):
            continue

        label = str(detection.get("label", "Object"))
        box = detection.get("box")

        pixel_box = convert_box(box, width, height)

        if pixel_box is None:
            print("Skipping invalid box:", box)
            continue

        x1, y1, x2, y2 = pixel_box

        # Draw the object bounding box.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3
        )

        # Draw a filled background for the label.
        label_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = label_bbox[2] - label_bbox[0]
        text_height = label_bbox[3] - label_bbox[1]

        label_y = max(0, y1 - text_height - 8)

        draw.rectangle(
            (
                x1,
                label_y,
                min(width, x1 + text_width + 8),
                label_y + text_height + 6
            ),
            fill="red"
        )

        draw.text(
            (x1 + 4, label_y + 2),
            label,
            fill="white",
            font=font
        )

        valid_count += 1

    return annotated, valid_count


# 7. PROCESS THE THREE TEST IMAGES
summary = []

try:
    for index, image_path in enumerate(
        selected_images, start=1
    ):

        print("\n" + "-" * 55)
        print(f"Checking image {index}/3")
        print("Filename:", image_path.name)

        # --------------------------------------------------
        # CHECK FOR PREVIOUSLY COMPLETED IMAGES
        # --------------------------------------------------

        output_path = RESULTS_DIR / (
            f"gemini_image_{index}.jpg"
        )

        json_path = RESULTS_DIR / (
            f"gemini_image_{index}.json"
        )

        # Skip an image if both output files exist.
        if output_path.exists() and json_path.exists():

            print(
                f"Skipping image {index}: "
                "Already completed."
            )

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:
                saved_result = json.load(file)

            saved_detections = saved_result.get(
                "detections", []
            )

            summary.append({
                "image": image_path.name,
                "status": "Completed (previous run)",
                "detections": len(saved_detections)
            })

            continue

        # --------------------------------------------------
        # LOAD IMAGE
        # --------------------------------------------------

        print(f"\nProcessing image {index}/3")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        response = None

        # --------------------------------------------------
        # SEND IMAGE TO GEMINI WITH RETRIES
        # --------------------------------------------------

        for attempt in range(1, MAX_ATTEMPTS + 1):

            try:
                print(
                    f"Gemini request "
                    f"{attempt}/{MAX_ATTEMPTS}"
                )

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[PROMPT, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0
                    )
                )

                break

            except errors.APIError as error:

                print("API error:", error.code)
                print("Message:", error.message)

                # Retry temporary API errors.
                if (
                    error.code in (500, 502, 503, 504)
                    and attempt < MAX_ATTEMPTS
                ):

                    # Exponential backoff:
                    # First failure: 15 seconds
                    # Second failure: 30 seconds

                    wait_seconds = 15 * (
                        2 ** (attempt - 1)
                    )

                    print(
                        f"Waiting {wait_seconds} "
                        "seconds before retry..."
                    )

                    time.sleep(wait_seconds)

                else:
                    print(
                        "Unable to process this image."
                    )
                    break

        # --------------------------------------------------
        # HANDLE FAILED API REQUESTS
        # --------------------------------------------------

        if response is None:

            summary.append({
                "image": image_path.name,
                "status": "API request failed",
                "detections": 0
            })

            continue

        # --------------------------------------------------
        # PARSE GEMINI JSON RESPONSE
        # --------------------------------------------------

        try:
            result = json.loads(response.text)

            detections = result.get(
                "detections", []
            )

            if not isinstance(detections, list):
                raise ValueError(
                    "detections must be a list"
                )

        except (ValueError, TypeError) as error:

            print(
                "Invalid Gemini JSON:",
                error
            )

            summary.append({
                "image": image_path.name,
                "status": "Invalid JSON",
                "detections": 0
            })

            continue

        # --------------------------------------------------
        # DISPLAY DETECTED OBJECTS
        # --------------------------------------------------

        print(
            "Objects returned:",
            len(detections)
        )

        for detection in detections:
            print(detection)

        # --------------------------------------------------
        # DRAW BOUNDING BOXES
        # --------------------------------------------------

        annotated_image, valid_count = (
            draw_detections(
                image,
                detections
            )
        )

        print(
            "Valid boxes drawn:",
            valid_count
        )

        # --------------------------------------------------
        # SAVE ANNOTATED IMAGE
        # --------------------------------------------------

        annotated_image.save(
            output_path
        )

        print(
            "Annotated image saved:",
            output_path
        )

        # --------------------------------------------------
        # SAVE GEMINI JSON RESULTS
        # --------------------------------------------------

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "JSON saved:",
            json_path
        )

        # --------------------------------------------------
        # RECORD SUCCESSFUL IMAGE
        # --------------------------------------------------

        summary.append({
            "image": image_path.name,
            "status": "Completed",
            "detections": valid_count
        })

finally:

    client.close()


# 8. SAVE SUMMARY
summary_path = RESULTS_DIR / "summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )

print("\n" + "=" * 55)
print("PART 3 - GEMINI DETECTION FINISHED")
print("=" * 55)

for item in summary:
    print(
        item["image"],
        "|",
        item["status"],
        "| Detections:",
        item["detections"]
    )

print("\nResults folder:", RESULTS_DIR)
