
# Construction Site Safety Using Artificial Intelligence

## Project Overview

This project investigates three artificial intelligence approaches for analyzing construction site safety images:

1. Convolutional Neural Network (CNN) image classification
2. YOLO26 object detection
3. Google Gemini Flash prompt-based object detection

The goal is to examine how different AI techniques can identify construction workers, protective equipment, and potential safety violations in construction site images.

The project was developed in Python using Eclipse IDE with PyDev as part of a Multimedia Processing course.

---

## 1. Dataset

**Dataset:** Construction Site Safety — Roboflow Universe

**Source:** [Construction Site Safety Dataset](https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1)

The original dataset contains 717 images.

| Dataset Split | Images |
|---|---:|
| Training | 521 |
| Validation | 114 |
| Testing | 82 |
| **Total** | **717** |

The dataset includes 25 annotated object classes, including:

- Person
- Hardhat
- NO-Hardhat
- Safety Vest
- NO-Safety Vest

The original YOLO annotations were used for object detection. A separate binary classification dataset was prepared for the CNN.

The full dataset is not included in this repository. It can be obtained from the original source.

---

## 2. Part 1 — CNN Image Classification

### Objective

Develop a convolutional neural network to classify construction site images according to hardhat presence.

### Implementation

The CNN was developed using TensorFlow and Keras.

The original object detection annotations were used to prepare an image-level binary classification dataset.

The model used:

- Convolutional layers
- Max pooling
- Dropout
- Image preprocessing
- Binary image classification

These components allow the network to learn visual patterns associated with hardhat presence.

### Dataset Preparation

| Dataset Split | Images |
|---|---:|
| Training | 306 |
| Validation | 78 |
| Testing | 51 |
| **Total** | **435** |

### Evaluation Results

| Metric | Result |
|---|---:|
| Test Accuracy | 54.90% |
| Test Loss | 0.6627 |
| Epochs Completed | 7 |

The CNN achieved a test accuracy of 54.90%, indicating limited classification performance.

Possible contributing factors include:

- A relatively small training dataset
- Differences in lighting conditions
- Variations in camera angles
- Multiple objects appearing in the same image
- Difficulty identifying safety equipment from complete construction site images

### Generated Results

Results are saved in:

`results/part1_cnn/`

Generated files include:

- `construction_safety_cnn.keras` — Trained CNN model
- `accuracy_graph.png` — Training and validation accuracy
- `loss_graph.png` — Training and validation loss
- `test_predictions.png` — CNN prediction examples
- `metrics.json` — Evaluation metrics

**Main program:** `part1_cnn.py`

**Dataset preparation program:** `prepare_cnn_dataset.py`

---

## 3. Part 2 — YOLO26 Object Detection

### Objective

Train a YOLO26 object detection model to identify construction workers, protective equipment, and potential safety violations.

### Implementation

The Ultralytics YOLO26 nano model was trained using the original construction safety dataset.

Unlike the CNN, which performs image-level classification, YOLO26 predicts individual objects using class labels and bounding boxes.

The model was trained for 10 epochs and evaluated using the test dataset.

### Evaluation Results

| Metric | Result |
|---|---:|
| Precision | 63.12% |
| Recall | 31.16% |
| mAP50 | 32.14% |
| mAP50-95 | 21.99% |

The results demonstrate that YOLO26 can identify construction safety objects.

However, the recall and mean average precision indicate that some objects were missed or incorrectly classified.

Additional training, dataset improvements, and hyperparameter adjustments could potentially improve detection performance.

### Generated Results

**Training results:**

`results/part2_yolo/training/`

Contains:

- Trained model weights
- Training loss graphs
- Evaluation statistics
- Validation predictions

**Test evaluation:**

`results/part2_yolo/test_evaluation/`

Contains:

- Confusion matrices
- Precision-recall curves
- Test evaluation visualizations

**Test predictions:**

`results/part2_yolo/test_predictions/`

Contains three construction site images annotated with predicted object labels and bounding boxes.

**Main programs:**

- `part2_yolo26.py`
- `part2_yolo_predictions.py`

**Configuration program:** `prepare_yolo_config.py`

---

## 4. Part 3 — Google Gemini Flash Detection

### Objective

Use a multimodal generative AI model to detect construction site safety objects through image prompts without additional model training.

### Implementation

Google Gemini Flash was accessed using the Google GenAI Python SDK.

**Model used:** `gemini-3.8-flash`

The program sends construction site images and a structured prompt requesting object labels and bounding-box coordinates.

The returned JSON detections are converted into pixel coordinates and drawn onto the original images using Pillow.

### Detection Results

Three test images were selected from the YOLO26 test dataset.

| Image | Status | Reported Detections |
|---|---|---:|
| Image 1 | Completed | 3 |
| Image 2 | API request failed | Not evaluated |
| Image 3 | Completed | 11 |

Gemini returned 14 detections across the two successfully processed images.

The second image could not be processed because the Gemini API returned HTTP 503 and 504 errors, followed by HTTP 429 when the free-tier daily request quota was exceeded.

The program preserved both successful results.

The failed request was not counted as zero detected objects because the API did not complete its analysis.

### Generated Results

Results are saved in:

`results/part3_gemini/`

Generated files include:

- `gemini_image_1.jpg` — First annotated image
- `gemini_image_1.json` — First image detection data
- `gemini_image_3.jpg` — Third annotated image
- `gemini_image_3.json` — Third image detection data
- `summary.json` — Processing status for all three images

**Main program:** `part3_gemini_detection.py`

---

## 5. Technologies and Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Eclipse IDE / PyDev | Development environment |
| TensorFlow / Keras | CNN development and training |
| PyTorch | Machine learning framework used by Ultralytics |
| Ultralytics YOLO26 | Object detection |
| Google GenAI SDK | Gemini API integration |
| Pillow | Image processing and annotation |
| NumPy | Numerical operations |
| Matplotlib | Graphs and visualization |
| Roboflow | Construction safety dataset |

---

## 6. Project Structure

The project contains the following main source files and directories:

```text
ConstructionSafetyAI/
|
|-- part1_cnn.py
|-- part2_yolo26.py
|-- part2_yolo_predictions.py
|-- part3_gemini_detection.py
|
|-- prepare_cnn_dataset.py
|-- prepare_yolo_config.py
|-- check_dataset_duplicates.py
|-- inspect_dataset.py
|
|-- test_environment.py
|-- test_tensorflow.py
|-- test_yolo26.py
|-- test_gemini_setup.py
|
|-- results/
|   |-- part1_cnn/
|   |-- part2_yolo/
|   |-- part3_gemini/
|
|-- README.md
|-- .gitignore
```

The full datasets, generated model weights, Python cache files, and local Eclipse configuration files are excluded from the public repository.

Selected evaluation graphs, metrics, and prediction images may be included to demonstrate the project's results.

---

## 7. Running the Project

### Step 1 — Python Environment

The project was developed using Python 3.12 in a Windows virtual environment.

Create a virtual environment:

```bash
python -m venv ConstructionSafetyAI_venv
```

Activate the environment on Windows:

```bat
ConstructionSafetyAI_venv\Scripts\activate
```

Install the required libraries:

```bash
pip install tensorflow ultralytics google-genai pillow numpy matplotlib pyyaml
```

Additional dependencies may be required depending on the local environment and installed package versions.

### Step 2 — Download the Dataset

Download the Construction Site Safety dataset from Roboflow:

https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1

Extract the dataset into the project's `dataset/` directory.

The original dataset should contain the training, validation, and testing image folders and their associated YOLO annotation files.

Dataset paths may need to be adjusted to match the local environment.

### Step 3 — CNN Classification

Prepare the CNN dataset by running:

```bash
python prepare_cnn_dataset.py
```

Run the CNN program:

```bash
python part1_cnn.py
```

The program trains the CNN, evaluates its performance, and saves the results.

Generated output is stored in:

`results/part1_cnn/`

### Step 4 — YOLO26 Object Detection

Prepare the YOLO configuration:

```bash
python prepare_yolo_config.py
```

Train and evaluate the YOLO26 model:

```bash
python part2_yolo26.py
```

Generate predictions using the trained model:

```bash
python part2_yolo_predictions.py
```

The prediction program uses previously saved trained weights and does not require retraining when those weights are available.

Results are stored in:

`results/part2_yolo/`

### Step 5 — Gemini Flash Detection

Configure a valid Gemini API key using the environment variable:

`GEMINI_API_KEY`

For Windows Command Prompt:

```bat
set GEMINI_API_KEY=YOUR_API_KEY
```

For PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with a valid API key. Do not save the actual key in the repository.

Run the Gemini detection program:

```bash
python part3_gemini_detection.py
```

The program sends selected images to Gemini, processes the JSON responses, and saves annotated images and detection information.

Internet connectivity and available API quota are required.

**Security note:** Never include API keys, passwords, or other credentials in source code or public GitHub files.

---

## 8. Comparison of the Three Approaches

| Feature | CNN | YOLO26 | Gemini Flash |
|---|---|---|---|
| Primary Task | Image classification | Object detection | Prompt-based detection |
| Task-Specific Training | Required | Required | Not required |
| Output | Image-level class | Labels and bounding boxes | Generated labels and bounding boxes |
| Internet Required for Inference | No | No | Yes |
| Main Limitation | Low classification accuracy | Missed detections | API availability and quota |

The evaluation metrics are not directly interchangeable because the CNN performs binary classification, YOLO26 performs multi-class object detection, and Gemini produces prompt-based detection results without a comparable ground-truth evaluation in this project.

---

## 9. Project Limitations

### CNN Limitations

The CNN was trained on a relatively small binary classification dataset and achieved 54.90% test accuracy.

The model may benefit from additional training data, image augmentation, and architecture improvements.

### YOLO26 Limitations

YOLO26 was trained for only 10 epochs, which may have limited its detection performance.

The model achieved 31.16% recall, indicating that a substantial portion of relevant objects may not have been detected.

### Gemini Limitations

Gemini's object labels and bounding boxes are generated from prompts and require visual inspection.

The API's daily quota prevented completion of one test image.

Gemini detections were not evaluated against ground-truth annotations using the same metrics as YOLO26.

---

## 10. Individual Project Contribution

This project was developed as part of my Multimedia Processing coursework.

I implemented and evaluated three artificial intelligence approaches for analyzing construction site safety images.

My responsibilities included:

- Preparing and inspecting the construction safety dataset.
- Developing a CNN classification model using TensorFlow.
- Training and evaluating a YOLO26 object detection model.
- Integrating Google Gemini Flash for prompt-based detection.
- Generating annotated images and evaluation graphs.
- Comparing the results and limitations of all three approaches.
- Troubleshooting Python dependencies and API errors.
- Documenting the implementation and evaluation results.

The project provided practical experience with machine learning, computer vision, object detection, and AI-assisted image analysis.

---

## 11. Conclusion

This project demonstrates three approaches to analyzing construction site safety images.

The CNN provides image-level hardhat classification, YOLO26 identifies individual objects and safety equipment using bounding boxes, and Gemini Flash performs prompt-based image analysis through a cloud API.

The results highlight differences in model training requirements, evaluation methods, detection capabilities, and reliance on external services.

Future improvements could include:

- Expanding the training dataset.
- Increasing the number of training epochs.
- Adjusting model hyperparameters.
- Improving image preprocessing.
- Comparing multiple YOLO model configurations.
- Evaluating Gemini predictions against manually verified ground-truth annotations.
- Testing the models on additional construction site images.

The project demonstrates both the potential and the limitations of applying artificial intelligence to construction safety monitoring.
