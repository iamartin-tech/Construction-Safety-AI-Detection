# Construction Safety AI Project
# Part 1 - CNN Image Classification
#
# Dataset:
# https://universe.roboflow.com/safety-helmet-apd/construction-site-safety-w0dlf/dataset/1
#
# Classification:
# 0 = Hardhat Only
# 1 = No-Hardhat Present
#
# Uses separate training, validation, and test images.

import os

# Reduce information TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models

# 1. PROJECT SETTINGS
# =======================================
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "cnn_dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "part1_cnn"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

CLASS_NAMES = [
    "Hardhat_Only",
    "No_Hardhat_Present"
]

tf.keras.utils.set_random_seed(SEED)

print("=" * 50)
print("CONSTRUCTION SAFETY CNN CLASSIFICATION")
print("=" * 50)

print("TensorFlow:", tf.__version__)
print("Dataset:", DATASET_DIR)

# 2. LOAD DATASETS
# =======================================

def load_dataset(split_name, shuffle):
    folder = DATASET_DIR / split_name
    
    return tf.keras.utils.image_dataset_from_directory(
        str(folder),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_ds = load_dataset("train", True)
valid_ds = load_dataset("valid", False)
test_ds = load_dataset("test", False)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 3. BUILD THE CNN MODEL
# =======================================

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    
    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),
    
    # Random variations to improve generalization
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    
    # Convolution block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Convolution block 2
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Convolution block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    
    # Binary output: probability of No-Hardhat Present
    layers.Dense(1, activation="sigmoid")
])

# 4. COMPILE MODEL
# =======================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL SUMMARY")
model.summary()

# 5. TRAIN CNN
# =======================================

print("\nSTARTING CNN TRAINING")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )
    ]
)

# 6. EVALUATE TEST DATASET
# =======================================

print("\nEVALUATING TEST DATASET")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
# 7. SAVE MODEL
# =======================================

model_path = RESULTS_DIR / "construction_safety_cnn.keras"
 
model.save(str(model_path))
 
print("\nModel saved:", model_path)
 
# 8. SAVE ACCURACY AND LOSS GRAPHS
# =======================================
 
epochs_completed = range(
    1, len(history.history["accuracy"]) + 1
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_completed,
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "accuracy_graph.png",
    dpi=150
)

plt.show()

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_completed,
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    epochs_completed,
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "loss_graph.png",
    dpi=150
)

plt.show()

# 9. PREDICT TEST IMAGES
# =======================================

print("\nGENERATING TEST IMAGE PREDICTIONS")

images_to_show = []
actual_labels = []
predicted_labels = []
probabilities = []

for images, labels in test_ds:
    
    predictions = model.predict(
        images,
        verbose=0
    ).flatten()
    
    for i in range(len(images)):
        
        images_to_show.append(images[i].numpy())
        
        actual_labels.append(
            int(labels[i].numpy().item())
        )
        
        probabilities.append(float(predictions[i]))
        
        predicted_labels.append(
            int(predictions[i] >= 0.5)
        )

# Save a visual sample of nine test predictions

sample_count = min(9, len(images_to_show))

fig, axes = plt.subplots(3, 3, figsize=(12, 11))

for i, ax in enumerate(axes.flat):
    
    if i >= sample_count:
        ax.axis("off")
        continue
    
    ax.imshow(
        images_to_show[i].astype("uint8")
    )
    
    actual = CLASS_NAMES[actual_labels[i]]
    predicted = CLASS_NAMES[predicted_labels[i]]
    
    correct = actual_labels[i] == predicted_labels[i]
    
    ax.set_title(
        f"Actual: {actual}\nPredicted: {predicted}",
        color="green" if correct else "red",
        fontsize=10
    )
    
    ax.axis("off")
    
plt.suptitle("CNN Test Image Predictions")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "test_predictions.png",
    dpi=150
)

plt.show()

# 10. SAVE METRICS
# =======================================

metrics = {
    "test_accuracy": float(test_accuracy),
    "rest_loss": float(test_loss),
    "epochs_completed": len(history.history["loss"]),
    "training_images": 306,
    "validation_images": 78,
    "testing_images": 51
}

with open(
    RESULTS_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    
    json.dump(metrics, file, indent=4)

print("\n" + "=" * 50)
print("CNN TRAINING COMPLETE")
print("=" * 50)

print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("Results saved to:", RESULTS_DIR)


    