from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import numpy as np

# =====================
# KONFIGURASI
# =====================
BASE_DIR = "BISINDO_FINAL"
IMG_SIZE = 224
BATCH_SIZE = 8

# Z-SCORE
#nilai data- mean/standar deviasi
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# LOAD CLASS INDEX
with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)

NUM_CLASSES = len(class_indices)

print("Class mapping digunakan:")
print(class_indices)

# Z-SCORE FUNCTION
def zscore(img):
    img = img / 255.0
    return (img - MEAN) / STD

# =====================
# DATA GENERATOR
# =====================
datagen = ImageDataGenerator(
    preprocessing_function=zscore
)

train_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nSummary:")
print("Train:", train_data.samples)
print("Val  :", val_data.samples)
print("Test :", test_data.samples)
