from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# =====================
# KONFIGURASI
# =====================
BASE_DIR = "BISINDO_FINAL"
IMG_SIZE = 224
BATCH_SIZE = 8

# =====================
# LOAD CLASS INDEX
# =====================
with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)

print("Class mapping (LOCKED):")
print(class_indices)

# =====================
# DATA GENERATOR (NO AUG)
# =====================
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================
# CEK JUMLAH
# =====================
print("\nJumlah data:")
print("Train:", train_data.samples)
print("Val  :", val_data.samples)
print("Test :", test_data.samples)

# =====================
# CEK KELAS
# =====================
print("\nJumlah kelas:")
print("Train:", train_data.num_classes)
print("Val  :", val_data.num_classes)
print("Test :", test_data.num_classes)

# =====================
# CEK KONSISTENSI
# =====================
assert train_data.num_classes == len(class_indices)
assert val_data.num_classes == len(class_indices)
assert test_data.num_classes == len(class_indices)

print("\nSTEP 4 FINAL berhasil — data siap training")
