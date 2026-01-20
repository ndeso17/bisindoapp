import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# =====================
# KONFIGURASI
# =====================
BASE_DIR = "BISINDO_FINAL"
MODEL_PATH = "model_bisindo_mobilenet.h5"
IMG_SIZE = 224
BATCH_SIZE = 8

# =====================
# LOAD CLASS INDEX (WAJIB)
# =====================
with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)

# urut berdasarkan index output model
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

# =====================
# DATA GENERATOR (TEST)
# =====================
datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# PREDIKSI
# =====================
test_data.reset()
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

# =====================
# CLASSIFICATION REPORT
# =====================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n=== CLASSIFICATION REPORT BISINDO (MobileNetV2) ===\n")
print(report)

# =====================
# SIMPAN KE FILE
# =====================
with open("classification_report_final.txt", "w") as f:
    f.write(report)

print("STEP 12 FINAL selesai — classification_report_final.txt berhasil disimpan")
