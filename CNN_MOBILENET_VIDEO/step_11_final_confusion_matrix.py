import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# =====================
# KONFIGURASI
# =====================
BASE_DIR = "BISINDO_FINAL"
MODEL_PATH = "model_bisindo_mobilenet.h5"
IMG_SIZE = 224
BATCH_SIZE = 8

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# =====================
# LOAD CLASS INDEX
# =====================
with open(os.path.join(BASE_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)

class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

# =====================
# Z-SCORE FUNCTION
# =====================
def zscore(img):
    img = img / 255.0
    return (img - MEAN) / STD

# =====================
# DATA GENERATOR (TEST)
# =====================
datagen = ImageDataGenerator(
    preprocessing_function=zscore
)

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
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 7}
)

plt.title("Confusion Matrix BISINDO - MobileNetV2", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=300)
plt.show()

print("STEP 11 FINAL — Confusion Matrix valid & konsisten")
