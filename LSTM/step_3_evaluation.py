import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATASET_DIR = "landmark_sequence_dataset2"
MODEL_PATH = "model_bisindo_landmark_lstm.h5"
CLASS_INDEX_PATH = "class_index_lstm.json"
OUTPUT_DIR = "evaluation_plots"

SEQUENCE_LEN = 10
NUM_FEATURES = 126

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CLASS_INDEX_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
labels = [idx_to_class[i] for i in range(len(idx_to_class))]

X, y = [], []

for cls, idx in class_to_idx.items():
    cls_dir = os.path.join(DATASET_DIR, cls)
    for file in os.listdir(cls_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(cls_dir, file))
            if seq.shape == (SEQUENCE_LEN, NUM_FEATURES):
                X.append(seq)
                y.append(idx)

X = np.array(X)
y = np.array(y)

print("Total data:", X.shape)

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

model = load_model(MODEL_PATH)
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(13, 11))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.3
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - All Classes (LSTM BISINDO)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_all_classes.png"), dpi=300)
plt.close()

huruf_mirip = ['A', 'D', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'X']
idx = [class_to_idx[h] for h in huruf_mirip]

cm_mirip = cm[np.ix_(idx, idx)]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_mirip,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=huruf_mirip,
    yticklabels=huruf_mirip,
    linewidths=0.3
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Huruf Mirip (LSTM BISINDO)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_huruf_mirip.png"), dpi=300)
plt.close()

print(" Semua grafik berhasil disimpan ke folder:", OUTPUT_DIR)
