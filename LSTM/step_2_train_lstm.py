import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


DATASET_DIR = "landmark_sequence_dataset2"
SEQUENCE_LEN = 10
NUM_FEATURES = 126
MODEL_PATH = "model_bisindo_landmark_lstm.h5"
CLASS_INDEX_PATH = "class_index_lstm.json"

EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

X, y = [], []
classes = sorted(os.listdir(DATASET_DIR))
class_to_idx = {c: i for i, c in enumerate(classes)}

for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    for file in os.listdir(cls_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(cls_dir, file))
            if seq.shape == (SEQUENCE_LEN, NUM_FEATURES):
                X.append(seq)
                y.append(class_to_idx[cls])

X = np.array(X)
y = to_categorical(y, num_classes=len(classes))

print("Total data :", X.shape[0])
print("Input shape:", X.shape[1:])

with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(class_to_idx, f, indent=2)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train:", X_train.shape[0])
print("Val  :", X_val.shape[0])
print("Test :", X_test.shape[0])

model = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LEN, NUM_FEATURES), return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
=
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)

history_path = "history_lstm2.json"

history_data = {
    "loss": history.history.get("loss", []),
    "accuracy": history.history.get("accuracy", []),
    "val_loss": history.history.get("val_loss", []),
    "val_accuracy": history.history.get("val_accuracy", [])
}

with open(history_path, "w") as f:
    json.dump(history_data, f, indent=2)

print("History training disimpan di:", history_path)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
