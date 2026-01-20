# =========================================
# STEP 10 TRAINING MOBILENETV2 (FULL SCRIPT)
# =========================================

import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =====================
# IMPORT DATA & MODEL
# =====================
from step_8_data_generator_zscore import train_data, val_data
from step_9_model_mobilenet import model

# =====================
# KONFIGURASI TRAINING
# =====================
EPOCHS = 30

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "model_bisindo_mobilenet.h5",
        monitor="val_loss",
        save_best_only=True
    )
]

# =====================
# TRAINING MODEL
# =====================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining selesai — model terbaik tersimpan")

# =====================
# SIMPAN HISTORY TRAINING
# =====================
history_dict = history.history

with open("training_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

print("History training disimpan ke training_history.json")

# =====================
# PLOT ACCURACY & LOSS
# =====================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# ---- Accuracy ----
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc)
plt.plot(epochs_range, val_acc)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# ---- Loss ----
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss)
plt.plot(epochs_range, val_loss)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()

print("\nSTEP 10 SELESAI — MODEL SIAP DIEVALUASI")
