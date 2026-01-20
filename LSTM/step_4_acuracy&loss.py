import json
import matplotlib.pyplot as plt
import os

HISTORY_PATH = "history_lstm2.json"
OUTPUT_DIR = "evaluation_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["accuracy"]) + 1)

plt.figure(figsize=(7,5))
plt.plot(epochs, history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy (LSTM BISINDO)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.plot(epochs, history["loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (LSTM BISINDO)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
plt.close()

print(" Grafik accuracy & loss berhasil disimpan")
