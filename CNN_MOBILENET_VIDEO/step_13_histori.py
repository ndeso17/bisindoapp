from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from step6_data_generator_zscore import train_data, val_data
from step7_model_mobilenet import model
import json

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

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================
# SIMPAN HISTORY
# =====================
with open("history_bisindo_mobilenet.json", "w") as f:
    json.dump(history.history, f)

print("\nTraining selesai — model & history tersimpan")
