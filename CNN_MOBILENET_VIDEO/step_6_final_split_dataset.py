import os
import json
import shutil
import random

# =====================
# KONFIGURASI
# =====================
SOURCE_DIR = "BISINDO_split_aug/train"
TARGET_DIR = "BISINDO_FINAL"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# =====================
# SETUP FOLDER
# =====================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

# =====================
# AMBIL KELAS (SORTED!)
# =====================
classes = sorted([
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
])

class_indices = {cls: idx for idx, cls in enumerate(classes)}

# Simpan mapping
with open(os.path.join(TARGET_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=4)

print("Class index disimpan:")
print(class_indices)

# =====================
# SPLIT PER KELAS
# =====================
for cls in classes:
    src_cls = os.path.join(SOURCE_DIR, cls)
    images = [
        f for f in os.listdir(src_cls)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dst_cls = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for img in files:
            shutil.copy(
                os.path.join(src_cls, img),
                os.path.join(dst_cls, img)
            )

    print(
        f"[OK] {cls}: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

print("\nSTEP 3 FINAL selesai")
