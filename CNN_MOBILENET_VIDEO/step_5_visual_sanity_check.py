import os
import cv2
import random
import matplotlib.pyplot as plt

# =====================
# KONFIGURASI
# =====================
DATASET_DIR = "BISINDO_split_aug/train"
OUTPUT_DIR = "sanity_check"

IMG_SIZE = 224
SAMPLES_PER_CLASS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print(f"Total kelas: {len(classes)}")

# =====================
# VISUAL CHECK
# =====================
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) == 0:
        print(f"[WARNING] Kelas {cls} kosong!")
        continue

    samples = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

    plt.figure(figsize=(12, 3))
    plt.suptitle(f"Kelas {cls}", fontsize=14)

    for i, img_name in enumerate(samples):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, SAMPLES_PER_CLASS, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{cls}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[OK] Sanity check saved: {save_path}")

print("\nSTEP 2 FINAL selesai")
