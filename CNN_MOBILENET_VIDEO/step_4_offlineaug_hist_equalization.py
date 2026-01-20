import os
import cv2
import random
import shutil
import numpy as np

# =====================
# KONFIGURASI
# =====================
SOURCE_DIR = "BISINDO_split"
TARGET_DIR = "BISINDO_split_aug"

IMG_SIZE = 224
AUG_MULTIPLIER = 3  # setiap gambar train → 3 versi baru

random.seed(42)

# =====================
# FUNGSI AUGMENTASI
# =====================
def apply_clahe(img):
    """Histogram Equalization pakai CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def augment_image(img):
    h, w, _ = img.shape

    # Flip horizontal (50%)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Rotasi ringan
    angle = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    # Zoom ringan
    scale = random.uniform(0.9, 1.05)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Brightness ringan
    value = random.randint(-15, 15)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Histogram Equalization
    img = apply_clahe(img)

    return img

# =====================
# COPY VAL & TEST (NO AUG)
# =====================
for split in ["val", "test"]:
    src = os.path.join(SOURCE_DIR, split)
    dst = os.path.join(TARGET_DIR, split)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

# =====================
# AUGMENT TRAIN
# =====================
train_src = os.path.join(SOURCE_DIR, "train")
train_dst = os.path.join(TARGET_DIR, "train")

os.makedirs(train_dst, exist_ok=True)

for cls in sorted(os.listdir(train_src)):
    src_cls = os.path.join(train_src, cls)
    dst_cls = os.path.join(train_dst, cls)
    os.makedirs(dst_cls, exist_ok=True)

    images = [
        f for f in os.listdir(src_cls)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_name in images:
        img_path = os.path.join(src_cls, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Simpan original
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(
            os.path.join(dst_cls, f"{base_name}_orig.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        # Simpan hasil augmentasi
        for i in range(AUG_MULTIPLIER):
            aug = augment_image(img)
            cv2.imwrite(
                os.path.join(dst_cls, f"{base_name}_aug{i}.jpg"),
                cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            )

    print(f"[OK] Train class {cls} augmented")

print("\nselesai: Offline Augmentation + CLAHE")
