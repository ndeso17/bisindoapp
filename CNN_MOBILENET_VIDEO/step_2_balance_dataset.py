import os
import cv2
import random
import numpy as np

# =====================
# KONFIGURASI
# =====================
SOURCE_DIR = "Citra_BISINDO"
BALANCED_DIR = "dataset_balanced"
TARGET_PER_CLASS = 120
IMG_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(BALANCED_DIR, exist_ok=True)

# =====================
# AUGMENTASI AMAN
# =====================
def safe_augment(img):
    h, w, _ = img.shape

    # Flip horizontal (opsional)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Rotasi kecil
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(
        hsv[..., 2] + random.randint(-15, 15), 0, 255
    )
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Blur ringan (opsional)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

# =====================
# PROSES BALANCING
# =====================
for label in sorted(os.listdir(SOURCE_DIR)):
    src_class = os.path.join(SOURCE_DIR, label)
    dst_class = os.path.join(BALANCED_DIR, label)

    if not os.path.isdir(src_class):
        continue

    os.makedirs(dst_class, exist_ok=True)

    images = [f for f in os.listdir(src_class)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    random.shuffle(images)

    count = 0
    idx = 0

    while count < TARGET_PER_CLASS:
        img_path = os.path.join(src_class, images[idx % len(images)])
        img = cv2.imread(img_path)

        if img is None:
            idx += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Augment hanya jika data asli habis
        if idx >= len(images):
            img = safe_augment(img)

        save_path = os.path.join(dst_class, f"{label}_{count:04d}.jpg")
        cv2.imwrite(save_path, img)

        count += 1
        idx += 1

    print(f"[OK] {label}: {count} images")

print("\n STEP 1 selesai: Dataset seimbang & aman")
