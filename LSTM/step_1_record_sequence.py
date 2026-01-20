import cv2
import numpy as np
import mediapipe as mp
import os
import json

OUTPUT_DIR = "landmark_sequence_dataset2"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SEQUENCE_LEN = 10
NUM_FEATURES = 126
TARGET_SEQ_PER_CLASS = 500

IMG_SIZE = 640

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for c in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

def normalize_landmarks(landmarks):
    lm = np.array(landmarks)
    lm -= lm[0]  # anchor wrist
    lm /= np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    return lm.flatten()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

current_class_idx = 0
sequence = []
recording = False
sequence_count = 0

print("=== RECORD DATA LSTM BISINDO ===")
print("S : mulai rekam sequence")
print("N : ganti huruf")
print("Q : keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    all_landmarks = []

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )
            for lm in hand_lms.landmark:
                all_landmarks.append([lm.x, lm.y, lm.z])

    # HANDLE 1 / 2 TANGAN
    if len(all_landmarks) == 21:
        all_landmarks += [[0, 0, 0]] * 21

    if len(all_landmarks) == 42 and recording:
        features = normalize_landmarks(all_landmarks)
        sequence.append(features)

    # SIMPAN SEQUENCE
    if len(sequence) == SEQUENCE_LEN:
        label = CLASSES[current_class_idx]
        save_path = os.path.join(
            OUTPUT_DIR,
            label,
            f"{label}_{sequence_count:04d}.npy"
        )
        np.save(save_path, np.array(sequence))
        sequence = []
        recording = False
        sequence_count += 1
        print(f"[SAVED] {save_path}")

    # DISPLAY
    cv2.rectangle(frame, (0, 0), (640, 90), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Huruf : {CLASSES[current_class_idx]}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Sequence : {sequence_count}/{TARGET_SEQ_PER_CLASS}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2
    )

    cv2.imshow("Record LSTM BISINDO", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if sequence_count < TARGET_SEQ_PER_CLASS:
            recording = True
            sequence = []
            print("Mulai rekam sequence...")
        else:
            print("Target kelas ini sudah tercapai")

    elif key == ord("n"):
        current_class_idx = (current_class_idx + 1) % len(CLASSES)
        sequence_count = len(os.listdir(
            os.path.join(OUTPUT_DIR, CLASSES[current_class_idx])
        ))
        sequence = []
        recording = False
        print(f"Ganti ke huruf {CLASSES[current_class_idx]}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
