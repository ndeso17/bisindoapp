import cv2
import os

# VIDEO_DIR  : folder berisi video BISINDO per huruf
# OUTPUT_DIR : folder hasil ekstraksi frame (dataset citra)
VIDEO_DIR = "Video_BISINDO"
OUTPUT_DIR = "Citra_BISINDO"

# FRAME_SKIP : interval pengambilan frame
#              (1 frame diambil setiap 5 frame video)
# RESIZE     : ukuran citra hasil ekstraksi (256.256)
FRAME_SKIP = 5
RESIZE = (256, 256)

# Membuat folder output utama jika belum tersedia
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan daftar folder kelas (huruf BISINDO)
print(" Folder huruf:", os.listdir(VIDEO_DIR))

# PROSES EKSTRAKSI FRAME PER KELAS
# Setiap folder dianggap sebagai satu label kelas
for label in os.listdir(VIDEO_DIR):
    label_path = os.path.join(VIDEO_DIR, label)

    # pastikan ini folder (a, b, c, ...)
    if not os.path.isdir(label_path):
        continue

    # Mendukung beberapa format video umum
    video_files = [
        f for f in os.listdir(label_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    # Jika tidak ditemukan video, kelas dilewati
    if len(video_files) == 0:
        print(f"⚠️ Tidak ada video di folder {label}")
        continue

    # Mengambil satu video sebagai sumber data
    video_name = video_files[0] 
    video_path = os.path.join(label_path, video_name)

    print(f"▶️ Processing {label} → {video_name}")

     # Struktur ini sesuai dengan format dataset CNN
    output_label_dir = os.path.join(OUTPUT_DIR, label.upper())
    os.makedirs(output_label_dir, exist_ok=True)

    # Membuka video menggunakan OpenCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Gagal membuka video {video_name}")
        continue

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
         # Jika frame tidak terbaca, video telah selesai
        if not ret:
            break
        # Mengambil frame berdasarkan interval FRAME_SKIP
        if frame_idx % FRAME_SKIP == 0:
            frame = cv2.resize(frame, RESIZE)
            # Menyimpan frame sebagai file citra (.jpg)
            cv2.imwrite(
                os.path.join(output_label_dir, f"{label}_{saved:04d}.jpg"),
                frame
            )
            saved += 1

        frame_idx += 1
    # Melepas resource video
    cap.release()
     # Menampilkan jumlah frame yang berhasil diekstraks
    print(f" {label.upper()} → {saved} frame disimpan\n")
