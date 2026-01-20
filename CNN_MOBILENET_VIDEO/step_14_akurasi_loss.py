import os
import json
import matplotlib.pyplot as plt

# ======================================================
# STEP 14: VISUALISASI HASIL TRAINING (AKURASI & LOSS)
# ======================================================

# File history yang dihasilkan dari Step 10
HISTORY_FILE = "training_history.json"

def show_plots():
    # 1. Cek apakah file history sudah ada
    if not os.path.exists(HISTORY_FILE):
        print(f"\n[ERROR] File '{HISTORY_FILE}' tidak ditemukan.")
        print("Silahkan tunggu hingga proses training (Step 10) selesai.")
        print("Atau pastikan file tersebut berada di direktori yang sama dengan script ini.")
        return

    # 2. Load data history
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        print(f"[INFO] Berhasil memuat history dari {HISTORY_FILE}")
    except Exception as e:
        print(f"[ERROR] Gagal membaca file history: {e}")
        return

    # 3. Ekstrak data (mengantisipasi variasi nama key di Keras)
    acc = history.get('accuracy') or history.get('acc')
    val_acc = history.get('val_accuracy') or history.get('val_acc')
    loss = history.get('loss')
    val_loss = history.get('val_loss')

    # Validasi jika data kosong atau tidak lengkap
    if acc is None or val_acc is None or loss is None or val_loss is None:
        print("[ERROR] Struktur data di dalam JSON tidak sesuai (accuracy/loss tidak ditemukan).")
        return

    epochs = range(1, len(acc) + 1)

    # 4. Konfigurasi Visualisasi
    # Menggunakan style yang bersih dan profesional
    plt.style.use('seaborn-v0_8-muted') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Plot Akurasi ----
    ax1.plot(epochs, acc, 'tab:blue', label='Train Accuracy', marker='o', markersize=4, linewidth=2)
    ax1.plot(epochs, val_acc, 'tab:red', label='Validation Accuracy', marker='s', markersize=4, linewidth=2)
    ax1.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', frameon=True, shadow=True)

    # ---- Plot Loss ----
    ax2.plot(epochs, loss, 'tab:blue', label='Train Loss', marker='o', markersize=4, linewidth=2)
    ax2.plot(epochs, val_loss, 'tab:red', label='Validation Loss', marker='s', markersize=4, linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', frameon=True, shadow=True)

    # Header Utama
    plt.suptitle("MobileNetV2 Training Performance - BISINDO Gestures", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 5. Simpan Hasil ke Disk
    output_image = "akurasi_loss_plot_final.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"[INFO] Grafik visualisasi berhasil disimpan ke: {output_image}")
    
    # 6. Tampilkan Plot
    plt.show()

if __name__ == "__main__":
    show_plots()
