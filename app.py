from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import cv2

# =====================
# IMPORT LOGIC ML
# =====================
from utils.cnn_preprocess import predict_image
from utils.lstm_predict import process_frame

# =====================
# FLASK CONFIG
# =====================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["CITRA_BISINDO"] = "Citra_BISINDO"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =====================
# CAMERA (LSTM)
# =====================
camera = None

def get_camera():
    global camera
    if camera is None:
        try:
            # Menggunakan index -1 atau 0, dan tambahkan CAP_DSHOW jika di windows local
            # Tapi untuk server linux, cukup 0.
            # Kita bungkus try-except agar tidak crash saat init
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                camera = cap
        except Exception as e:
            print(f"Warning: Camera init failed: {e}")
            camera = None
    return camera

# =====================
# STATE ML
# =====================
latest_lstm_prediction = "..."

# =====================
# PAGE ROUTES
# =====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("translate_upload.html")

@app.route("/realtime")
def realtime_page():
    return render_template("translate_realtime.html")

@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")

@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    try:
        result = predict_image(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-realtime-prediction")
def get_realtime_prediction():
    return jsonify({"label": latest_lstm_prediction})

@app.route("/bisindo-image/<letter>/<filename>")
def get_bisindo_image(letter, filename):
    return send_from_directory(os.path.join(app.config["CITRA_BISINDO"], letter), filename)

@app.route("/bisindo-list")
def get_bisindo_list():
    folder_path = app.config["CITRA_BISINDO"]
    letters = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    mapping = {}
    for letter in letters:
        images = os.listdir(os.path.join(folder_path, letter))
        if images:
            mapping[letter] = images[0] # Just take the first image
    return jsonify(mapping)

# =====================
# LSTM - REALTIME STREAM
# =====================

def generate_frames():
    global latest_lstm_prediction
    
    # Placeholder jika kamera mati
    blank_frame = cv2.imencode('.jpg', 
        cv2.putText(
            ((0 * np.ones((480, 640, 3), np.uint8))), 
            "Camera Not Found", (200, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )[0]
    )[1].tobytes() if 'cv2' in globals() and 'np' in globals() else b''
    
    while True:
        cap = get_camera()
        
        if cap is None or not cap.isOpened():
            # Jika tidak ada kamera (server), stream black frame/placeholder
            # Sleep agar tidak membebani CPU
            cv2.waitKey(1000)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                blank_frame +
                b"\r\n"
            )
            continue

        success, frame = cap.read()
        if not success:
            continue

        try:
            # Update: Sekarang mengembalikan (frame, label)
            frame, label = process_frame(frame)
            latest_lstm_prediction = label
            if frame is None:
                continue
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
