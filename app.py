import os
import requests
import cv2
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# ================= PATH =================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, 'smartbin.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_float32.tflite')

# ================= SIMPAN URL STREAM =================
stream_url_store = {"url": ""}

# ================= LOAD MODEL =================
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model TFLite berhasil dimuat")

# ================= SETTING YOLO — disamakan dengan versi Pi lama =================
CONF_THRESHOLD = 0.55   # ← naik dari 0.45 ke 0.55 (sama dengan Pi lama)
NMS_THRESHOLD  = 0.45
MIN_BOX_RATIO  = 0.05
MAX_BOX_RATIO  = 0.70   # ← turun dari 0.95 ke 0.70 (sama dengan Pi lama)

# Zona deteksi valid — pusat objek harus ada di sini
# Filter ini yang bikin versi lama "hanya fokus 1 objek dan tidak deteksi tangan/wajah di pinggir"
ZONE_X1_RATIO = 0.10   # 10% dari kiri
ZONE_X2_RATIO = 0.90   # 90% dari kiri
ZONE_Y1_RATIO = 0.05   # 5% dari atas
ZONE_Y2_RATIO = 0.95   # 95% dari atas

class_names = ["Plastik", "anorganik", "b3", "organik"]
LABEL_MAP   = {
    "Plastik":   "Anorganik",
    "anorganik": "Anorganik",
    "organik":   "Organik",
    "b3":        "B3 - Berbahaya!",
}

# ================= INIT DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS deteksi (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu      DATETIME DEFAULT CURRENT_TIMESTAMP,
        kategori   TEXT,
        confidence REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS counter (
        id             INTEGER PRIMARY KEY,
        organik        INTEGER DEFAULT 0,
        anorganik      INTEGER DEFAULT 0,
        b3             INTEGER DEFAULT 0,
        kapasitas      INTEGER DEFAULT 50,
        last_detection TEXT DEFAULT ""
    )''')
    c.execute("SELECT COUNT(*) FROM counter")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO counter VALUES (1,0,0,0,50,'')")
    conn.commit()
    conn.close()
    print("✅ Database siap")

init_db()

# ================= YOLO DETECT =================
def detect_from_frame(frame):
    original_h, original_w = frame.shape[:2]

    # Zona valid dalam piksel
    zone_x1 = int(original_w * ZONE_X1_RATIO)
    zone_x2 = int(original_w * ZONE_X2_RATIO)
    zone_y1 = int(original_h * ZONE_Y1_RATIO)
    zone_y2 = int(original_h * ZONE_Y2_RATIO)

    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output      = interpreter.get_tensor(output_details[0]['index'])
    predictions = output[0].T

    all_boxes   = []
    all_confs   = []
    all_classes = []

    for det in predictions:
        cx, cy, w, h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        class_scores = det[4:]
        cls  = int(np.argmax(class_scores))
        conf = float(class_scores[cls])

        if conf < CONF_THRESHOLD:
            continue

        if cx <= 2.0:
            cx = cx * original_w; cy = cy * original_h
            w  = w  * original_w; h  = h  * original_h
        else:
            cx = cx * original_w / 640; cy = cy * original_h / 640
            w  = w  * original_w / 640; h  = h  * original_h / 640

        x1 = max(0, int(cx - w / 2)); y1 = max(0, int(cy - h / 2))
        x2 = min(original_w, int(cx + w / 2)); y2 = min(original_h, int(cy + h / 2))
        bw = x2 - x1; bh = y2 - y1

        # Filter ukuran
        if bw < original_w * MIN_BOX_RATIO: continue
        if bh < original_h * MIN_BOX_RATIO: continue
        if bw > original_w * MAX_BOX_RATIO and bh > original_h * MAX_BOX_RATIO: continue

        # ✅ Filter zona — pusat box harus ada di zona valid
        # Ini yang mencegah deteksi tangan/wajah di pinggir frame
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        if not (zone_x1 < cx_box < zone_x2 and zone_y1 < cy_box < zone_y2):
            continue

        all_boxes.append([x1, y1, bw, bh])
        all_confs.append(conf)
        all_classes.append(cls)

    if len(all_boxes) == 0:
        return {"label": "", "confidence": 0, "box": None}

    indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return {"label": "", "confidence": 0, "box": None}

    # Ambil deteksi dengan confidence tertinggi setelah NMS
    best_conf = 0
    best_idx  = -1
    for i in indices.flatten():
        if all_confs[i] > best_conf:
            best_conf = all_confs[i]
            best_idx  = i

    if best_idx == -1:
        return {"label": "", "confidence": 0, "box": None}

    x1, y1, bw, bh = all_boxes[best_idx]
    raw_label    = class_names[all_classes[best_idx]]
    display_name = LABEL_MAP.get(raw_label, raw_label)

    return {
        "label":      display_name,
        "confidence": round(best_conf, 2),
        "box":        [x1, y1, x1 + bw, y1 + bh]
    }

# ================= NOTIF WA =================
def kirim_wa(pesan):
    token = "GzUZtmAHjEx28eeCPNrZ"
    requests.post(
        "https://api.fonnte.com/send",
        headers={"Authorization": token},
        data={"target": "6285394312574", "message": pesan}
    )

# ================= API DETECT =================
@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'frame' not in request.files:
        return jsonify({"label": "", "confidence": 0, "box": None})
    file  = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"label": "", "confidence": 0, "box": None})
    result = detect_from_frame(frame)
    if result["label"] != "":
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "INSERT INTO deteksi (waktu, kategori, confidence) VALUES (?,?,?)",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result["label"], result["confidence"])
        )
        conn.commit()
        conn.close()
    return jsonify(result)

# ================= API UPDATE COUNTER =================
@app.route('/api/update', methods=['POST'])
def api_update():
    data     = request.json
    kategori = data.get('kategori', '')
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    if kategori == 'organik':
        c.execute("UPDATE counter SET organik=organik+1, last_detection='Organik' WHERE id=1")
    elif kategori == 'anorganik':
        c.execute("UPDATE counter SET anorganik=anorganik+1, last_detection='Anorganik' WHERE id=1")
    elif kategori == 'b3':
        c.execute("UPDATE counter SET b3=b3+1, last_detection='B3 - Berbahaya!' WHERE id=1")
    conn.commit()
    c.execute("SELECT organik, anorganik, b3, kapasitas FROM counter WHERE id=1")
    row = c.fetchone()

    if row[0] >= row[3]: kirim_wa("🚨 Tempat sampah ORGANIK penuh!")
    if row[1] >= row[3]: kirim_wa("🚨 Tempat sampah ANORGANIK penuh!")
    if row[2] >= row[3]: kirim_wa("🚨 Tempat sampah B3 penuh!")

    conn.close()
    return jsonify({"success": True, "penuh": {
        "organik":   row[0] >= row[3],
        "anorganik": row[1] >= row[3],
        "b3":        row[2] >= row[3],
    }})

# ================= API DATA =================
@app.route('/api/data')
def api_data():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT organik, anorganik, b3, kapasitas, last_detection FROM counter WHERE id=1")
    row  = c.fetchone()
    conn.close()
    return jsonify({
        "organik": row[0], "anorganik": row[1], "b3": row[2],
        "kapasitas": row[3], "last_detection": row[4],
        "penuh": {
            "organik":   row[0] >= row[3],
            "anorganik": row[1] >= row[3],
            "b3":        row[2] >= row[3],
        },
        "stream_url": stream_url_store["url"]
    })

# ================= API HISTORI =================
@app.route('/api/histori')
def api_histori():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT waktu, kategori, confidence FROM deteksi ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"waktu": r[0], "kategori": r[1], "confidence": r[2]} for r in rows])

# ================= API STATISTIK =================
@app.route('/api/statistik')
def api_statistik():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT DATE(waktu) as tgl, kategori, COUNT(*) as total
        FROM deteksi WHERE waktu >= DATE('now', '-7 days')
        GROUP BY tgl, kategori ORDER BY tgl
    """)
    rows  = c.fetchall()
    conn.close()
    hasil = {}
    for tgl, kategori, total in rows:
        if tgl not in hasil:
            hasil[tgl] = {"organik": 0, "anorganik": 0, "b3": 0}
        if "Organik" == kategori:
            hasil[tgl]["organik"] += total
        elif "Anorganik" == kategori:
            hasil[tgl]["anorganik"] += total
        elif "B3" in kategori:
            hasil[tgl]["b3"] += total
    return jsonify(hasil)

# ================= API RESET =================
@app.route('/api/reset', methods=['POST'])
def api_reset():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("UPDATE counter SET organik=0, anorganik=0, b3=0, last_detection='' WHERE id=1")
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ================= API REGISTER STREAM =================
@app.route('/api/register-stream', methods=['POST'])
def register_stream():
    data = request.json
    url  = data.get("stream_url", "")
    if url:
        stream_url_store["url"] = url
        print(f"📡 Stream URL terdaftar: {url}")
        return jsonify({"success": True, "stream_url": url})
    return jsonify({"success": False}), 400

# ================= API STREAM URL =================
@app.route('/api/stream-url')
def get_stream_url():
    return jsonify({"stream_url": stream_url_store["url"]})

# ================= HALAMAN WEB =================
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_web():
    api_reset()
    return redirect(url_for('dashboard'))

# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)