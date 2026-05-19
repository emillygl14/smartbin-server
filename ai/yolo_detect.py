import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# ================= LOAD MODEL =================
interpreter = Interpreter(
    model_path="/home/pi/smartbin/models/best_float32.tflite"
)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================= SETTING =================
CONF_THRESHOLD = 0.55       # dinaikkan lagi
NMS_THRESHOLD  = 0.45
MIN_BOX_RATIO  = 0.05       # minimal 5% frame
MAX_BOX_RATIO  = 0.70

# Zona deteksi valid (tengah frame)
# objek sampah harus ada di area ini
ZONE_X1_RATIO = 0.10        # 10% dari kiri
ZONE_X2_RATIO = 0.90        # 90% dari kiri
ZONE_Y1_RATIO = 0.05        # 5% dari atas
ZONE_Y2_RATIO = 0.95        # 95% dari atas

class_names = ["Plastik", "anorganik", "b3", "organik"]

LABEL_MAP = {
    "Plastik":   ("Anorganik",       (0, 165, 255)),
    "anorganik": ("Anorganik",       (0, 165, 255)),
    "organik":   ("Organik",         (0, 255, 0)),
    "b3":        ("B3 - Berbahaya!", (0, 0, 255)),
}

# ================= DETECT FUNCTION =================
def detect(frame):
    original_h, original_w = frame.shape[:2]

    # zona valid dalam piksel
    zone_x1 = int(original_w * ZONE_X1_RATIO)
    zone_x2 = int(original_w * ZONE_X2_RATIO)
    zone_y1 = int(original_h * ZONE_Y1_RATIO)
    zone_y2 = int(original_h * ZONE_Y2_RATIO)

    # ================= PREPROCESS =================
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # ================= INFERENCE =================
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predictions = output[0].T

    # ================= KUMPULKAN SEMUA DETEKSI =================
    all_boxes   = []
    all_confs   = []
    all_classes = []

    for det in predictions:
        cx, cy, w, h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        class_scores  = det[4:]
        cls           = int(np.argmax(class_scores))
        conf          = float(class_scores[cls])

        if conf < CONF_THRESHOLD:
            continue

        # scale box
        if cx <= 2.0:
            cx = cx * original_w
            cy = cy * original_h
            w  = w  * original_w
            h  = h  * original_h
        else:
            cx = cx * original_w / 640
            cy = cy * original_h / 640
            w  = w  * original_w / 640
            h  = h  * original_h / 640

        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(original_w, int(cx + w / 2))
        y2 = min(original_h, int(cy + h / 2))

        bw = x2 - x1
        bh = y2 - y1

        # filter ukuran
        if bw < original_w * MIN_BOX_RATIO:
            continue
        if bh < original_h * MIN_BOX_RATIO:
            continue
        if bw > original_w * MAX_BOX_RATIO and bh > original_h * MAX_BOX_RATIO:
            continue

        # ✅ filter zona — pusat box harus ada di zona valid
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        if not (zone_x1 < cx_box < zone_x2 and zone_y1 < cy_box < zone_y2):
            continue

        all_boxes.append([x1, y1, bw, bh])
        all_confs.append(conf)
        all_classes.append(cls)

    # ================= TIDAK ADA DETEKSI =================
    if len(all_boxes) == 0:
        return {"label": "", "confidence": 0, "box": None, "color": (255, 255, 255)}

    # ================= NMS =================
    indices = cv2.dnn.NMSBoxes(
        all_boxes, all_confs, CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) == 0:
        return {"label": "", "confidence": 0, "box": None, "color": (255, 255, 255)}

    # ================= AMBIL TERBAIK SETELAH NMS =================
    best_conf = 0
    best_idx  = -1

    for i in indices.flatten():
        if all_confs[i] > best_conf:
            best_conf = all_confs[i]
            best_idx  = i

    if best_idx == -1:
        return {"label": "", "confidence": 0, "box": None, "color": (255, 255, 255)}

    x1, y1, bw, bh = all_boxes[best_idx]
    x2  = x1 + bw
    y2  = y1 + bh
    cls = all_classes[best_idx]

    raw_label           = class_names[cls]
    display_name, color = LABEL_MAP.get(raw_label, (raw_label, (255, 255, 255)))

    return {
        "label":      display_name,
        "confidence": best_conf,
        "box":        (x1, y1, x2, y2),
        "color":      color
    }