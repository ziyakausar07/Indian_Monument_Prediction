import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input  # [IMPORTANT]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================
# LOAD MODEL
# ========================
MODEL_PATH = "monument_model.h5"
BEST_MODEL_PATH = "best_model.h5"

active_model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH

model = None
if os.path.exists(active_model_path):
    model = load_model(active_model_path, compile=False)
    print(f"[SUCCESS] Model Loaded from {active_model_path}")
else:
    print("[ERROR] No model found")

# ========================
# LOAD CLASS INDICES
# ========================
with open("class_indices.json") as f:
    class_indices = json.load(f)

# [FIX] mapping
idx_to_class = {int(v): k for k, v in class_indices.items()}

# ========================
# LOAD DETAILS
# ========================
try:
    with open("monument_details.json") as f:
        monument_details = json.load(f)
except:
    monument_details = {}

# ========================
# CONFIG
# ========================
IMG_SIZE = 300

# ========================
# PREDICTION FUNCTION
# ========================
def predict(img_path):
    if model is None:
        raise ValueError("Model not loaded")

    # 🔹 Load image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img.convert("RGB")

    # 🔹 Convert to array
    img = img_to_array(img)

    # [IMPORTANT] FIX (EfficientNet preprocessing)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # 🔹 Prediction
    pred = model.predict(img)[0]

    # ========================
    # DEBUG (IMPORTANT)
    # ========================
    print("\n--- FULL PREDICTIONS ---")
    for i, prob in enumerate(pred):
        print(idx_to_class[i], ":", round(prob * 100, 2), "%")
    print("-----------------------\n")

    # ========================
    # TOP-2 LOGIC
    # ========================
    top2 = pred.argsort()[-2:][::-1]

    top1_idx = int(top2[0])
    top2_idx = int(top2[1])

    top1_conf = float(pred[top1_idx])
    top2_conf = float(pred[top2_idx])

    # ========================
    # SMART DECISION
    # ========================
    confidence = top1_conf * 100

    # 🔴 LOW CONFIDENCE → UNKNOWN
    if confidence < 50:
        label = "Unknown"

    # 🟡 SIMILAR CONFUSION (only close values)
    elif (top1_conf - top2_conf) < 0.15:
        label = f"Confused ({idx_to_class[top1_idx]} / {idx_to_class[top2_idx]})"

    else:
        label = idx_to_class[top1_idx]

    print("Final Label:", label)
    print("Confidence:", confidence)

    return label, confidence


# ========================
# HOME
# ========================
@app.route('/')
def index():
    return render_template('index.html', missing_model=(model is None))


# ========================
# PREDICT ROUTE
# ========================
@app.route('/predict', methods=['POST'])
def upload():
    file = request.files.get('file')

    if file is None or file.filename == "":
        return "No file uploaded"

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    label, confidence = predict(path)

    details = monument_details.get(label, {})

    return render_template(
        "result.html",
        monument=label.replace("_", " ").title(),
        confidence=round(confidence, 2),
        filename=filename,
        details=details
    )


# ========================
# RUN
# ========================
if __name__ == "__main__":
    app.run(debug=True)