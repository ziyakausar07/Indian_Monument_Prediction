import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================
# CONFIG
# ========================
IMG_SIZE = 300
TFLITE_PATH = "model.tflite"
BEST_MODEL_PATH = "best_model.h5"
MODEL_PATH = "monument_model.h5"

# ========================
# LOAD MODEL (TFLite or Keras)
# ========================
interpreter = None
input_details = None
output_details = None
keras_model = None

if os.path.exists(TFLITE_PATH):
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path=TFLITE_PATH)
    except Exception:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[SUCCESS] TFLite Model Loaded from {TFLITE_PATH}")
elif os.path.exists(BEST_MODEL_PATH):
    from tensorflow.keras.models import load_model
    keras_model = load_model(BEST_MODEL_PATH, compile=False)
    print(f"[SUCCESS] Keras Model Loaded from {BEST_MODEL_PATH}")
elif os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model
    keras_model = load_model(MODEL_PATH, compile=False)
    print(f"[SUCCESS] Keras Model Loaded from {MODEL_PATH}")
else:
    print("[ERROR] No model found")

model_loaded = (interpreter is not None or keras_model is not None)

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
except Exception:
    monument_details = {}

# ========================
# PREDICTION FUNCTION
# ========================
def predict(img_path):
    if not model_loaded:
        raise ValueError("Model not loaded")

    # 🔹 Load & resize image using PIL
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)

    # 🔹 Prediction (TFLite or Keras)
    if interpreter is not None:
        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        pred = keras_model.predict(img_arr)[0]

    # ========================
    # DEBUG (IMPORTANT)
    # ========================
    print("\n--- FULL PREDICTIONS ---")
    for i, prob in enumerate(pred):
        print(idx_to_class.get(i, f"Class {i}"), ":", round(prob * 100, 2), "%")
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
        label = f"Confused ({idx_to_class.get(top1_idx, 'Unknown')} / {idx_to_class.get(top2_idx, 'Unknown')})"

    else:
        label = idx_to_class.get(top1_idx, "Unknown")

    print("Final Label:", label)
    print("Confidence:", confidence)

    return label, confidence


# ========================
# HOME
# ========================
@app.route('/')
def index():
    return render_template('index.html', missing_model=not model_loaded)


# ========================
# PREDICT ROUTE
# ========================
@app.route('/predict', methods=['POST'])
def upload():
    file = request.files.get('file')

    if file is None or file.filename == "":
        return "No file uploaded", 400

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)