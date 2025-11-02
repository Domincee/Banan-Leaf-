from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from extract_features import extract_features
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)
CORS(app)

# Config
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 10)) * 1024 * 1024
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "knn_model.pkl")

# ============================
# Load trained KNN model
# ============================
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

knn = model_data["model"]                   # trained KNN model
scaler = model_data["scaler"]               # MinMaxScaler
label_encoder_classes = model_data["classes"]  # class names

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_encoder_classes)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ============================
# Routes
# ============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    return {"ok": True}, 200

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower() or ".jpg"
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)

    try:
        # Extract features and scale
        features = extract_features(temp_path)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict class
        pred_encoded = knn.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        if "Diseased" in pred_label:
            pred_label = pred_label.replace("Diseased", "Unhealthy")

        # Predict probabilities (use knn.classes_ for correct order)
        prob_array = knn.predict_proba(features_scaled)[0]
        prob_class_labels = label_encoder.inverse_transform(knn.classes_)
        prob_dict = {}
        for cls, prob in zip(prob_class_labels, prob_array):
            cls_name = cls.replace("Diseased", "Unhealthy") if "Diseased" in cls else cls
            prob_dict[cls_name] = int(round(prob * 100))

        print(f"Image uploaded: {file.filename} -> {pred_label} | {prob_dict}")

        return jsonify({
            "success": True,
            "prediction": pred_label,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)