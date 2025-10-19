from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from extract_features import extract_features
from flask_cors import CORS
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# ============================
# Load trained KNN model
# ============================
with open("knn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Extract objects from pickle
knn = model_data['model']                # trained KNN model
scaler = model_data['scaler']            # MinMaxScaler
label_encoder_classes = model_data['classes']  # class names

# Create LabelEncoder and set its classes
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# ============================
# Routes
# ============================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = "temp.jpg"
    file.save(temp_path)

    try:
        # ----------------------------
        # Extract features from uploaded image
        # ----------------------------
        features = extract_features(temp_path)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # ----------------------------
        # Predict class
        # ----------------------------
        pred_encoded = knn.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        # Replace any old "Diseased" label with "Unhealthy"
        if "Diseased" in pred_label:
            pred_label = pred_label.replace("Diseased", "Unhealthy")

        # ----------------------------
        # Predict probabilities
        # ----------------------------
        prob_array = knn.predict_proba(features_scaled)[0]
        class_labels = label_encoder.inverse_transform(np.arange(len(prob_array)))

        prob_dict = {}
        for cls, prob in zip(class_labels, prob_array):
            cls_name = cls.replace("Diseased", "Unhealthy") if "Diseased" in cls else cls
            prob_dict[cls_name] = round(prob * 100)

        # ----------------------------
        # Print results to terminal
        # ----------------------------
        print(f"📌 Image uploaded: {file.filename}")
        print(f"Top Prediction: {pred_label}")
        print("Class probabilities:")
        for cls, prob in prob_dict.items():
            print(f"  {cls}: {prob}%")
        print("-" * 40)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Return JSON response
        return jsonify({
            'success': True,
            'prediction': pred_label,
            'probabilities': prob_dict
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
  