from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from extract_features import extract_features
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model
with open("knn_features_model.pkl", "rb") as f:
    model_data = pickle.load(f)

knn = model_data['knn']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save temporarily
    temp_path = "temp.jpg"
    file.save(temp_path)
    
    try:
        # Extract features and predict
        features = extract_features(temp_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred_encoded = knn.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'prediction': pred_label
        })
    
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)