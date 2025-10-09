from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import cv2
from extract_features import extract_features
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='template')
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

@app.route('/model-info')
def model_info():
    """Endpoint to check model configuration"""
    return jsonify({
        'scaler_type': type(scaler).__name__,
        'scaler_feature_range': getattr(scaler, 'feature_range_', 'N/A'),
        'knn_neighbors': knn.n_neighbors,
        'classes': label_encoder.classes_.tolist(),
        'n_features': knn.n_features_in_
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save with original extension to preserve format
    file_ext = os.path.splitext(file.filename)[1]
    temp_path = f"temp{file_ext}"
    file.save(temp_path)
    
    try:
        # Read and verify image
        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Failed to read image")
        
        print(f"\n{'='*50}")
        print(f"Processing: {file.filename}")
        print(f"Original image shape: {img.shape}")
        
        # CRITICAL: Resize to match training images (128x128)
        img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        
        # Save resized image for feature extraction
        resized_temp_path = f"temp_resized{file_ext}"
        cv2.imwrite(resized_temp_path, img_resized)
        
        print(f"Resized to: {img_resized.shape}")
        print(f"Image dtype: {img_resized.dtype}")
        
        # Extract features from resized image
        features = extract_features(resized_temp_path)
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Features sample (first 10): {features[:10]}")
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        print(f"Scaled features range: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")
        print(f"Scaled features sample (first 10): {features_scaled[0][:10]}")
        
        # Get prediction with probability
        pred_encoded = knn.predict(features_scaled)[0]
        pred_proba = knn.predict_proba(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence and all predictions sorted
        confidence = max(pred_proba) * 100
        all_probs = {
            label_encoder.inverse_transform([i])[0]: prob * 100
            for i, prob in enumerate(pred_proba)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        print(f"\nPrediction: {pred_label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"All probabilities:")
        for label, prob in sorted_probs.items():
            print(f"  {label}: {prob:.2f}%")
        print(f"{'='*50}\n")
        
        # Clean up temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(resized_temp_path):
            os.remove(resized_temp_path)
        
        return jsonify({
            'success': True,
            'prediction': pred_label,
            'confidence': f"{confidence:.2f}%",
            'all_probabilities': {k: f"{v:.2f}%" for k, v in sorted_probs.items()}
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp files on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if 'resized_temp_path' in locals() and os.path.exists(resized_temp_path):
            os.remove(resized_temp_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
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