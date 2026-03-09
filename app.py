from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'ford_model.pkl')
META_PATH  = os.path.join(BASE_DIR, 'model', 'metadata.json')

bundle = joblib.load(MODEL_PATH)
rf_model  = bundle['model']
scaler    = bundle['scaler']
le_model  = bundle['le_model']
le_trans  = bundle['le_trans']
le_fuel   = bundle['le_fuel']

with open(META_PATH) as f:
    metadata = json.load(f)


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), path)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify(metadata)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        car_model    = data['model']
        year         = int(data['year'])
        transmission = data['transmission']
        mileage      = int(data['mileage'])
        fuel_type    = data['fuelType']
        tax          = float(data['tax'])
        mpg          = float(data['mpg'])
        engine_size  = float(data['engineSize'])

        # Encode categoricals
        model_enc = le_model.transform([car_model])[0]
        trans_enc = le_trans.transform([transmission])[0]
        fuel_enc  = le_fuel.transform([fuel_type])[0]

        features = np.array([[model_enc, year, trans_enc, mileage, fuel_enc, tax, mpg, engine_size]])
        features_scaled = scaler.transform(features)
        predicted_price = rf_model.predict(features_scaled)[0]

        # Confidence band ±8%
        low  = predicted_price * 0.92
        high = predicted_price * 1.08

        return jsonify({
            'success': True,
            'price': round(float(predicted_price), 2),
            'low':   round(float(low), 2),
            'high':  round(float(high), 2),
            'model': car_model,
            'year':  year
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare multiple car configurations at once."""
    try:
        configs = request.get_json()  # list of car configs
        results = []
        for data in configs:
            model_enc = le_model.transform([data['model']])[0]
            trans_enc = le_trans.transform([data['transmission']])[0]
            fuel_enc  = le_fuel.transform([data['fuelType']])[0]
            features  = np.array([[model_enc, int(data['year']), trans_enc,
                                   int(data['mileage']), fuel_enc,
                                   float(data['tax']), float(data['mpg']),
                                   float(data['engineSize'])]])
            features_scaled = scaler.transform(features)
            price = float(rf_model.predict(features_scaled)[0])
            results.append({'label': data.get('label', data['model']), 'price': round(price, 2)})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("🚗 Ford Car Price Predictor API running on http://localhost:5000")
    app.run(debug=True, port=5000)