from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load saved model and encoder
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("encoder.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")
numeric_cols = joblib.load("numeric_cols.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        new_data = pd.DataFrame([data])

        # Separate categorical and numeric values
        new_cat = encoder.transform(new_data[categorical_cols])
        new_num = new_data[numeric_cols].values

        # Final input format
        new_final = np.hstack([new_cat, new_num])

        # Model prediction
        prediction = model.predict(new_final)[0]

        return jsonify({"price": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Car Price Prediction API is running!"})

if __name__ == "__main__":
    app.run(debug=True)
