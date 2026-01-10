from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "smartleak_xgb.pkl")

app = Flask(__name__)

# Load trained model
model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "Pressure", "Flow_Rate", "Temperature",
    "Vibration", "RPM", "Operational_Hours",
    "Zone", "Block", "Pipe",
    "Latitude", "Longitude"
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json

    # Validate input
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    df = pd.DataFrame([payload])

    # Predict
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return jsonify({
        "leak_probability": round(float(prob), 4),
        "leak_detected": bool(pred)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)