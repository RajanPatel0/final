from flask import Flask, request, jsonify
import joblib
import json
from preprocess import transform_input
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load model and columns
model = joblib.load("model.pkl")
with open("columns.json", "r") as f:
    location_columns = json.load(f)["location_columns"]

@app.route('/locations', methods=['GET'])
def get_locations():
    locations = location_columns[4:]  # Assuming first 4 columns are non-location
    return jsonify({'locations': [loc.title() for loc in locations]})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = transform_input(data, location_columns)
    print("📊 Input feature vector length:", len(features))  # Should be 1202

    prediction = model.predict([features])
    return jsonify({"predicted_price_lakhs": round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
