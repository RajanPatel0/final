from flask import Flask, request, jsonify
import joblib
import json
from preprocess import transform_input

app = Flask(__name__)
model = joblib.load("model.pkl")
with open("columns.json", "r") as f:
    location_columns = json.load(f)["location_columns"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = transform_input(data, location_columns)
    prediction = model.predict([features])
    return jsonify({"predicted_price_lakhs": round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
