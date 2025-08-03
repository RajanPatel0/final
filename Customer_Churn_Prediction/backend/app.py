from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load model & encoders
model = joblib.load("customer_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert JSON to DataFrame
    input_df = pd.DataFrame([data])
    
    # Encode using saved label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform([input_df[col][0]])

    # Ensure correct order of columns
    required_columns = ['Age', 'Gender', 'State', 'ServiceType', 'MonthlyCharges',
                        'TenureMonths', 'InternetUsageGB', 'CallDropsPerMonth',
                        'ComplaintsLast6Months', 'IsActive', 'PaymentMethod']
    
    prediction = model.predict(input_df[required_columns])[0]
    return jsonify({'prediction': 'Yes, Possibility Of Csx Churn' if prediction == 1 else 'No, Not Churn'})

if __name__ == '__main__':
    app.run(debug=True)
