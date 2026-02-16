from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load trained ML models
loan_model = joblib.load(os.path.join("models", "loan_model.pkl"))
fraud_model = joblib.load(os.path.join("models", "fraud_model.pkl"))

@app.route("/")
def home():
    return {"message": "AI-Based Smart Loan Approval Backend Running"}

@app.route("/apply-loan", methods=["POST"])
def apply_loan():
    data = request.json

    # ---------------- STEP 1: VALIDATION ----------------
    required_fields = ["income", "loan_amount", "credit_score", "tenure"]
    for field in required_fields:
        if field not in data:
            return jsonify({
                "status": "Error",
                "message": f"{field} is missing"
            }), 400

    # ---------------- STEP 2: PREPROCESSING ----------------
    df = pd.DataFrame([{
        "income": data["income"],
        "loan_amount": data["loan_amount"],
        "credit_score": data["credit_score"],
        "tenure": data["tenure"]
    }])

    print("\n📥 INPUT DATA")
    print(df)

    # ---------------- STEP 3: FRAUD DETECTION ----------------
    fraud_prediction = fraud_model.predict(df)[0]
    print("🚨 FRAUD PREDICTION:", fraud_prediction)

    if fraud_prediction == -1:
        print("❌ RESULT: FRAUD DETECTED")
        return jsonify({
            "status": "Rejected",
            "reason": "Fraud detected by ML model"
        })

    # ---------------- STEP 4: LOAN APPROVAL ----------------
    approval = loan_model.predict(df)[0]
    probability = loan_model.predict_proba(df)[0][1]

    print("📊 LOAN APPROVAL PREDICTION:", approval)
    print("📈 APPROVAL PROBABILITY:", probability)

    # ---------------- STEP 5: DECISION ENGINE ----------------
    if approval == 1:
        decision = "Approved"
        reason = "Low risk profile (ML decision)"
    else:
        decision = "Rejected"
        reason = "High risk profile (ML decision)"

    print("✅ FINAL DECISION:", decision)

    # ---------------- STEP 6: RESPONSE ----------------
    return jsonify({
        "status": decision,
        "approval_probability": round(float(probability), 2),
        "reason": reason
    })

if __name__ == "__main__":
    app.run(debug=True)
