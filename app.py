from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import threading
import time
import shap
import pandas as pd

app = Flask(__name__)
model = joblib.load("C:\\Users\\USER\\OneDrive\\Desktop\\Churn App\\churn_app\\churn_xgb_model.pkl")

def delayed_redirect():
    time.sleep(300)
    with app.test_request_context():
        return redirect(url_for("home"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form["gender"]
        SeniorCitizen = int(request.form["SeniorCitizen"])
        Partner = request.form["Partner"]
        Dependents = request.form["Dependents"]
        tenure = float(request.form["tenure"])
        PhoneService = request.form["PhoneService"]
        MultipleLines = request.form["MultipleLines"]
        InternetService = request.form["InternetService"]
        OnlineSecurity = request.form["OnlineSecurity"]
        OnlineBackup = request.form["OnlineBackup"]
        DeviceProtection = request.form["DeviceProtection"]
        TechSupport = request.form["TechSupport"]
        StreamingTV = request.form["StreamingTV"]
        StreamingMovies = request.form["StreamingMovies"]
        Contract = request.form["Contract"]
        PaperlessBilling = request.form["PaperlessBilling"]
        PaymentMethod = request.form["PaymentMethod"]
        MonthlyCharges = float(request.form["MonthlyCharges"])
        TotalCharges = float(request.form["TotalCharges"])
        
        features = []
        
        features.append(1 if gender == "Male" else 0)
        features.append(SeniorCitizen)
        features.append(1 if Partner == "Yes" else 0)
        features.append(1 if Dependents == "Yes" else 0)
        features.append(tenure)
        features.append(1 if PhoneService == "Yes" else 0)
        features.append(1 if PaperlessBilling == "Yes" else 0)
        features.append(MonthlyCharges)
        features.append(TotalCharges)
    
        # MultipleLines 
        features.append(1 if MultipleLines == "No" else 0)
        features.append(1 if MultipleLines == "No phone service" else 0)
        features.append(1 if MultipleLines == "Yes" else 0)
        
        # InternetService
        features.append(1 if InternetService == "DSL" else 0)
        features.append(1 if InternetService == "Fiber optic" else 0)
        features.append(1 if InternetService == "No" else 0)
        
        # Online Security
        features.append(1 if OnlineSecurity == "No" else 0)                
        features.append(1 if OnlineSecurity == "No internet service" else 0)  
        features.append(1 if OnlineSecurity == "Yes" else 0)   
        
        # OnlineBackup
        features.append(1 if OnlineBackup == 'No' else 0)
        features.append(1 if OnlineBackup == 'No internet service' else 0)
        features.append(1 if OnlineBackup == 'Yes' else 0)

        # DeviceProtection
        features.append(1 if DeviceProtection == 'No' else 0)
        features.append(1 if DeviceProtection == 'No internet service' else 0)
        features.append(1 if DeviceProtection == 'Yes' else 0)

        # TechSupport
        features.append(1 if TechSupport == 'No' else 0)
        features.append(1 if TechSupport == 'No internet service' else 0)
        features.append(1 if TechSupport == 'Yes' else 0)

        # StreamingTV
        features.append(1 if StreamingTV == 'No' else 0)
        features.append(1 if StreamingTV == 'No internet service' else 0)
        features.append(1 if StreamingTV == 'Yes' else 0)

        # StreamingMovies
        features.append(1 if StreamingMovies == 'No' else 0)
        features.append(1 if StreamingMovies == 'No internet service' else 0)
        features.append(1 if StreamingMovies == 'Yes' else 0)

        # Contract
        features.append(0 if Contract == 'Month-to-month' else 1 if Contract == 'One year' else 2)

        # PaymentMethod
        features.append(1 if PaymentMethod == '2' else 0)
        features.append(1 if PaymentMethod == '3' else 0)
        features.append(1 if PaymentMethod == '0' else 0)
        features.append(1 if PaymentMethod == '1' else 0)

        final_features = [np.array(features)]

        prediction_proba = model.predict_proba(final_features)[0]

        churn_percentage = round(prediction_proba[1] * 100, 2)
        not_churn_percentage = round(prediction_proba[0] * 100, 2)
        theme = request.form.get("theme", "light")
        output = "✅ Customer is likely to Churn." if prediction_proba[1] > prediction_proba[0] else "✅ Customer is not likely to Churn."
        feature_names = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "PaperlessBilling",
            "MonthlyCharges", "TotalCharges", "MultipleLines_No", "MultipleLines_No phone service", "MultipleLines_Yes",
            "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
            "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
            "OnlineBackup_No", "OnlineBackup_No internet service", "OnlineBackup_Yes",
            "DeviceProtection_No", "DeviceProtection_No internet service", "DeviceProtection_Yes",
            "TechSupport_No", "TechSupport_No internet service", "TechSupport_Yes",
            "StreamingTV_No", "StreamingTV_No internet service", "StreamingTV_Yes",
            "StreamingMovies_No", "StreamingMovies_No internet service", "StreamingMovies_Yes",
            "Contract", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
        ]
        X_user = pd.DataFrame(final_features, columns=feature_names)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_user)
        shap_df = pd.DataFrame({"feature": feature_names, "shap_value": shap_values.values[0]})
        top_features = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index).head(5)
        reason_text = ", ".join(top_features['feature'].tolist())
        
        threading.Thread(target=delayed_redirect).start()
        
        return render_template('result.html', 
                               prediction_text=output,
                               churn_percentage = churn_percentage,
                               not_churn_percentage = not_churn_percentage,
                               theme = theme,
                               reason_text = reason_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

@app.route("/exit")
def exit_app():
    return render_template("exit.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
