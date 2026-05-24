# Customer Churn Prediction (Churn AI) 📊

A full-stack machine learning web application built with **Flask** and **XGBoost** that predicts whether a telecom customer is likely to churn. The application not only provides the probability of a customer churning, but also utilizes **SHAP** (SHapley Additive exPlanations) to explain the *reasons* behind the prediction, giving actionable insights into the top driving factors for churn or retention.

## Features ✨
- **Interactive Web Interface**: A clean, responsive form to input customer details.
- **Real-Time Predictions**: Calculates the percentage probability of Churn vs. Retention.
- **Visualizations**: Interactive pie chart displaying the prediction breakdown using Chart.js.
- **Explainable AI**: Highlights the top 5 most impactful features influencing the prediction using SHAP values.
- **Theming**: Built-in toggle for Light and Dark modes.
- **Deployment Ready**: Fully configured for seamless deployment on both Render and Vercel.

## Tech Stack 🛠️
- **Backend**: Python, Flask
- **Machine Learning**: XGBoost, Scikit-learn, SHAP, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Deployment servers**: Gunicorn (for Render), `@vercel/python` (for Vercel)

## Project Structure 📁
```text
.
├── churn_app/
│   ├── static/
│   │   └── style.css            # Stylesheets
│   ├── templates/
│   │   ├── index.html           # Input form UI
│   │   ├── result.html          # Prediction results & charts UI
│   │   └── exit.html            # Exit page UI
│   ├── __init__.py              # Marks directory as a package
│   ├── app.py                   # Main Flask application logic
│   ├── churn_xgb_model.pkl      # Pre-trained XGBoost Model
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original Dataset
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── vercel.json                  # Vercel deployment configuration
├── wsgi.py                      # WSGI entry point for production
└── README.md                    # Project documentation
```

## Running Locally 💻

1. **Clone the repository**:
   ```bash
   git clone https://github.com/debanjan-mondal-2005/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**:
   Make sure you have Python installed. It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   python wsgi.py
   ```

4. **Access the web app**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Deployment 🚀

This project is pre-configured to be deployed easily on modern platforms.

### Vercel
1. Link your GitHub repository in the Vercel Dashboard.
2. Vercel will automatically detect the `vercel.json` file.
3. Deploy! No further configuration is needed.

### Render
1. Create a new **Web Service** in the Render Dashboard and link your GitHub repository.
2. Set the environment to **Python 3**.
3. Set the Build Command to: `pip install -r requirements.txt`
4. Set the Start Command to: `gunicorn wsgi:app`

## License 📄
This project is open-source and available under standard open source licenses.
