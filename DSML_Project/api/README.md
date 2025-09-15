🛠️ Product Sales Forecasting API

This API serves predictions for product sales using CatBoostRegressor and SARIMAX models. It is built with Flask and supports JSON input/output for seamless integration.

📦 Project Setup
1. Clone or Download Repository
git clone <your-repo-url>
cd <your-repo-folder>

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

3. Install Dependencies

All required packages are listed in requirements.txt.

pip install -r requirements.txt

🚀 Running the API
1. Development Server (Flask)
python app.py


This will start the server on http://127.0.0.1:5000/.

2. Production Server (Gunicorn + Flask)
gunicorn -w 4 -b 0.0.0.0:5000 app:app


-w 4 → number of workers (adjust as per server capacity).

app:app → first app is the filename (app.py), second app is the Flask object inside.

📬 API Endpoints
POST /predict

Make a sales prediction.

Input (JSON via Postman or other client):
{
  "Store_Type": "S1",
  "Location_Type": "L2",
  "Region_Code": "R1",
  "Holiday": 0,
  "Discount": 1,
  "Order": 72
}

Output (JSON Response):
{
  "prediction": 42780.65
}

⚙️ Notes

Model files (catboost_model.pkl, sarimax_model.pkl) must be placed in the project root.

API uses CatBoostRegressor for tabular predictions and SARIMAX for time-series predictions.

Ensure the input JSON matches the expected feature names and datatypes.

✅ Requirements

See requirements.txt
.
Key dependencies:

Flask

Gunicorn

Pandas, Numpy, Scipy

Statsmodels

scikit-learn

CatBoost

Joblib