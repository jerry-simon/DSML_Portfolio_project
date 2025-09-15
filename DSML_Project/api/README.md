# 🛠️ Product Sales Forecasting API

This API serves predictions for product sales using CatBoostRegressor and SARIMAX models. It is built with Flask and supports JSON input/output for seamless integration.

# 📦 Project Setup
1. Clone or Download Repository
```bash
git clone https://github.com/jerry-simon/DSML_Portfolio_project/tree/48507f820cb1908962c8e3354e806288743f08e0/DSML_Project
cd DSML_Project
```

3. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

5. Install Dependencies

All required packages are listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```

🚀 Running the API

1. Development Server (Flask)
```bash
python ProductSalesForecasting.py #VS code editor
flask --app ProductSalesForecasting.py run #PyCharm code
```

Make sure you use the correct command to run the Flask application, depending on your code editor

This will start the server on http://127.0.0.1:5000/.

2. Production Server (Gunicorn + Flask) # Optional

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app: app
```

-w 4 → number of workers (adjust as per server capacity).

app: app → first app is the filename (app.py), second app is the Flask object inside.

📬 API Endpoints

POST /PREDICT

Make a sales prediction.

a) CatBoost Regression

**Possible question: What will be the predicted sales on 06-06-2019 for Store ID 100 (Store Type S3, Location Type L2, Region R3), if no holiday is observed and discounts are offered?**

Input (JSON via Postman or other client):
```bash
{
    "Store_id" : 100,
    "Store_Type" : "S3",
    "Location_Type" : "L2",
    "Region_Code" : "R3",
    "Date" : "06-06-2019",
    "Holiday" : 0,
    "Discount" : "Yes"
}
```

Output (JSON Response):
```bash
{
    "1_model": "CatBoost",
    "2_prediction": 57100.07307448184,
    "3_lower_CI_95": 40619.86363142608,
    "4_upper_CI_95": 73580.2825175376,
    "5_CI_method": "residual_std"
}
```

b) SARIMAX Regression

**Possible question: What will be the predicted sales on 06-06-2019 if 30% of stores are on holiday, 60% offer a discount, and the remaining 10% operate normally?**

Input (JSON via Postman or other client):
```bash
{
    "Date" : "06-06-2019",
    "Holiday" : 0.30,
    "Discount" : 0.60
}
```

Output (JSON Response):
```bash
{
    "06-06-2019": 19808791.291501004,
    "lower_bound": 14531814.297463099,
    "model": "SARIMAX",
    "upper_bound": 25085768.28553891
}
```
c) SARIMAX Regression

**Possible question: What will be the predicted Sales on 06-06-2019?**

Input (JSON via Postman or other client):
```bash
{
    "Date" : "06-06-2019"
}
```

Output (JSON Response):
```bash
{
    "06-06-2019": 18042535.32550656,
    "lower_bound": 12765558.331468655,
    "model": "SARIMAX",
    "upper_bound": 23319512.319544464
}
```
⚙️ Notes

**Model files (catboost_model.pkl, sarimax_model.pkl, and so on) must be placed in the project root and update the file path accordingly.**

API uses CatBoostRegressor for tabular predictions and SARIMAX for time-series predictions.

Ensure the input JSON matches the expected feature names, datatypes, and the length of the JSON input should be either of the following

a. length = 1, make sure Date is mandatory.

b. length = 3, make sure Date, Holiday (float) and Discount (float) are mandatory.

c. length = 6 or 7, make sure Store_id (int), Store_Type (str), Location_Type (str), Region_Code (str), Date (str), Holiday (int), Discount (str) ['Yes' or 'No'], Order (int) (optional)

# ✅ Requirements

See requirements.txt

# Key dependencies:

Flask

Gunicorn

Pandas, Numpy, Scipy

Statsmodels

scikit-learn

CatBoost

Joblib


