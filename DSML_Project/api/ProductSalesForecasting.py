from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn import set_config
set_config(transform_output="pandas")

# Import custom transformers so unpickling works
from custom_transformers import CalendarFeatureEngineer, AOVFeatureMaker, ColumnDropper, DataTypeCaster

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "<p><H3>Welcome to product sales forecasting app</H3></p>"

# -----------------------------
# Load models & preprocessors
# -----------------------------
try:
    prep = joblib.load("dsml_project/models/preprocessor.pkl")
    ts_model = joblib.load("dsml_project/models/sarimax_model.pkl")
    cats_model = joblib.load("dsml_project/models/cat_model.pkl")
    resid = joblib.load("dsml_project/models/cat_model_CI.pkl")

    print("✅ Models and preprocessor loaded successfully")
except Exception as e:
    print(f"❌ Error loading models/preprocessor: {e}")
    prep, ts_model, cats_model = None, None, None

def sanitize_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Convert Discount to categorical ("Yes"/"No") safely
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].map(
            {"Yes": "Yes", "No": "No", 1: "Yes", 0: "No"}
        ).fillna("No")  # default to "No"

    # Convert Holiday to numeric or categorical depending on your training
    if "Holiday" in df.columns:
        df["Holiday"] = pd.to_numeric(df["Holiday"], errors="coerce").fillna(0)

    # Convert categorical fields to category dtype
    categorical_cols = ["Store_Type", "Location_Type", "Region_Code", "Discount"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Parse Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return df

# -----------------------------
# Helper Functions
# -----------------------------
def predict_with_sarimax(Date, Holiday, Discount):
    exog_df = pd.DataFrame({
        "holiday": [Holiday],
        "discount": [Discount]
    })

    forecast = ts_model.get_forecast(steps=1, exog=exog_df)
    p = forecast.predicted_mean.tolist()[0]

    # Confidence Interval
    conf_int = forecast.conf_int(alpha=0.05)  # 95% CI
    lower, upper = conf_int.iloc[0]

    return jsonify({
        "model": "SARIMAX",
        Date : p,
        "lower_bound": float(lower),
        "upper_bound": float(upper)
    })

def predict_with_catboost(data: dict):
    # Sanitize raw input
    df = sanitize_input(data)

    # Apply preprocessing
    df_preprocessed = prep.transform(df)
    print("Outcoming preprocessing",df_preprocessed)
    # Predict
    pred = cats_model.predict(df_preprocessed).tolist()[0]
    return pred

# -----------------------------
# API Route
# -----------------------------
@app.route("/PREDICT", methods=["POST"])
def predictions():
    # Parse JSON safely
    data = request.get_json(force=True)  # force=True ensures JSON parsing even if headers missing

    if not data:
        return jsonify({"error": "No JSON payload received"}), 400

    print("Incoming JSON:", data)  # <-- Debug print

    length = len(data)

    try:
        # --- SARIMAX single-date case ---
        if length == 1:
            Date = data.get("Date", "")
            Holiday, Discount = 0.0, 0.0
            prediction = predict_with_sarimax(Date, Holiday, Discount)
            return prediction

        # --- SARIMAX with exogenous inputs ---
        elif length <= 3:
            Date = data.get("Date", "")
            Holiday = data.get("Holiday", 0.0)
            Discount = data.get("Discount", 0.0)

            prediction = predict_with_sarimax(Date, Holiday, Discount)
            return prediction

        # --- CatBoost store-level case ---
        elif length > 3:
            # Convert dict -> DataFrame inside predict_with_catboost
            prediction = predict_with_catboost(data)
            margin = 1.96 * resid

            return jsonify({
                "1_model": "CatBoost",
                "2_prediction": prediction,
                "3_lower_CI_95": prediction - margin,
                "4_upper_CI_95": prediction + margin,
                "5_CI_method": "residual_std"
            })

        else:
            return jsonify({"error": "Invalid input"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
