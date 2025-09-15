🛒 Product Sales Forecasting

This project forecasts daily product sales using a combination of machine learning and time-series forecasting techniques.
It leverages both structured features (store, region, discount, holiday, etc.) and historical patterns (trend, seasonality) to generate reliable sales predictions.

📂 Project Structure

TRAIN.csv – Training dataset with historical sales and orders (used for model development).

TEST_FINAL.csv – Test dataset without target values (used for final predictions).

custom_transformers/ – Custom Python scripts for feature engineering and preprocessing.

ProductSalesForecasting.ipynb – Jupyter Notebook containing the full workflow (EDA → Feature Engineering → Modeling → Evaluation → Prediction).

⚙️ Prerequisites

Before running the notebook, ensure the following:

Python: Version 3.8+

Jupyter Notebook or JupyterLab installed

Required Python libraries (install via requirements.txt):

pip install -r requirements.txt


If requirements.txt is not included, install these manually:

pip install numpy pandas matplotlib seaborn scikit-learn catboost statsmodels

🚀 How to Run

Clone or download this repository (or create a local folder).

Place the following files inside the project directory:

TRAIN.csv

TEST_FINAL.csv

custom_transformers/ (folder)

Open the notebook:

jupyter notebook ProductSalesForecasting.ipynb


Run the cells sequentially. The workflow covers:

Data Preprocessing & Cleaning

Exploratory Data Analysis (EDA)

Outlier Handling & Feature Engineering

Model Training (Linear Regression, RandomForestRegressor, GradientBoostingRegressor, CatBoostRegressor, SARIMAX)

Model Evaluation (R², RMSE, MAE, MAPE)

Predictions on TEST_FINAL.csv

📊 Key Outputs

EDA Visualizations – Distribution plots, trends, and risk alerts.

Outlier Analysis – Identifying real-world sales spikes from discounts and holidays.

Machine Learning Models – Comparative evaluation of regression models, with CatBoostRegressor as the final selection.

Time-Series Model – SARIMAX to capture seasonality and exogenous factors.

Final Predictions – Sales forecasts for unseen data (TEST_FINAL.csv).

🛠️ Troubleshooting

ModuleNotFoundError: No module named 'catboost'
→ Run pip install catboost

ModuleNotFoundError: No module named 'statsmodels'
→ Run pip install statsmodels

FileNotFoundError: TRAIN.csv or TEST_FINAL.csv not found
→ Ensure both datasets are in the same directory as the notebook.

If you face Jupyter issues, run:

pip install notebook jupyterlab

📝 Notes

TRAIN.csv is used for model training & validation.

TEST_FINAL.csv is used only for prediction (to mimic real-world unseen data).

Outliers are capped thoughtfully (99th percentile) to preserve business signals.

Models are compared on accuracy, robustness, and ability to generalize.

✅ Conclusion

This project demonstrates a complete end-to-end sales forecasting pipeline, blending traditional time-series approaches (SARIMAX) with advanced ML techniques (CatBoost).
The final model is designed to be production-ready and adaptable for API integration.