import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re


# -------------------------
# 1. Data Type Caster
# -------------------------
class DataTypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self,
                 store_col="Store_Type",
                 loc_col="Location_Type",
                 region_col="Region_Code",
                 holiday_col="Holiday",
                 discount_col="Discount",
                 date_col="Date"):
        self.store_col = store_col
        self.loc_col = loc_col
        self.region_col = region_col
        self.holiday_col = holiday_col
        self.discount_col = discount_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert to categorical dtype
        for col in [self.store_col, self.loc_col, self.region_col, self.holiday_col, self.discount_col]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # Convert Date
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")

        return X


# -------------------------
# 2. Calendar Feature Engineer
# -------------------------
class CalendarFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="Date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            X["Day_of_week"] = X[self.date_col].dt.dayofweek.fillna(-1).astype(int)
            X["Month"] = X[self.date_col].dt.month.fillna(-1).astype(int)
            X["Is_weekend"] = (X[self.date_col].dt.dayofweek >= 5).fillna(False).astype(int)
            X["Is_month_end"] = X[self.date_col].dt.is_month_end.fillna(False).astype(int)
            X["Is_quarter_end"] = X[self.date_col].dt.is_quarter_end.fillna(False).astype(int)
        return X


# -------------------------
# 3. AOV Feature Maker
# -------------------------
class AOVFeatureMaker(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols=None):
        # add Store_id to the default group_cols
        self.group_cols = group_cols or ['Store_id', 'Store_Type', 'Location_Type', 'Region_Code', 'Holiday', 'Discount']
        self.maps_ = {}
        self.global_aov_ = None

    def fit(self, X, y=None):
        dfx = X.copy()
        # ensure numerics for AOV
        dfx['Sales'] = pd.to_numeric(dfx['Sales'], errors='coerce')
        dfx['Order'] = pd.to_numeric(dfx['Order'], errors='coerce')

        # avoid division by zero
        aov = dfx['Sales'] / dfx['Order'].replace(0, np.nan)
        self.global_aov_ = float(aov.mean(skipna=True))

        # build per-category maps (keys as strings for robustness)
        for col in self.group_cols:
            m = (aov.groupby(dfx[col].astype(str)).mean()).to_dict()
            self.maps_[col] = m
        return self

    def transform(self, X):
        dfx = X.copy()
        for col in self.group_cols:
            s = dfx[col].astype(str).map(self.maps_[col])          # map AOV to group
            s = pd.to_numeric(s, errors='coerce')                  # force float
            dfx[f'AOV_{col}'] = s.fillna(self.global_aov_).astype('float64')
        return dfx


# -------------------------
# 4. Column Dropper
# -------------------------
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        if drop_cols is None:
            drop_cols = ["Sales", "Order", "Store_id", "Date"]
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")