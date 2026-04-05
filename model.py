import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import pickle
import sys
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.model_selection import train_test_split
pd.options.mode.copy_on_write = True

#os.chdir(os.path.dirname(__file__))


df = pd.read_csv("data_sample/Superstore.csv", encoding= "latin1")

def clasificar_profit(x):
    if x > 0:
        return 1 # -> beneficio
    else:
        return 0 # -> pérdida 

df["Profit_Class"] = df["Profit"].apply(clasificar_profit)


def preprocess(df):

    df = df.copy()

    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

    # Drop columns
    cols_to_drop = [
        "row_id", "order_id", "product_id",
        "customer_id", "postal_code",
        "customer_name", "country", "profit"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Dates
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"])

    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_day"] = df["order_date"].dt.day

    if "ship_date" in df.columns:
        df["ship_year"] = df["ship_date"].dt.year
        df["ship_month"] = df["ship_date"].dt.month
        df["ship_day"] = df["ship_date"].dt.day

    # Delivery days
    if "ship_date" in df.columns and "order_date" in df.columns:
        df["delivery_days"] = (df["ship_date"] - df["order_date"]).dt.days

    df = df.drop(columns=["order_date", "ship_date"], errors="ignore")

    # Feature engineering
    if "sales" in df.columns and "delivery_days" in df.columns:
        df["impact_sales_delay"] = df["sales"] * df["delivery_days"]

    # Drop columns
    df = df.drop(columns=["state", "city"], errors="ignore")
    df = df.drop(columns=["sales", "ship_year", "ship_month", "ship_day", "order_day"], errors="ignore")

    # Log transform
    if "impact_sales_delay" in df.columns:
        df["impact_sales_delay"] = np.log1p(df["impact_sales_delay"])

    return df

preprocess_transformer = FunctionTransformer(preprocess, validate = False)


X = df.drop(columns=['Profit_Class'])
y = df['Profit_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


features_num = [
  'quantity',
  'discount',
  'order_year',
  'order_month',
  'delivery_days',
  'impact_sales_delay'
]

features_cat = ['ship_mode', 'segment', 'region', 'category', 'sub_category']

numeric_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy = "mean")),
    ("scaler", StandardScaler())
])

categoric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy= "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

catboost_pipeline = Pipeline([
    ("encoder", ce.CatBoostEncoder())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, features_num),
    ("cat", categoric_pipeline, features_cat),
    ("catboost", catboost_pipeline, ["product_name"])
])

model = Pipeline([
    ('preprocess', preprocess_transformer),
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Cross validation
cross_val_train_accuracy = cross_val_score(model, X_train, y_train, cv=4, scoring="accuracy")
cross_val_train_precision = cross_val_score(model, X_train, y_train, cv=4, scoring="precision")
cross_val_train_recall = cross_val_score(model, X_train, y_train, cv=4, scoring="recall")
cross_val_train_f1 = cross_val_score(model, X_train, y_train, cv=4, scoring="f1")

accuracy_cross_val = np.mean(cross_val_train_accuracy)
precision_cross_val = np.mean(cross_val_train_precision)
recall_cross_val = np.mean(cross_val_train_recall)
f1_cross_val = np.mean(cross_val_train_f1)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Train Target Mean:", y_train.mean())
print("Accuracy Cross: ", accuracy_cross_val)
print("Precision Cross: ", precision_cross_val)
print("Recall Cross: ", recall_cross_val)
print("F1 Cross: ", f1_cross_val)

print("**********")

print("Accuracy Test: ", accuracy_score(y_test, y_pred))
print("Precision Test: ", precision_score(y_test, y_pred))
print("Recall Test: ", recall_score(y_test, y_pred))
print("F1 Test: ", f1_score(y_test, y_pred))

model.fit(X, y)

with open('ad_model.pkl', 'wb') as f:
    pickle.dump(model, f)