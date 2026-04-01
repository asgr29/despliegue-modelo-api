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

    df.columns = df.columns.str.replace(" ", "_")
    # Drop columns
    cols_to_drop = [
        "Row_ID", "Order_ID", "Product_ID",
        "Customer_ID", "Postal_Code",
        "Customer_Name", "Country","Profit"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Dates
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])

    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Day"] = df["Order_Date"].dt.day

    df["Ship_Year"] = df["Ship_Date"].dt.year
    df["Ship_Month"] = df["Ship_Date"].dt.month
    df["Ship_Day"] = df["Ship_Date"].dt.day

    # Delivery days
    df["Delivery_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days

    df = df.drop(columns=["Order_Date", "Ship_Date"], errors="ignore")
    
    # Feature engineering
    df["Impact_Sales_Delay"] = df["Sales"] * df["Delivery_Days"]

    # Drop columns
    df = df.drop(columns=["State", "City"], errors="ignore")
    df = df.drop( columns = ["Sales","Ship_Year", "Ship_Month", "Ship_Day", "Order_Day"])
    # Log transform
    df["Impact_Sales_Delay"] = np.log1p(df["Impact_Sales_Delay"])

    return df

df = preprocess(df)

X = df.drop(columns=['Profit_Class'])
y = df['Profit_Class']

te_product = ce.CatBoostEncoder(cols=["Product_Name"])
te_product.fit(X["Product_Name"], y)

# Transformar
X["ProductName_TE"] = te_product.transform(X["Product_Name"])

X = X.drop(columns=["Product_Name"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


features_num = [
  'Quantity',
  'Discount',
  'Order_Year',
  'Order_Month',
  'Delivery_Days',
  'Impact_Sales_Delay',
  'ProductName_TE'
]

features_cat = ['Ship_Mode', 'Segment', 'Region', 'Category', 'Sub-Category']

numeric_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy = "mean")),
    ("scaler", StandardScaler())
])

categoric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy= "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, features_num),
    ("cat", categoric_pipeline, features_cat)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LogisticRegression())
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