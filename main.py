from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Cargar modelo
with open('ad_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Landing page
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensaje": "API de predicción de pérdidas/ganancias",
        "uso": "/api/v1/predict?Quantity=2&Discount=0.2&Order_Year=2020&Order_Month=5&Delivery_Days=3&Impact_Sales_Delay=10&ProductName_TE=0.5&Ship_Mode=Second Class&Segment=Consumer&Region=West&Category=Furniture&Sub-Category=Chairs"
    })


# PREDICT
@app.route('/api/v1/predict', methods=['GET'])
def predict():

    # Variables numéricas
    quantity = request.args.get('Quantity', np.nan, type=float)
    discount = request.args.get('Discount', np.nan, type=float)
    order_year = request.args.get('Order_Year', np.nan, type=float)
    order_month = request.args.get('Order_Month', np.nan, type=float)
    delivery_days = request.args.get('Delivery_Days', np.nan, type=float)
    impact_sales_delay = request.args.get('Impact_Sales_Delay', np.nan, type=float)
    product_te = request.args.get('ProductName_TE', np.nan, type=float)

    # Variables categóricas

    ship_mode = request.args.get('Ship_Mode', None)
    segment = request.args.get('Segment', None)
    region = request.args.get('Region', None)
    category = request.args.get('Category', None)
    sub_category = request.args.get('Sub-Category', None)

    # DETECTAR NULOS
    missing = []

    # numéricos
    for name, val in [
        ('Quantity', quantity),
        ('Discount', discount),
        ('Order_Year', order_year),
        ('Order_Month', order_month),
        ('Delivery_Days', delivery_days),
        ('Impact_Sales_Delay', impact_sales_delay),
        ('ProductName_TE', product_te)
    ]:
        if np.isnan(val):
            missing.append(name)

    # categóricos
    for name, val in [
        ('Ship_Mode', ship_mode),
        ('Segment', segment),
        ('Region', region),
        ('Category', category),
        ('Sub-Category', sub_category)
    ]:
        if val is None:
            missing.append(name)

    # CREAR DATAFRAME
    input_data = pd.DataFrame({
        'Quantity': [quantity],
        'Discount': [discount],
        'Order_Year': [order_year],
        'Order_Month': [order_month],
        'Delivery_Days': [delivery_days],
        'Impact_Sales_Delay': [impact_sales_delay],
        'ProductName_TE': [product_te],
        'Ship_Mode': [ship_mode],
        'Segment': [segment],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [sub_category]
    })

    # PREDICCIÓN
    prediction = model.predict(input_data)

    resultado = "ganancia" if prediction[0] == 1 else "perdida"

    response = {
        "prediccion": resultado
    }

    if missing:
        response["warning"] = f"Valores faltantes: {', '.join(missing)}"

    return jsonify(response)


# RETRAIN 
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    return jsonify({
        "mensaje": "Reentrenamiento no implementado en esta versión"
    })


# RUN
if __name__ == '__main__':
    app.run(debug=True)