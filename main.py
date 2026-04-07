from flask import Flask, jsonify, request
import os
import pickle
from model import preprocess
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Cargar modelo
with open('ad_model.pkl', 'rb') as f:
    model = pickle.load(f)

def train_model(df):

    def clasificar_profit(x):
        return 1 if x > 0 else 0

    df["Profit_Class"] = df["Profit"].apply(clasificar_profit)

    X = df.drop(columns=["Profit_Class"])
    y = df["Profit_Class"]

    global model
    model.fit(X, y)

    with open('ad_model_v2.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model    

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan    


# Landing page
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensaje": "API de predicción de pérdidas/ganancias",
        "uso": "/api/v1/predict?Quantity=2&Discount=0.2&Order_Year=2020&Order_Month=5&Delivery_Days=3&Impact_Sales_Delay=10&Product_Name=Office Chair&Ship_Mode=Second Class&Segment=Consumer&Region=West&Category=Furniture&sub_category=Chairs"
    })


# PREDICT
@app.route('/api/v1/predict', methods=['GET'])
def predict():

    args = {k.lower(): v for k, v in request.args.items()}

    # Variables numéricas
    quantity = to_float(args.get('quantity', np.nan))
    discount = to_float(args.get('discount', np.nan))
    order_year = to_float(args.get('order_year', np.nan))
    order_month = to_float(args.get('order_month', np.nan))
    delivery_days = to_float(args.get('delivery_days', np.nan))
    impact_sales_delay = to_float(args.get('impact_sales_delay', np.nan))
    

    # Variables categóricas

    ship_mode = args.get('ship_mode')
    segment = args.get('segment')
    region = args.get('region')
    category = args.get('category')
    sub_category = args.get('sub_category')
    product_name = args.get('product_name')

    #  VALIDACIONES
    errors = []

    if quantity < 0:
        errors.append("quantity no puede ser negativa")

    if discount < 0 or discount > 1:
        errors.append("discount debe estar entre 0 y 1")

    if order_month < 1 or order_month > 12:
        errors.append("order_month debe estar entre 1 y 12")

    if delivery_days < 0:
        errors.append("delivery_days no puede ser negativo")

    if errors:
        return jsonify({
            "error": errors
        }), 400

    # DETECTAR NULOS
    missing = []

    # numéricos
    for name, val in [
        ('quantity', quantity),
        ('discount', discount),
        ('order_year', order_year),
        ('order_month', order_month),
        ('delivery_days', delivery_days),
        ('impact_sales_delay', impact_sales_delay)        
    ]:
        if np.isnan(val):
            missing.append(name)

    # categóricos
    for name, val in [
        ('ship_mode', ship_mode),
        ('segment', segment),
        ('region', region),
        ('category', category),
        ('sub_category', sub_category),
        ('product_name', product_name)
    ]:
        if val is None:
            missing.append(name)

    # CREAR DATAFRAME
    input_data = pd.DataFrame({
        'quantity': [quantity],
        'discount': [discount],
        'order_year': [order_year],
        'order_month': [order_month],
        'delivery_days': [delivery_days],
        'impact_sales_delay': [impact_sales_delay],
        'product_name': [product_name],
        'ship_mode': [ship_mode],
        'segment': [segment],
        'region': [region],
        'category': [category],
        'sub_category': [sub_category]
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

    if os.path.exists("data_sample/Superstore_synthetic.csv"):
        data = pd.read_csv('data_sample/Superstore_synthetic.csv')

        train_model(data)

        return jsonify({
            "mensaje": "Modelo reentrenado correctamente"
            })
    else:
        return jsonify({
            "error": "Archivo no encontrado"})


# RUN
if __name__ == '__main__':
    app.run(debug=True)