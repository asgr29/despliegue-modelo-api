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
        "uso": "/api/v1/predict?"
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

    ship_mode = to_float(args.get('ship_mode', None))
    segment = to_float(args.get('segment', None))
    region = to_float(args.get('region', None))
    category = to_float(args.get('category', None))
    sub_category = to_float(args.get('sub_category', None))
    product_name = to_float(args.get('product_name', None))

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

'''
# EXPLICACIÓN MODELO

@app.route('/api/v2/explain', methods=['GET'])
def explain():

    args = {k.lower(): v for k, v in request.args.items()}

    # Variables numéricas
    quantity = to_float(args.get('quantity', np.nan))
    discount = to_float(args.get('discount', np.nan))
    delivery_days = to_float(args.get('delivery_days', np.nan))

    # Variables categóricas
    category = args.get('category', None)

    # Lógica simple de explicación (heurística)
    explicaciones = []

    if not np.isnan(discount) and discount > 0.5:
        explicaciones.append("Descuento alto -> reduce margen -> posible pérdida")

    if not np.isnan(delivery_days) and delivery_days > 5:
        explicaciones.append("Entrega lenta -> afecta ventas")

    if not np.isnan(quantity) and quantity < 2:
        explicaciones.append("Baja cantidad -> menor volumen de ingresos")

    if category == "Furniture":
        explicaciones.append("Categoría Furniture suele tener menor margen")

    # Crear input mínimo para modelo
    input_data = pd.DataFrame({
        'quantity': [quantity],
        'discount': [discount],
        'order_year': [2020],
        'order_month': [5],
        'delivery_days': [delivery_days],
        'impact_sales_delay': [1],
        'product_name': ["Sample"],
        'ship_mode': ["Second Class"],
        'segment': ["Consumer"],
        'region': ["West"],
        'category': [category],
        'sub_category': ["Chairs"]
    })

    pred = model.predict(input_data)
    resultado = "ganancia" if pred[0] == 1 else "perdida"

    return jsonify({
        "prediccion": resultado,
        "explicacion": explicaciones if explicaciones else ["No hay factores claros detectados"]
    })
'''

# RETRAIN 
@app.route('/api/v3/retrain', methods=['GET'])
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