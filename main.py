'''from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Para que funcione  en cualquier entorno
os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Carga del modelo
modelo = joblib.load("logistic_regression.joblib")


# Landing page
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensaje": "API de predicción de pérdidas de productos",
        "endpoints": {
            "/api/v1/predict": "Predicción -> se debe pasar sales, quantity, discount",
            "/api/v1/retrain": "Reentrenar modelo (opcional)"
        },
        "ejemplo": "/api/v1/predict?sales=13&quantity=2&discount=0.7"
    })


# Predict
@app.route('/api/v1/predict', methods=['GET'])
def predict():

    sales = request.args.get('sales', np.nan, type=float)
    quantity = request.args.get('quantity', np.nan, type=float)
    discount = request.args.get('discount', np.nan, type=float)

    # Valores faltantes
    missing = [
        name for name, val in [
            ('sales', sales),
            ('quantity', quantity),
            ('discount', discount)
        ] if np.isnan(val)
    ]

    # crear dataframe como en entrenamiento
    input_data = pd.DataFrame({
        'Sales': [sales],
        'Quantity': [quantity],
        'Discount': [discount]
    })

    # predicción
    pred = modelo.predict(input_data)

    resultado = "perdida" if pred[0] == 0 else "ganancia"

    response = {
        "prediccion": resultado
    }

    if missing:
        response["warning"] = f"Faltan valores: {', '.join(missing)}"

    return jsonify(response)


# Retrain (EXTRA)
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    global modelo

    if os.path.exists("data/new_data.csv"):

        data = pd.read_csv("data/new_data.csv")

        X = data[['Sales', 'Quantity', 'Discount']]
        y = data['target']  # ← cambia esto si tu variable objetivo tiene otro nombre

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        modelo.fit(X_train, y_train)

        accuracy = modelo.score(X_test, y_test)

        # Guardar modelo actualizado (opcional)
        joblib.dump(modelo, "src/models/logistic_regression.joblib")

        return jsonify({
            "mensaje": "Modelo reentrenado",
            "accuracy": accuracy
        })

    else:
        return jsonify({
            "error": "No hay nuevos datos para reentrenar"
        })


# RUN
if __name__ == '__main__':
    app.run(debug=True)'''


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


# Predict
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        # NUMÉRICAS
        quantity = request.args.get('Quantity', type=float)
        discount = request.args.get('Discount', type=float)
        order_year = request.args.get('Order_Year', type=float)
        order_month = request.args.get('Order_Month', type=float)
        delivery_days = request.args.get('Delivery_Days', type=float)
        impact_sales_delay = request.args.get('Impact_Sales_Delay', type=float)
        product_te = request.args.get('ProductName_TE', type=float)

        # CATEGÓRICAS
        ship_mode = request.args.get('Ship_Mode')
        segment = request.args.get('Segment')
        region = request.args.get('Region')
        category = request.args.get('Category')
        sub_category = request.args.get('Sub-Category')

        # Crear dataframe exacto al entrenamiento
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

        # Predicción
        pred = model.predict(input_data)

        resultado = "ganancia" if pred[0] == 1 else "perdida"

        return jsonify({
            "prediccion": resultado
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# RETRAIN 
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    return jsonify({
        "mensaje": "Reentrenamiento no implementado en esta versión"
    })


# RUN
if __name__ == '__main__':
    app.run(debug=True)