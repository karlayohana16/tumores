from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load("modelo_cancer.pkl")

# Para este ejemplo, asumimos que el modelo espera solo 3 variables
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        float(data["mean_radius"]),
        float(data["mean_texture"]),
        float(data["mean_perimeter"])
    ]
    prediction = model.predict([features])[0]
    return jsonify({"prediccion": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
