import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Caminho local (corrigido) para o modelo salvo
model_path = r'c:\Users\guilh\OneDrive\Área de Trabalho\Preditor_de_Arremessos\models\modelo_final.pkl'
model = mlflow.pyfunc.load_model(f"file://{model_path}")
print("✅ Modelo carregado com sucesso!")

@app.route('/')
def home():
    return "API de Predição de Arremessos do Kobe está online!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return jsonify({
            'input': input_data,
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

    