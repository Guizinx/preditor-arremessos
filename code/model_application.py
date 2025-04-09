from pycaret.classification import load_model, predict_model
import pandas as pd

def aplicar_modelo(path_modelo, path_novos_dados):
    modelo = load_model(path_modelo)
    df = pd.read_parquet(path_novos_dados)
    preds = predict_model(modelo, data=df)
    preds.to_csv("data/processed/predicoes.csv", index=False)
    print(preds.head())

if __name__ == "__main__":
    aplicar_modelo("models/modelo_final", "data/processed/base_test.parquet")