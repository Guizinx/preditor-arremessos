import pandas as pd
import mlflow
from pycaret.classification import setup, create_model, predict_model, pull, save_model
from sklearn.metrics import log_loss
import os


def treinar_modelo(path_train, nome_modelo_final="modelo_final"):
    df_train = pd.read_parquet(path_train)

    # Define o experimento no MLflow
    mlflow.set_experiment("Treinamento")

    # Inicializa o ambiente do PyCaret
    setup(data=df_train, target='shot_made_flag', session_id=42, verbose=False)

    modelos_resultados = {}

    # Modelo 1: Regressão Logística
    with mlflow.start_run(run_name="Treinamento_Logistica"):
        modelo_log = create_model("lr")
        predict_model(modelo_log)
        resultados_log = pull()

        f1_log = resultados_log.loc[0, "F1"]
        mlflow.log_metric("f1_score", f1_log)
        mlflow.log_param("modelo", "Logistic Regression")
        modelos_resultados["log"] = (modelo_log, f1_log)

    # Modelo 2: Árvore de Decisão
    with mlflow.start_run(run_name="Treinamento_Arvore"):
        modelo_dt = create_model("dt")
        predict_model(modelo_dt)
        resultados_dt = pull()

        f1_dt = resultados_dt.loc[0, "F1"]
        mlflow.log_metric("f1_score", f1_dt)
        mlflow.log_param("modelo", "Decision Tree")
        modelos_resultados["dt"] = (modelo_dt, f1_dt)

    # Escolhe o melhor modelo com base na F1
    modelo_final, f1_final = max(modelos_resultados.values(), key=lambda x: x[1])
    nome_modelo_escolhido = "Decision Tree" if modelo_final == modelos_resultados["dt"][0] else "Logistic Regression"

    # Log do modelo final
    with mlflow.start_run(run_name="Modelo_Final"):
        # Garantir que o diretório 'models' existe
        output_dir = 'models'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Salva o modelo final com o caminho correto
        model_path = os.path.join(output_dir, f"{nome_modelo_final}.pkl")
        model_saved = save_model(modelo_final, model_path)

        mlflow.log_param("modelo_escolhido", nome_modelo_escolhido)
        mlflow.log_metric("f1_score_final", f1_final)

        # Log do artefato com o caminho correto
        mlflow.log_artifact(model_path)

    return modelo_final


if __name__ == "__main__":
    treinar_modelo(
        r"C:\Users\guilh\OneDrive\Área de Trabalho\Preditor_de_Arremessos\data\processed\base_train.parquet",
        "modelo_final"  # Caminho correto para o nome do modelo final
    )
