import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preparar_dados(path_dev, path_prod, save_path_filtered):
    """
    Lê a base de desenvolvimento ou produção, filtra e trata os dados, e salva a base processada.
    """
    with mlflow.start_run(run_name="PreparacaoDados"):
        try:
            df = pd.read_parquet(path_dev)
            mlflow.log_param("fonte_dados", "dev")
        except Exception as e:
            df = pd.read_parquet(path_prod)
            mlflow.log_param("fonte_dados", "prod")

        colunas = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
        df = df[colunas].dropna()

        df['shot_made_flag'] = df['shot_made_flag'].astype(int)
        df['playoffs'] = df['playoffs'].astype(int)

        mlflow.log_param("colunas_usadas", colunas)
        mlflow.log_metric("linhas_filtradas", df.shape[0])
        mlflow.log_metric("colunas", df.shape[1])

        df.to_parquet(save_path_filtered, index=False)
        return df


def split_dados(df, save_path_train, save_path_test):
    """
    Divide os dados em treino e teste estratificados e salva as bases separadas.
    """
    with mlflow.start_run(run_name="SplitDados"):
        X = df.drop("shot_made_flag", axis=1)
        y = df["shot_made_flag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        df_train.to_parquet(save_path_train, index=False)
        df_test.to_parquet(save_path_test, index=False)

        mlflow.log_param("percent_teste", 0.2)
        mlflow.log_metric("tamanho_treino", df_train.shape[0])
        mlflow.log_metric("tamanho_teste", df_test.shape[0])

if __name__ == "__main__":
    df_filtrado = preparar_dados(
        path_dev="data/raw/dataset_kobe_dev.parquet",
        path_prod='',
        save_path_filtered="data/processed/data_filtered.parquet"
    )

    split_dados(
        df=df_filtrado,
        save_path_train="data/processed/base_train.parquet",
        save_path_test="data/processed/base_test.parquet"
    )
