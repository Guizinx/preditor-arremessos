Lista de Ferramentas:

Ferramenta     - Função
PyCaret	       - Automatiza preparação, treino, avaliação e comparação de modelos.
Scikit-Learn   - Backend dos modelos utilizados (Logística e Árvore).
MLFlow	       - Rastreia experimentos, registra métricas, armazena e serve modelos.
Streamlit      - Cria dashboards interativos para monitoramento e aplicação do modelo.

Lista de Artefatos:

Artefato	                            - Descrição
/data/raw/dataset_kobe_{dev,prod}.parquet   - Dados originais.
/data/processed/data_filtered.parquet	    - Dados tratados e filtrados.
/data/processed/base_train.parquet	    - Conjunto de treino.
/data/processed/base_test.parquet	    - Conjunto de teste.
notebooks/	                            - Notebooks de exploração e análise.
src/preprocessing.py		            - Código de preparação de dados.
src/training.py			            - Código de treinamento do modelo.
src/applicacao.py			    - Pipeline para aplicar modelo em produção.
models/			                    - Modelos salvos e registrados pelo MLFlow.
streamlit_app.py	                    - Dashboard de operação.
README.md	                            - Explicação do projeto e instruções.
requirements.txt	                    - Lista de dependências.

