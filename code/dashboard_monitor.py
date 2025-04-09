import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Dashboard de Monitoramento - Kobe Bryant")

# Carrega os dados de predição
df = pd.read_csv("data/processed/predicoes.csv")

# Exibe as 5 primeiras linhas
st.subheader("Pré-visualização dos dados")
st.write(df.head())

# Distribuição de rótulos previstos
st.subheader("Distribuição das Previsões")
st.bar_chart(df["prediction_label"].value_counts())

# Histograma da probabilidade
st.subheader("Distribuição da Probabilidade de Acerto")
fig, ax = plt.subplots()
ax.hist(df["prediction_score"], bins=20, color="skyblue", edgecolor="black")
ax.set_xlabel("Probabilidade de Acerto (prediction_score)")
ax.set_ylabel("Contagem")
st.pyplot(fig)