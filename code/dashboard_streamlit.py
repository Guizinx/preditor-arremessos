import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Dashboard de Arremessos do Kobe")

# Simulação de dados (ou você pode puxar de um log real futuramente)
data = pd.read_csv("../data/processed/predicoes.csv") # Altere se necessário

st.subheader("Distribuição dos Arremessos por Zona")
zone_count = data['shot_zone_area'].value_counts()
st.bar_chart(zone_count)

st.subheader("Distribuição por Tipo de Arremesso")
type_count = data['combined_shot_type'].value_counts()
st.bar_chart(type_count)

st.subheader("Taxa de Acerto por Período")
df_accuracy = data.groupby("period")['shot_made_flag'].mean()
st.line_chart(df_accuracy)

st.markdown("**Fonte de dados:** Kobe Bryant Shot Selection (Kaggle)")
