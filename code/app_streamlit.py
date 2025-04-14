import streamlit as st
import pandas as pd
import requests

st.title("🏀 Preditor de Arremessos do Kobe")

st.markdown("Preencha os dados abaixo para prever se Kobe acertaria o arremesso:")

# Entradas
action_type = st.selectbox("Tipo de ação", ['Jump Shot', 'Layup', 'Dunk', 'Hook Shot'])
combined_shot_type = st.selectbox("Tipo de arremesso", ['2PT Field Goal', '3PT Field Goal'])
shot_distance = st.slider("Distância do arremesso (pés)", 0, 40, 10)
shot_zone_area = st.selectbox("Zona do arremesso", ['Center(C)', 'Right Side(R)', 'Left Side(L)', 'Back Court(BC)'])
opponent = st.text_input("Adversário (sigla)", "LAL")
period = st.slider("Período do jogo", 1, 4, 1)
playoffs = st.selectbox("É playoffs?", [0, 1])
minutes_remaining = st.slider("Minutos restantes no período", 0, 12, 6)
seconds_remaining = st.slider("Segundos restantes no período", 0, 59, 30)

# Montagem do JSON
input_data = {
    "action_type": action_type,
    "combined_shot_type": combined_shot_type,
    "shot_distance": shot_distance,
    "shot_zone_area": shot_zone_area,
    "opponent": opponent,
    "period": period,
    "playoffs": playoffs,
    "minutes_remaining": minutes_remaining,
    "seconds_remaining": seconds_remaining
}

if st.button("Prever"):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        result = response.json()
        if 'prediction' in result:
            if result['prediction'] == 1:
                st.success("🏀 Kobe **acertaria** esse arremesso!")
            else:
                st.error("❌ Kobe **erraria** esse arremesso.")
        else:
            st.error("Erro na predição: " + str(result))
    except Exception as e:
        st.error("Erro ao conectar com a API: " + str(e))
