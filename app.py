import streamlit as st
import pandas as pd
import joblib
import random

# Configuração da página Web
st.set_page_config(page_title="LoL Predictor AI", page_icon="🔮", layout="centered")

st.title("🔮 League of Predictions: LCK")
st.markdown("Bem-vindo ao painel da nossa Inteligência Artificial preditiva. Clique no botão abaixo para puxar uma partida do banco de dados e ver se a IA consegue adivinhar o vencedor!")

# Função com 'cache' para não precisar carregar o cérebro toda vez que clicar no botão
@st.cache_resource
def carregar_dados():
    # Carregando a partir da pasta raiz (LeagueOfPredictions)
    modelo = joblib.load('training_and_validation/league_ai_model.pkl')
    df = pd.read_csv('training_and_validation/new.csv', delimiter=';')
    df = df[df['win'] != 'ERR']
    df['win'] = df['win'].astype(int)
    return modelo, df

try:
    modelo, df = carregar_dados()
except FileNotFoundError:
    st.error("Arquivos do modelo não encontrados. Certifique-se de estar rodando o app na pasta raiz do projeto.")
    st.stop()

st.divider()

# Botão principal da interface
if st.button("🎲 Sortear Partida e Fazer Previsão", type="primary", use_container_width=True):
    
    with st.spinner("Analisando KDA, Maestria e Winrate dos 10 jogadores..."):
        # Sorteio e Preparação da partida
        indice_aleatorio = random.choice(df.index)
        partida_real = df.loc[[indice_aleatorio]]
        
        X_nova_partida = partida_real.drop(['id', 'win'], axis=1)
        X_nova_partida = X_nova_partida.replace('EMPTY', 0).fillna(0)
        vencedor_real = partida_real['win'].values[0]
        id_partida = partida_real['id'].values[0]
        
        # Previsão da IA
        previsao = modelo.predict(X_nova_partida)[0]
        probabilidades = modelo.predict_proba(X_nova_partida)[0]
        
        certeza_azul = probabilidades[0] * 100
        certeza_vermelha = probabilidades[1] * 100
        
        # --- EXIBIÇÃO VISUAL ---
        st.subheader(f"Partida Analisada: `{id_partida}`")
        
        # Criando duas colunas para mostrar os times lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🔵 **Time Azul**")
            st.metric(label="Chance de Vitória", value=f"{certeza_azul:.1f}%")
            
        with col2:
            st.error("🔴 **Time Vermelho**")
            st.metric(label="Chance de Vitória", value=f"{certeza_vermelha:.1f}%")
            
        st.divider()
        
        # Revelação do Vencedor
        vencedor_previsto_nome = "Time Azul" if previsao == 0 else "Time Vermelho"
        vencedor_real_nome = "Time Azul" if vencedor_real == 0 else "Time Vermelho"
        
        st.markdown(f"### 🤖 Aposta da IA: **{vencedor_previsto_nome}**")
        st.markdown(f"### 📺 Resultado Real: **{vencedor_real_nome}**")
        
        if previsao == vencedor_real:
            st.success("✅ A Inteligência Artificial ACERTOU o resultado da partida!")
            st.balloons() # Efeito visual de comemoração na tela
        else:
            st.warning("❌ A IA ERROU. Mas culpe o Yasuo 0/10 do time aliado.")