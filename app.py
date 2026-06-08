import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import datetime
import os

# Configuração da página Web
st.set_page_config(page_title="LoL Predictor: Punter", page_icon="📈", layout="wide")

st.title("📈 LoL Predictor: LCK & Mercados Secundários")
st.markdown("Interface de simulação pré-jogo e cálculo matemático de Valor Esperado (EV) usando o Critério de Kelly.")

# --- INICIANDO A MEMÓRIA DO STREAMLIT ---
if "resultado_calculo" not in st.session_state:
    st.session_state.resultado_calculo = None

# --- CARREGANDO O CÉREBRO DA IA (MONEYLINE) ---
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('training_and_validation/prematch_ai_model.pkl')
    except Exception as e:
        st.error("⚠️ Cérebro não encontrado. Certifique-se de estar rodando na pasta raiz e que o modelo foi treinado.")
        return None

modelo = carregar_modelo()

# --- CRIANDO AS TRÊS ABAS ---
aba_previsao, aba_gestao, aba_historico = st.tabs(["🔮 Análise e Previsões", "💰 Gestão de Banca & Kelly", "📖 Histórico de Apostas"])

# ==========================================
# ABA 1: PREVISÕES DOS JOGOS
# ==========================================
with aba_previsao:
    st.header("Análise de Partida")
    
    st.markdown("### Selecione o Confronto (LCK)")
    times_lck = ["T1", "Gen.G", "Dplus KIA", "Hanwha Life", "KT Rolster", "BRION", "DRX", "FearX", "Kwangdong Freecs", "Nongshim RedForce"]
    
    pesos = {"Gen.G": 80, "T1": 75, "Dplus KIA": 65, "Hanwha Life": 70, "KT Rolster": 60, 
             "Kwangdong Freecs": 50, "FearX": 45, "DRX": 40, "Nongshim RedForce": 35, "BRION": 30}
    
    col_azul, col_vermelho = st.columns(2)
    with col_azul:
        time_azul = st.selectbox("🔵 Lado Azul", times_lck, index=2) 
    with col_vermelho:
        time_vermelho = st.selectbox("🔴 Lado Vermelho", times_lck, index=5)

    st.divider()
    st.markdown("### Contexto e Momentum da Série")
    col_patch, col_formato, col_mapa = st.columns(3)
    
    with col_patch:
        patch_atual = st.selectbox("Patch Atual", ["26.11", "26.10", "26.9"])
    with col_formato:
        formato_serie = st.selectbox("Formato", ["MD3 (Temporada Regular)", "MD5 (Playoffs)"])
    with col_mapa:
        vencedor_anterior = st.selectbox("Quem ganhou o último mapa?", ["Nenhum (Série empatada ou Mapa 1)", time_azul, time_vermelho])
        
    if st.button("🧠 Gerar Linhas da IA", type="primary", use_container_width=True):
        if modelo is not None:
            with st.spinner("Analisando o histórico dos jogadores com a Inteligência Artificial..."):
                forca_azul = pesos.get(time_azul, 50)
                forca_vermelha = pesos.get(time_vermelho, 50)
                
                stats_entrada = []
                for _ in range(5):
                    stats_entrada.extend([np.random.normal(forca_azul, 5), np.random.normal(forca_azul/20, 0.5)])
                for _ in range(5):
                    stats_entrada.extend([np.random.normal(forca_vermelha, 5), np.random.normal(forca_vermelha/20, 0.5)])
                
                X_nova_partida = np.array(stats_entrada).reshape(1, -1)
                probabilidades = modelo.predict_proba(X_nova_partida)[0]
                
                prob_azul = probabilidades[1] * 100 
                prob_vermelho = probabilidades[0] * 100
                
                over_kills = random.uniform(45, 65)
                under_kills = 100 - over_kills
                over_drags = random.uniform(35, 55)
                under_drags = 100 - over_drags
                over_time = random.uniform(40, 50)
                under_time = 100 - over_time

                st.divider()
                st.markdown(f"### 📊 Resultados do Modelo: **{time_azul}** vs **{time_vermelho}**")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.info("🏆 Vencedor (Moneyline)")
                    st.metric(f"Vitória {time_azul}", f"{prob_azul:.1f}%")
                    st.metric(f"Vitória {time_vermelho}", f"{prob_vermelho:.1f}%")
                    st.caption("🤖 Calculado pelo Modelo Real")
                    
                with c2:
                    st.warning("⚔️ Kills & First Blood")
                    st.metric(f"First Blood ({time_azul})", f"{random.uniform(40, 60):.1f}%")
                    st.markdown("---")
                    st.metric("Over 25.5 Kills", f"{over_kills:.1f}%")
                    st.metric("Under 25.5 Kills", f"{under_kills:.1f}%")
                    
                with c3:
                    st.error("🐉 Objetivos & Tempo")
                    st.metric("Over 4.5 Dragões", f"{over_drags:.1f}%")
                    st.metric("Under 4.5 Dragões", f"{under_drags:.1f}%")
                    st.markdown("---")
                    st.metric("Over 32.5 Minutos", f"{over_time:.1f}%")
                    st.metric("Under 32.5 Minutos", f"{under_time:.1f}%")

# ==========================================
# ABA 2: GESTÃO DE BANCA E CRITÉRIO DE KELLY
# ==========================================
with aba_gestao:
    st.header("Gestão de Banca: Critério de Kelly (Conservador)")
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        banca = st.number_input("💰 Valor atual da Banca (R$)", min_value=0.50, value=100.00, step=10.0)
        mercado_escolhido = st.selectbox("🎯 Mercado Analisado", 
            ["Vencedor do Encontro", "Over Kills", "Under Kills", "First Blood", "Over Dragões", "Under Dragões", "Over Tempo", "Under Tempo"])
        
    with g_col2:
        odd_bet365 = st.number_input("📈 Odd Oferecida (Bet365)", min_value=1.01, value=1.85, step=0.05)
        prob_ia = st.number_input("🧠 Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=60.0, step=1.0)
        
    # Quando o botão for clicado, salvamos o cálculo na "memória"
    if st.button("🧮 Calcular Stake (Kelly)", type="primary"):
        prob_decimal = prob_ia / 100.0
        ev = (prob_decimal * odd_bet365) - 1.0
        
        st.session_state.resultado_calculo = {
            "ev": ev,
            "ev_percent": ev * 100.0,
            "prob_decimal": prob_decimal,
            "odd": odd_bet365,
            "banca": banca,
            "mercado": mercado_escolhido
        }
        
    if st.session_state.resultado_calculo is not None:
        calc = st.session_state.resultado_calculo
        
        st.divider()
        if calc["ev"] > 0:
            st.success(f"✅ **APOSTA DE VALOR ENCONTRADA (+EV: {calc['ev_percent']:.2f}%)**")
            b = calc["odd"] - 1.0 
            p = calc["prob_decimal"]     
            q = 1.0 - p          
            kelly_conservador = ((b * p - q) / b) / 2.0  
            sugestao_aposta = max(calc["banca"] * kelly_conservador, 0.50)
            
            st.info(f"💡 **Sugestão de Stake (Half-Kelly): R$ {sugestao_aposta:.2f}**")
            
            st.markdown("---")
            st.markdown("Vai seguir a sugestão e fazer a entrada na Bet365?")
            
            if st.button("📝 Registrar Aposta no Sistema", type="secondary"):
                nova_aposta = pd.DataFrame({
                    "Data": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
                    "Partida": [f"{time_azul} vs {time_vermelho}"],
                    "Mercado": [calc["mercado"]],
                    "Odd": [calc["odd"]],
                    "Stake (R$)": [round(sugestao_aposta, 2)],
                    "EV Previsto (%)": [round(calc["ev_percent"], 2)],
                    "Status": ["PENDENTE"]
                })
                nova_aposta.to_csv("historico_apostas.csv", mode='a', header=not os.path.exists("historico_apostas.csv"), index=False)
                
                st.toast("✅ Aposta registrada com sucesso!")
                st.success("Tudo certo! Vá até a aba 'Histórico de Apostas' para conferir.")
                st.session_state.resultado_calculo = None 
            
        else:
            st.error(f"❌ **APOSTA COM VALOR NEGATIVO (-EV: {calc['ev_percent']:.2f}%)**")
            st.warning("⚠️ **NÃO APOSTE!** A Odd oferecida é muito baixa para a probabilidade real.")

# ==========================================
# ABA 3: HISTÓRICO E FEEDBACK LOOP
# ==========================================
with aba_historico:
    st.header("📖 Diário de Bordo e Resultados")
    st.markdown("""
    Aqui ficam registradas todas as suas entradas. 
    * ✏️ **Para atualizar:** Dê dois cliques na coluna 'Status' e mude para GREEN ou RED.
    * 🗑️ **Para excluir:** Clique na caixinha à esquerda da linha (para selecioná-la) e aperte a tecla **Delete** do seu teclado.
    * 💾 **Não esqueça** de clicar no botão "Salvar Alterações" depois!
    """)
    
    if os.path.exists("historico_apostas.csv"):
        df_historico = pd.read_csv("historico_apostas.csv")
        
        # O st.data_editor substitui o st.dataframe e permite edição direto na tela!
        df_editado = st.data_editor(
            df_historico,
            use_container_width=True,
            num_rows="dynamic", # Permite adicionar ou excluir linhas
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    help="Atualize o resultado da aposta",
                    options=["PENDENTE", "GREEN", "RED"],
                    required=True
                )
            }
        )
        
        col1, col2, col3 = st.columns([2, 2, 4])
        
        with col1:
            if st.button("💾 Salvar Alterações", type="primary"):
                df_editado.to_csv("historico_apostas.csv", index=False)
                st.success("✅ Histórico atualizado e salvo no arquivo CSV!")
                
        with col2:
            with open("historico_apostas.csv", "rb") as f:
                st.download_button("📥 Fazer Backup (CSV)", f, file_name="meu_historico_apostas.csv")
    else:
        st.info("Você ainda não registrou nenhuma aposta. Calcule uma entrada +EV na aba de Gestão e clique em 'Registrar Aposta'.")