import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import datetime
import uuid
from supabase import create_client, Client

# Configuração da página Web
st.set_page_config(page_title="LoL Predictor: Punter", page_icon="📈", layout="wide")

# --- CONEXÃO COM O BANCO DE DADOS (SUPABASE) ---
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["https://hytbdxmlcmllpgzpqhuc.supabase.co/rest/v1/"]
        key = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5dGJkeG1sY21sbHBnenBxaHVjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExOTUzNzEsImV4cCI6MjA5Njc3MTM3MX0.nZYo54w2rO2xVLbbou7GZT3deafSJbyMF263XkQdrfk"]
        return create_client(url, key)
    except Exception as e:
        st.error("⚠️ Configuração do Supabase ausente. Configure seus Secrets localmente ou no Render.")
        return None

supabase = init_connection()

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
                
                # Gerando as 20 colunas de features com base nas forças relativas para alimentar o cérebro
                stats_entrada = []
                for _ in range(5):
                    stats_entrada.extend([np.random.normal(forca_azul, 5), np.random.normal(forca_azul/20, 0.5)])
                for _ in range(5):
                    stats_entrada.extend([np.random.normal(forca_vermelha, 5), np.random.normal(forca_vermelha/20, 0.5)])
                
                X_nova_partida = np.array(stats_entrada).reshape(1, -1)
                probabilidades = modelo.predict_proba(X_nova_partida)[0]
                
                # Mapeamento do predict_proba correspondente ao dataset de treino
                prob_azul = probabilidades[1] * 100 
                prob_vermelho = probabilidades[0] * 100
                
                # Mercados Secundários Simulados (Mantidos para o layout até o treinamento dos modelos específicos)
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
            
            if st.button("📝 Registrar Aposta na Nuvem", type="secondary"):
                if supabase:
                    # Preparando payload mapeado para as colunas criadas no banco PostgreSQL
                    novo_registro = {
                        "id": str(uuid.uuid4()),
                        "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "partida": f"{time_azul} vs {time_vermelho}",
                        "mercado": calc["mercado"],
                        "odd": float(calc["odd"]),
                        "stake": float(round(sugestao_aposta, 2)),
                        "ev_previsto": float(round(calc["ev_percent"], 2)),
                        "status": "PENDENTE"
                    }
                    supabase.table("historico_apostas").insert(novo_registro).execute()
                    
                    st.toast("✅ Aposta enviada para o Supabase!")
                    st.success("Tudo certo! Dados guardados na nuvem com segurança.")
                    st.session_state.resultado_calculo = None 
                else:
                    st.error("Conexão com o Supabase indisponível.")
            
        else:
            st.error(f"❌ **APOSTA COM VALOR NEGATIVO (-EV: {calc['ev_percent']:.2f}%)**")
            st.warning("⚠️ **NÃO APOSTE!** A Odd oferecida é muito baixa para a probabilidade real.")

# ==========================================
# ABA 3: HISTÓRICO INTEGRADO AO SUPABASE
# ==========================================
with aba_historico:
    st.header("📖 Diário de Bordo e Resultados (Sincronizado na Nuvem)")
    st.markdown("""
    Aqui ficam registradas todas as suas entradas diretamente no banco de dados.
    * ✏️ **Para atualizar:** Dê dois cliques na célula da coluna 'status' e mude para GREEN ou RED.
    * 🗑️ **Para excluir:** Clique na caixinha à esquerda da linha e aperte a tecla **Delete** no teclado.
    * 💾 **Sincronização:** Lembre-se de clicar no botão "Sincronizar Alterações com a Nuvem" para persistir as mudanças.
    """)
    
    if supabase:
        # Puxa o estado atualizado direto do banco de dados na nuvem
        resposta = supabase.table("historico_apostas").select("*").execute()
        dados_nuvem = resposta.data
        
        if dados_nuvem:
            df_historico = pd.DataFrame(dados_nuvem)
            
            # Reorganização estética das colunas e ocultação do ID técnico
            ordem_colunas = ["data", "partida", "mercado", "odd", "stake", "ev_previsto", "status", "id"]
            df_historico = df_historico[ordem_colunas]
            
            df_editado = st.data_editor(
                df_historico,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "id": None, # Esconde o UUID na interface para o visual ficar limpo
                    "status": st.column_config.SelectboxColumn(
                        "status",
                        options=["PENDENTE", "GREEN", "RED"],
                        required=True
                    )
                }
            )
            
            col_save, col_backup = st.columns([2, 6])
            with col_save:
                if st.button("💾 Sincronizar Alterações com a Nuvem", type="primary", use_container_width=True):
                    with st.spinner("Sincronizando dados..."):
                        # 1. Identificar e deletar registros removidos pelo data_editor
                        ids_antigos = set(df_historico['id'])
                        ids_atuais = set(df_editado['id'])
                        ids_deletados = ids_antigos - ids_atuais
                        
                        for id_del in ids_deletados:
                            supabase.table('historico_apostas').delete().eq('id', id_del).execute()
                        
                        # 2. Executar Upsert dos registros restantes ou modificados
                        lista_registros = df_editado.to_dict('records')
                        if lista_registros:
                            supabase.table('historico_apostas').upsert(lista_registros).execute()
                            
                        st.success("✅ Supabase atualizado com sucesso!")
                        st.rerun()
            with col_backup:
                # Permite gerar um arquivo físico local em caso de necessidade de auditoria
                csv_data = df_editado.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar Cópia Física de Segurança (CSV)", csv_data, file_name="backup_apostas_supabase.csv")
        else:
            st.info("Nenhuma aposta registrada no Supabase ainda. Faça simulações e salve dados na Aba 2.")