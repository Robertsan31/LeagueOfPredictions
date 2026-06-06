import pandas as pd
import joblib
import random

# 1. Carregar o cérebro treinado
print("🧠 Acordando a Inteligência Artificial...")
try:
    modelo = joblib.load('../training_and_validation/league_ai_model.pkl')
except FileNotFoundError:
    print("Erro: O arquivo 'league_ai_model.pkl' não foi encontrado na pasta prediction.")
    exit()

# 2. Carregar o banco de dados original para pegar uma partida real "emprestada"
print("📂 Buscando uma partida no banco de dados para analisar...")
try:
    df = pd.read_csv('../training_and_validation/new.csv', delimiter=';')
except FileNotFoundError:
    print("Erro: O arquivo 'new.csv' não foi encontrado na pasta data_collection.")
    exit()

# Limpar os dados da mesma forma que fizemos no treino
df = df[df['win'] != 'ERR']
df['win'] = df['win'].astype(int)

# 3. Escolher uma partida aleatória do banco para simular a previsão
indice_aleatorio = random.choice(df.index)
partida_real = df.loc[[indice_aleatorio]]

# Separar os números da partida e o resultado que realmente aconteceu
X_nova_partida = partida_real.drop(['id', 'win'], axis=1)
X_nova_partida = X_nova_partida.replace('EMPTY', 0).fillna(0)
vencedor_real = partida_real['win'].values[0]

# 4. Fazer a previsão!
print("🔮 Analisando as estatísticas dos 10 jogadores...")
previsao = modelo.predict(X_nova_partida)[0]
probabilidades = modelo.predict_proba(X_nova_partida)[0]

# 5. Exibir os resultados de forma amigável
# No LoL, o time 0 é o Azul (Blue Side) e o time 1 é o Vermelho (Red Side)
time_azul = "Time Azul 🔵"
time_vermelho = "Time Vermelho 🔴"

vencedor_previsto_nome = time_azul if previsao == 0 else time_vermelho
vencedor_real_nome = time_azul if vencedor_real == 0 else time_vermelho

certeza_azul = probabilidades[0] * 100
certeza_vermelha = probabilidades[1] * 100

print("\n" + "="*55)
print("🎯 RESULTADO DA PREVISÃO DA IA 🎯")
print("="*55)
print(f"Probabilidade de vitória do {time_azul}: {certeza_azul:.1f}%")
print(f"Probabilidade de vitória do {time_vermelho}: {certeza_vermelha:.1f}%")
print("-" * 55)
print(f"🤖 O modelo aposta que o vencedor será: {vencedor_previsto_nome}")
print(f"📺 O que realmente aconteceu no jogo: {vencedor_real_nome}")

if previsao == vencedor_real:
    print("\n✅ A IA ACERTOU NA MOSCA!")
else:
    print("\n❌ A IA ERROU. (Estatística não tanka jogador trollando!)")
print("="*55 + "\n")