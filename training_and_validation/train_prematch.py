import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import glob
import os

print("🧠 Iniciando o Treinamento do Modelo Pré-Jogo (Punter)...")

# Encontrar automaticamente o arquivo prematch mais recente gerado
arquivos_csv = glob.glob('../data_collection/prematch_*.csv')
if not arquivos_csv:
    print("❌ Nenhum arquivo de dados pré-jogo encontrado. Rode a mineração primeiro!")
    exit()

# Pega o arquivo mais recente
ARQUIVO_DADOS = max(arquivos_csv, key=os.path.getctime)
print(f"📂 Carregando dados de: {ARQUIVO_DADOS}")

df = pd.read_csv(ARQUIVO_DADOS, delimiter=';')

# Separar Estatísticas (X) e Resultado Final (y)
X = df.drop(['id', 'win'], axis=1)
y = df['win']

# Limpar dados vazios por segurança
X = X.fillna(0)

# Separar treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar a IA
print("⚙️ Treinando a Inteligência Artificial...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print("\n" + "="*50)
print(f"🎉 TREINAMENTO PRÉ-JOGO CONCLUÍDO! 🎉")
print(f"🎯 Acurácia Base do Modelo: {acuracia * 100:.2f}%")
print("="*50 + "\n")

# Salvar o novo cérebro
joblib.dump(modelo, 'prematch_ai_model.pkl')
print("✅ Cérebro salvo como: prematch_ai_model.pkl")