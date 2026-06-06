import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Carregar os dados
print("Carregando os dados do banco...")
FILENAME = 'new.csv' 
df = pd.read_csv(FILENAME, delimiter=';')

# Limpar qualquer dado com erro de leitura do script anterior
df = df[df['win'] != 'ERR']
df['win'] = df['win'].astype(int)

# 2. Separar quem é o Resultado (y) e quem são as Estatísticas (X)
print("Preparando a matriz para o treinamento...")
X = df.drop(['id', 'win'], axis=1) # Tira o ID da partida e o resultado final da visão da IA
y = df['win'] # O que a IA precisa aprender a adivinhar

# Substituir palavras vazias por 0 (caso algum jogador não tenha histórico)
X = X.replace('EMPTY', 0)
X = X.fillna(0)

# 3. Separar os dados: 80% para a IA estudar, 20% para a "prova final"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Criar o "Cérebro" da IA (Usando Floresta Aleatória)
print("Treinando a Inteligência Artificial... (Isso pode levar alguns segundos)")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Fazer a prova final e ver a nota da IA
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes)

print("\n" + "="*50)
print(f"🎉 TREINAMENTO CONCLUÍDO COM SUCESSO! 🎉")
print(f"🎯 Acurácia do Modelo: {acuracia * 100:.2f}%")
print("="*50 + "\n")

# 6. Salvar o cérebro treinado em um arquivo físico
arquivo_modelo = 'league_ai_model.pkl'
joblib.dump(modelo, arquivo_modelo)
print(f"Cérebro da IA salvo pronto para uso no arquivo: {arquivo_modelo}")