import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

print("🔍 Iniciando a Validação Profissional do Modelo...")

# 1. Carregar o modelo e os dados
try:
    modelo = joblib.load('league_ai_model.pkl')
    # O script vai buscar o csv lá na pasta data_collection
    df = pd.read_csv('../data_collection/new.csv', delimiter=';')
except FileNotFoundError as e:
    print(f"\nErro ao carregar arquivos: {e}")
    print("Certifique-se de que o 'league_ai_model.pkl' está na mesma pasta que este script.")
    exit()

# 2. Preparar os dados
df = df[df['win'] != 'ERR']
df['win'] = df['win'].astype(int)

X = df.drop(['id', 'win'], axis=1)
X = X.replace('EMPTY', 0).fillna(0)
y_real = df['win']

# 3. Fazer as previsões para TODO o banco de dados
print("🤖 Testando o modelo contra todo o histórico de partidas...")
previsoes = modelo.predict(X)

# 4. Mostrar o Boletim Completo
print("\n" + "="*50)
print("📊 BOLETIM DE DESEMPENHO DA INTELIGÊNCIA ARTIFICIAL")
print("="*50)

# Matriz de Confusão
matriz = confusion_matrix(y_real, previsoes)
print("\n🗺️ MATRIZ DE CONFUSÃO:")
print(f"Acertou vitória do Time Azul: {matriz[0][0]} vezes")
print(f"Errou (Apostou Vermelho, mas deu Azul): {matriz[0][1]} vezes")
print(f"Errou (Apostou Azul, mas deu Vermelho): {matriz[1][0]} vezes")
print(f"Acertou vitória do Time Vermelho: {matriz[1][1]} vezes")

# Relatório de Classificação
print("\n📈 RELATÓRIO DE MÉTRICAS:")
relatorio = classification_report(y_real, previsoes, target_names=['Time Azul (0)', 'Time Vermelho (1)'])
print(relatorio)
print("="*50 + "\n")