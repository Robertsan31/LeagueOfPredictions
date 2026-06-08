import time
import os
from riotwatcher import LolWatcher, ApiError

# ==========================================
# CONFIGURAÇÕES (COLOQUE SUA CHAVE AQUI)
# ==========================================
API_KEY = 'RGAPI-e42c2300-b63f-40fb-9204-3299dac726ff' # Cole sua chave da Riot
SERVER_REGION = 'kr'
MATCH_REGION = 'asia'
MATCHES_FILE = 'matches_002.txt' # Confira se o seu arquivo chama matches.txt ou matches_002.txt
OUTPUT_FILE = f'prematch_{int(time.time())}.csv'

watcher = LolWatcher(API_KEY)

def get_general_stats(puuid):
    """Busca o histórico recente do jogador e calcula Winrate e KDA globais"""
    try:
        # Puxa as últimas 15 partidas do jogador
        matchlist = watcher.match.matchlist_by_puuid(MATCH_REGION, puuid, count=15)
        
        wins = 0
        kills, deaths, assists = 0, 0, 0
        valid_matches = 0

        for match_id in matchlist:
            time.sleep(1) # Respeitando o Rate Limit da Riot
            match_detail = watcher.match.by_id(MATCH_REGION, match_id)
            
            for participant in match_detail['info']['participants']:
                if participant['puuid'] == puuid:
                    valid_matches += 1
                    if participant['win']:
                        wins += 1
                    kills += participant['kills']
                    deaths += participant['deaths']
                    assists += participant['assists']
                    break

        if valid_matches == 0:
            return 0.0, 0.0

        winrate = (wins / valid_matches) * 100
        deaths = 1 if deaths == 0 else deaths
        kda = (kills + assists) / deaths

        return round(winrate, 2), round(kda, 2)

    except ApiError as err:
        if err.response.status_code == 429:
            print("⏳ Rate Limit atingido! Pausando por 2 minutos...")
            time.sleep(120)
            return get_general_stats(puuid) 
        else:
            return 0.0, 0.0

def start_mining():
    print("🚀 Iniciando Mineração de Dados Pré-Jogo...")
    
    # Criar o cabeçalho do CSV
    header = "id;"
    for i in range(1, 11):
        header += f"p{i}_winrate;p{i}_kda;"
    header += "win\n"
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(header)

    with open(MATCHES_FILE, 'r') as f:
        match_ids = [line.strip() for line in f if line.strip()]

    for match_id in match_ids:
        print(f"[{match_id}] Analisando partida...")
        try:
            match_data = watcher.match.by_id(MATCH_REGION, match_id)
            participants = match_data['info']['participants']
            
            blue_team_win = 1 if match_data['info']['teams'][0]['win'] else 0
            
            row_data = f"{match_id};"
            
            for p in participants:
                nome_jogador = p.get('riotIdGameName', p.get('summonerName', 'Jogador Desconhecido'))
                print(f"   -> Extraindo stats globais de: {nome_jogador}")
                winrate, kda = get_general_stats(p['puuid'])
                row_data += f"{winrate};{kda};"
                
            row_data += f"{blue_team_win}\n"
            
            with open(OUTPUT_FILE, 'a') as f:
                f.write(row_data)
                
            print(f"✅ Partida {match_id} salva!")
            
        except ApiError as err:
            if err.response.status_code == 429:
                print("⏳ Rate Limit atingido. Pausando por 2 minutos...")
                time.sleep(120)
            else:
                print(f"❌ Erro ao ler a partida {match_id}: {err}")

if __name__ == "__main__":
    start_mining()