import time
import datetime
from riotwatcher import LolWatcher, ApiError
import configparser
import requests

config = configparser.ConfigParser()
config.read('config.ini')
API_KEY = config['DEFAULT']['API_KEY']

LOL_WATCHER = LolWatcher(API_KEY)


# ================= CONFIGURAÇÕES PARA A LCK =================
QUEUE_TYPE = 420     # 420 = Ranked Solo Queue (onde os prós treinam)
REGION = 'kr'        # Servidor Coreano
ROUTE = 'asia'       # Rota regional para o Riot ID
PAST_MATCHES_COUNT = 5  # Quantidade de partidas iniciais para rastrear (aumentei para pegar mais dados)

# Conta do Faker na Coreia como semente de dados
SEED_GAME_NAME = 'Hide on bush' 
SEED_TAG_LINE = 'KR1'

# Histórico dos últimos 14 dias
OLDEST_ALLOWED_DATE = datetime.date.today() - datetime.timedelta(days=14)
OLDEST_ALLOWED_DATE = int(time.mktime(OLDEST_ALLOWED_DATE.timetuple()))
# ============================================================

def get_past_matches(puuid):
    match_list = LOL_WATCHER.match.matchlist_by_puuid(ROUTE, puuid, start_time=OLDEST_ALLOWED_DATE, queue=QUEUE_TYPE, count=PAST_MATCHES_COUNT)
    return match_list

def get_match(match_id):
    while True:
        try:
            req_match = LOL_WATCHER.match.by_id(ROUTE, match_id)
        except ApiError as err:
            if err.response.status_code == 429:
                print('Aguardando o limite de requisições (Rate Limit)...')
                time.sleep(10)
                continue
            elif err.response.status_code == 404:
                print('Partida não encontrada.')
                return None
            elif err.response.status_code == 503:
                print('Servidor da Riot ocupado... Tentando novamente em 5s')
                time.sleep(5)
                continue
            else:
                print('Erro inesperado, aguardando 15 segundos...')
                time.sleep(15)
                continue
        break
    return req_match

def get_summoner_puuid(game_name, tag_line):
    # Fazemos a chamada direta para a API da Riot sem depender de métodos desatualizados da biblioteca
    url = f"https://{ROUTE}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    headers = {"X-Riot-Token": API_KEY}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()['puuid']
    elif response.status_code == 401 or response.status_code == 403:
        raise Exception("Sua API_KEY no config.ini é inválida ou expirou. Atualize-a no site da Riot Developers.")
    else:
        raise Exception(f"Erro ao buscar jogador: Status {response.status_code} - {response.text}")

def get_all_summoners(past_matches):
    summoners = []
    for match_id in past_matches:
        match = get_match(match_id)
        if match:
            summoners.extend([participant['puuid'] for participant in match['info']['participants']])
    return list(set(summoners))

def get_all_matches(summoners):
    matches = []
    total = len(summoners)
    for index, summoner in enumerate(summoners):
        print(f"Buscando partidas do jogador {index+1}/{total}...")
        try:
            matches.extend(get_past_matches(summoner))
        except Exception:
            continue
    return list(set(matches))

def write_matches_to_file(matches, filename):
    with open(filename, "a") as f:
        for match_id in matches:
            f.write(f"{match_id}\n")

def main():
    print("Iniciando a coleta de partidas da LCK...")
    try:
        puuid = get_summoner_puuid(SEED_GAME_NAME, SEED_TAG_LINE)
        print(f"PUUID do jogador semente encontrado. Buscando últimas {PAST_MATCHES_COUNT} partidas...")
        past_matches = get_past_matches(puuid)
        
        print(f"Extraindo outros jogadores dessas partidas...")
        summoners = get_all_summoners(past_matches)
        
        print(f"Varrendo o histórico de todos os {len(summoners)} jogadores encontrados...")
        matches = get_all_matches(summoners)
        
        write_matches_to_file(matches, "manymatches.txt")
        print(f"Sucesso! {len(matches)} ID's de partidas salvos em manymatches.txt")
    except ApiError as e:
        print(f"Erro na API da Riot: {e}")

if __name__ == "__main__":
    main()