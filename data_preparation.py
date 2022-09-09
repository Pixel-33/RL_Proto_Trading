import pandas as pd


def data_preparation_BTCUSD(filename):
    df_btc_data = pd.read_csv('./data/' + filename)
    df_btc_data.dropna(inplace=True)                                # 4857376 => 3613769 raws
    df_btc_data.sort_values(by='Timestamp', inplace=True)           # s'assurer de l'ordre chronologique
    df_btc_data['Timestamp'] = pd.to_datetime(df_btc_data['Timestamp'], unit='s', errors='coerce')
    df_btc_data.isna().sum()                                        # s'assurer que les conversions de date sont ok
    df_btc_data.rename(columns={'Timestamp': 'time'}, inplace=True)
    df_btc_data.columns = df_btc_data.columns.str.lower()
    df_btc_data.reset_index(inplace=True, drop=True)                # réindexer suite aux raws supprimées
    df_btc_data.drop(['volume_(btc)', 'weighted_price'], axis=1, inplace=True)  # suppression des colonnes inutiles

    # recherche de la série temporelle à la minute sans manquants
    '''df_btc_data['serie_continue'] = df_btc_data['time'] + timedelta(minutes=1)
    df_btc_data['serie_continue'][1:] = df_btc_data['serie_continue'][0:-1]
    df_btc_data['boolean'] = df_btc_data['time'] == df_btc_data['serie_continue']'''

    # Extraction per DAY at time 00:00:00 à partir de 2017 => BTCUSD_D.csv
    df_btc_data.set_index('time', inplace=True)
    df_btc_data = df_btc_data.loc['2017':]
    df_btc_data = df_btc_data.at_time('00:00:00')
    a = df_btc_data.open.count()                # 1521 raws
    df_btc_data.to_csv('./data/' + 'BTCUSD_D.csv')

    # Extraction per HOUR à partir de 2017
    df_btc_data = df_btc_data.at_time(str([0-9] + ':00:00'))
    print("toto")
    # df_btc_data.to_csv('./data/' + 'BTCUSD_60.csv')


def data_preparation_ETHUSD_1_BRUT(filename):
    df_eth_data = pd.read_csv('./data/' + filename)
    print("toto")

