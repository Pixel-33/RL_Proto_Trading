'''
Data Sources from kaggle
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data?resource=download
https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset
'''

import pandas as pd


def data_preparation_BTC(filename):
    df_btc_data = pd.read_csv('./data/' + filename)
    df_btc_data.dropna(inplace=True)                                # 4857376 => 3613769 raws
    df_btc_data.sort_values(by='Timestamp', inplace=True)           # s'assurer de l'ordre chronologique
    df_btc_data['Timestamp'] = pd.to_datetime(df_btc_data['Timestamp'], unit='s', errors='coerce')
    df_btc_data.isna().sum()                                        # s'assurer que les conversions de date sont ok
    df_btc_data.rename(columns={'Timestamp': 'time'}, inplace=True)
    df_btc_data.columns = df_btc_data.columns.str.lower()
    df_btc_data.reset_index(inplace=True, drop=True)                # réindexer suite aux raws supprimées
    df_btc_data.drop(['volume_(btc)', 'weighted_price'], axis=1, inplace=True)  # suppression des colonnes inutiles

    df_btc_data.rename({'volume_(currency)': 'volume'}, axis=1, inplace=True)

    filename_suffix = filename.split(".")[1]
    filename_prefix = filename.split(".")[0]
    crypto_pair = filename_prefix.split("_")[0]
    # interval = filename_prefix.split("_")[1]

    df_btc_data.set_index('time', inplace=True)
    df_btc_data = df_btc_data.loc['2017':]

    df_btc_data.reset_index(inplace=True)

    df_btc_data.to_csv('./data/' + crypto_pair + '_' + '1m' + '.' + filename_suffix)

    df_btc_1h = df_btc_data.copy()
    df_btc_1h['timestring'] = df_btc_1h['time'].astype(str)

    df_btc_1h.drop(df_btc_1h[~df_btc_1h['timestring'].str.endswith(":00:00")].index, inplace=True)
    df_btc_4h = df_btc_1h.copy()

    df_btc_1h.drop(['timestring'], axis=1, inplace=True)
    df_btc_1h.reset_index(inplace=True)
    df_btc_1h.to_csv('./data/' + crypto_pair + '_' + '1h' + '.' + filename_suffix)


    df_btc_4h.drop(df_btc_4h[~((df_btc_4h['timestring'].str.endswith("00:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("08:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("12:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("16:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("20:00:00"))
                               )].index
                   ,inplace=True)

    df_btc_4h.drop(['timestring'], axis=1, inplace=True)
    df_btc_4h.reset_index(inplace=True)
    df_btc_4h.to_csv('./data/' + crypto_pair + '_' + '4h' + '.' + filename_suffix)





def data_preparation_ETH(filename):
    df_btc_data = pd.read_csv('./data/' + filename)
    df_btc_data.dropna(inplace=True)
    df_btc_data.columns = df_btc_data.columns.str.lower()

    df_btc_data.sort_values(by='unix timestamp', inplace=True)
    df_btc_data.drop(['symbol'], axis=1, inplace=True)

    # df_btc_data['timestamp'] = pd.to_datetime(df_btc_data['unix timestamp'], unit='s', errors='coerce')
    # df_btc_data.isna().sum()                                        # s'assurer que les conversions de date sont ok

    df_btc_data.rename(columns={'date': 'time'}, inplace=True)

    df_btc_data.reset_index(inplace=True, drop=True)                # réindexer suite aux raws supprimées
    df_btc_data.drop(['unix timestamp'], axis=1, inplace=True)  # suppression des colonnes inutiles
    # df_btc_data.rename({'volume_(currency)': 'volume'}, axis=1, inplace=True)

    filename_suffix = filename.split(".")[1]
    filename_prefix = filename.split(".")[0]
    crypto_pair = filename_prefix.split("_")[0]
    # interval = filename_prefix.split("_")[1]

    df_btc_data.set_index('time', inplace=True)
    df_btc_data = df_btc_data.loc['2017':]

    df_btc_data.reset_index(inplace=True)

    df_btc_data.to_csv('./data/' + crypto_pair + '_' + '1m' + '.' + filename_suffix)

    df_btc_1h = df_btc_data.copy()
    df_btc_1h['timestring'] = df_btc_1h['time'].astype(str)

    df_btc_1h.drop(df_btc_1h[~df_btc_1h['timestring'].str.endswith(":00:00")].index, inplace=True)
    df_btc_4h = df_btc_1h.copy()

    df_btc_1h.drop(['timestring'], axis=1, inplace=True)
    df_btc_1h.reset_index(inplace=True)
    df_btc_1h.to_csv('./data/' + crypto_pair + '_' + '1h' + '.' + filename_suffix)


    df_btc_4h.drop(df_btc_4h[~((df_btc_4h['timestring'].str.endswith("00:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("08:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("12:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("16:00:00"))
                               | (df_btc_4h['timestring'].str.endswith("20:00:00"))
                               )].index
                   ,inplace=True)

    df_btc_4h.drop(['timestring'], axis=1, inplace=True)
    df_btc_4h.reset_index(inplace=True)
    df_btc_4h.to_csv('./data/' + crypto_pair + '_' + '4h' + '.' + filename_suffix)
