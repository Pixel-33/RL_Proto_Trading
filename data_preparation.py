import pandas as pd
import os
import config
import utils

default_features = ["open", "close", "high", "low", "volume"]


class DataDescription:
    def __init__(self, lst_symbols, lst_features):
        self.symbols = lst_symbols
        self.features = lst_features


def get_current_data(data_description):
    symbols = ','.join(data_description.symbols)
    symbols = symbols.replace('/', '_')
    params = {"service": "history", "exchange": "ftx", "symbol": symbols, "start": "2019-01-01", "interval": "1h"}
    response_json = utils.fdp_request(params)

    data = {feature: [] for feature in data_description.features}
    data["symbol"] = []

    if response_json["status"] == "ok":
        for symbol in data_description.symbols:
            formatted_symbol = symbol.replace('/', '_')
            df = pd.read_json(response_json["result"][formatted_symbol]["info"])
            # df = features.add_features(df, data_description.features)
            columns = list(df.columns)

            data["symbol"].append(symbol)
            for feature in data_description.features:
                if feature not in columns:
                    return None
                data[feature].append(df[feature].iloc[-1])

    df_result = pd.DataFrame(data)
    df_result.set_index("symbol", inplace=True)
    return df_result


def record(ds, dir_data, start_date, interval):
    str_symbols = ','.join(ds.symbols)
    params = {"service": "history", "exchange": "ftx", "symbol": str_symbols, "start": start_date, "interval": interval}
    response_json = utils.fdp_request(params)
    for symbol in ds.symbols:
        if response_json["result"][symbol]["status"] == "ko":
            print("no data for ", symbol)
            continue
        df = pd.read_json(response_json["result"][symbol]["info"])

        # pour vérifier les dates au format datetime de pandas
        # df['index'] = pd.to_datetime(df.index, unit='h', origin=pd.Timestamp(start_date))
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # pour déterminer si besoin, le nombre de lignes à garder dans nos data renvoyés par ftx
        # df = df.iloc[:2100, :]

        if not os.path.exists(dir_data):
            os.makedirs(dir_data)

        df.to_csv(dir_data + symbol + ".csv")

