import pandas as pd
import os
import config
import utils

default_features = ["open", "close", "high", "low", "volume"]


class DataDescription():
    def __init__(self):
        # self.symbols = default_symbols
        self.symbols = config.LST_CRYPTO
        # self.features = default_features
        self.features = config.DATA_COLUMNS


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


def record(data_description, target="./data/", start_date="2022-06-01", interval="1h"):
    symbols = ','.join(data_description.symbols)
    symbols = symbols.replace('/','_')
    params = { "service":"history", "exchange":"ftx", "symbol":symbols, "start":start_date, "interval": interval }
    response_json = utils.fdp_request(params)
    for symbol in data_description.symbols:
        formatted_symbol = symbol.replace('/','_')
        if response_json["result"][formatted_symbol]["status"] == "ko":
            print("no data for ",symbol)
            continue
        df = pd.read_json(response_json["result"][formatted_symbol]["info"])
        # df = features.add_features(df, data_description.features)
        if not os.path.exists(target):
            os.makedirs(target)
        df.to_csv(target+'/'+formatted_symbol+".csv")
