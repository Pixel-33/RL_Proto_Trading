import data_preparation
import q_learning_for_trading


class TradingBot:
    def __init__(self, lst_symbols, lst_features, start_date, interval, dir_data):
        self.ds = data_preparation.DataDescription(lst_symbols, lst_features)
        self.lst_symbols = lst_symbols
        self.start_date = start_date
        self.interval = interval
        self.dir_data = dir_data

    def tradingBot_record(self):
        data_preparation.record(self.ds, self.dir_data, self.start_date, self.interval)

    def run_rl_trading(self, reboot):
        q_learning_for_trading.run_rl_trading(self.lst_symbols, self.start_date, self.interval, reboot)



