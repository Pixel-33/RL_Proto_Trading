import data_preparation
import q_learning_for_trading
import os

class TradingBot():
    def __init__(self, lst_symbols, lst_features, dir_data, strat_date, interval):
        self.ds = data_preparation.DataDescription(lst_symbols, lst_features)
        self.start_date = strat_date
        self.dir_data = dir_data
        self.interval = interval

    def TradingBot_record(self):
        data_preparation.record(self.ds, self.dir_data, self.start_date, self.interval)

    def run_rl_trading(self, reboot):
        for symbol in self.ds.symbols:
            q_learning_for_trading.run_rl_trading(symbol, self.interval, self.start_date, reboot)



