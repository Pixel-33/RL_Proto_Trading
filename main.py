# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import timedelta, datetime

import pandas as pd
import os, sys

import config
import data_preparation
from q_learning_for_trading import run_rl_trading

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    ds = data_preparation.DataDescription()
    # DEBUG CEDE
    # ds.symbols = ["ETH/EURS", "BTC/EURS"]
    # ds.features = ["open", "high", "low", "close"]
    start_date = "2020-01-01"
    data_preparation.record(ds, start_date)
    # data_preparation.get_current_data(ds)
    for symbol in ds.symbols:
        run_rl_trading(symbol)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

