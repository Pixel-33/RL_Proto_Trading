# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import timedelta, datetime

import pandas as pd
import os, sys

import config
import data_preparation
import concurrent.futures
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

    data_preparation.record(ds, config.DIR_DATA, start_date)
    # data_preparation.get_current_data(ds)

    if len(sys.argv) == 2 and (sys.argv[1] == "--multi_thread"):
        # multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_THREAD) as executor:
            futures = []
            for symbol in ds.symbols:
                futures.append(
                    executor.submit(
                        run_rl_trading,
                        symbol,
                    )
                )
    else:
        for symbol in ds.symbols:
            run_rl_trading(symbol)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

