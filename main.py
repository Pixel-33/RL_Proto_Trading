# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import timedelta, datetime

import pandas as pd
import os, sys, shutil

import config
import data_preparation
import concurrent.futures
import TradingBot

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    myBot = TradingBot.TradingBot(config.LST_CRYPTO,
                                  config.DATA_COLUMNS,
                                  config.DIR_DATA,
                                  '2020-01-01',
                                  '1d')

    myBot.TradingBot_record()
    if len(sys.argv) == 2 and (sys.argv[1] == "--reboot"):
        myBot.run_rl_trading(True)
    else:
        if os.path.exists('./backup'):
            shutil.rmtree('./backup')
        myBot.run_rl_trading(False)


    '''    
    if len(sys.argv) == 2 and (sys.argv[1] == "--multi_thread"):
        # multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_THREAD) as executor:
            futures = []
            for symbol in ds.symbols:
                futures.append(
                    executor.submit(
                        run_rl_trading,
                        symbol,
                        interval,
                        start_date
                    )
                )
    else:
        for symbol in ds.symbols:
            run_rl_trading(symbol, interval, start_date)
    '''

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

