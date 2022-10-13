# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import timedelta, datetime

import os
import sys
import shutil
import config
import tradingBot


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lst_symbols = config.LST_CRYPTO
    lst_features = config.DATA_COLUMNS
    start_date = config.START_DATE
    interval = config.INTERVAL
    dir_data = config.DIR_DATA

    myBot = tradingBot.TradingBot(lst_symbols, lst_features, start_date, interval, dir_data)

    # Récupération des data => .csv
    myBot.tradingBot_record()

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


