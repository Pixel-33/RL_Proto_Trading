# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datetime import timedelta, datetime

import pandas as pd

from data_preparation import data_preparation_BTCUSD


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_preparation_BTCUSD("BTCUSD_1_brut.csv")
    # data_preparation_ETHUSD("ETHUSD_1_brut.csv")

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

