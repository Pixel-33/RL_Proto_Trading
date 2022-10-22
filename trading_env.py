"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import config
import logging
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale
import talib
from datetime import datetime   # pour vérifier les timestamps (optionnel)
# from yahoo_fin.stock_info import *


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment
    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    """

    def __init__(self, trading_days, lst_ticker, normalize=True):
        self.lst_ticker = lst_ticker
        self.interval = config.INTERVAL
        self.trading_days = trading_days
        self.normalize = normalize
        self.lst_df_data = self.load_lst_data()
        self.lst_df_features = self.preprocess_lst_features()
        self.min_values = self.calcul_min_value()
        self.max_values = self.calcul_max_value()
        self.compteur = 0
        self.current_features = None
        self.step = 0
        self.offset = None

    def load_lst_data(self):
        lst = []
        for ticker in self.lst_ticker:
            lst.append(self.load_data(ticker))
        return lst

    def load_data(self, ticker):
        log.info('loading data for {}...'.format(ticker))
        # idx = pd.IndexSlice

        df = pd.read_csv('./data/' + ticker + '.csv', index_col=[0])

        # df.columns = config.DATA_COLUMNS
        # df['time'] = pd.to_datetime(df['time'], unit='s')
        log.info('got data for {}...'.format(ticker))

        # startdate = df.index.tolist()[0][0]
        # enddate = df.index.tolist()[len(df) - 1][0]

        # startdate = "2000-01-01"
        # enddate = "2021-01-01"
        # ticker = df.index.tolist()[0][1]
        # ticker = 'aapl'
        # data = get_data(ticker, start_date = startdate, end_date = enddate)

        data = df.copy()
        for column in data.columns:
            is_in = 0
            for data_column in ['close', 'low', 'high']:    # 'volume'
                if column == data_column:
                    is_in = 1
            if is_in == 0:
                data = data.drop([column], axis=1)

        # HEURE #####################
        if config.INTERVAL == '1h':
            # Calcul high_day, low_day, close_day
            data['high_day'] = data.high.rolling(24).max()
            data['low_day'] = data.low.rolling(24).min()
            data['close_day'] = data['close'].shift(periods=23)

        return data

    def preprocess_lst_features(self):
        lst = []
        for df_data in self.lst_df_data:
            lst.append(self.preprocess_data(df_data))
        return lst

    def preprocess_data(self, df_data):
        """calculate returns and percentiles, then removes missing values"""

        df_data['returns'] = df_data.close.pct_change()
        df_data['ret_2'] = df_data.close.pct_change(2)      # (2)
        df_data['ret_5'] = df_data.close.pct_change(4)     # (5)
        df_data['ret_10'] = df_data.close.pct_change(10)    # (10)
        df_data['ret_21'] = df_data.close.pct_change(21)    # (21)

        df_data['rsi'] = talib.STOCHRSI(df_data.close_day)[1]
        df_data['macd'] = talib.MACD(df_data.close_day)[1]
        df_data['atr'] = talib.ATR(df_data.high_day, df_data.low_day, df_data.close_day)

        slowk, slowd = talib.STOCH(df_data.high_day, df_data.low_day, df_data.close_day)
        df_data['stoch'] = slowd - slowk
        # df_data['atr'] = talib.ATR(df_data.high, df_data.low, df_data.close)
        df_data['ultosc'] = talib.ULTOSC(df_data.high_day, df_data.low_day, df_data.close_day)
        df_data = (df_data.replace((np.inf, -np.inf), np.nan)
                   .drop(['high', 'low', 'close', 'high_day', 'low_day', 'close_day'], axis=1).dropna())   # , 'volume'

        r = df_data.returns.copy()
        if self.normalize:
            df_data = pd.DataFrame(scale(df_data),
                                   columns=df_data.columns,
                                   index=df_data.index)
        features = df_data.columns.drop('returns')
        df_data['returns'] = r  # don't scale returns
        df_data = df_data.loc[:, ['returns'] + list(features)]
        print("Features ", df_data.shape, " !!!!!!!!!!")
        log.info(df_data.info())
        return df_data

    def reset(self):
        self.compteur = self.compteur % len(self.lst_df_features)
        self.current_features = self.lst_df_features[self.compteur].copy()
        self.compteur += 1

        """Provides starting index for time series and resets step"""
        high = len(self.current_features.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0
        # print("OFFSET: ", self.offset, " ##################################")

    def take_step(self):
        """Returns data for current trading day and done signal (signal de fin d'épisode)"""
        obs = self.current_features.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days     # booléen pour savoir quand fin de l'épisode
        return obs, done

    def calcul_min_value(self):
        lst_min = []
        for df_features in self.lst_df_features:
            min_value = df_features.min()
            lst_min.append(min_value)
        df_min = pd.DataFrame(lst_min)
        return df_min.min()

    def calcul_max_value(self):
        lst_max = []
        for df_features in self.lst_df_features:
            max_value = df_features.max()
            lst_max.append(max_value)
        df_max = pd.DataFrame(lst_max)
        return df_max.max()


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        toto = self.step
        titi = self.step - 1
        tutu = max(0, self.step - 1)

        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        end_position = action - 1  # short, neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward, 'nav': self.navs[self.step], 'costs': self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG

    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trading_days, trading_cost_bps, time_cost_bps, lst_ticker):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.lst_ticker = lst_ticker

        self.data_source = DataSource(trading_days=self.trading_days, lst_ticker=self.lst_ticker)
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)
        # define type and shape of our action_space
        self.action_space = spaces.Discrete(3)
        # define the observation_space, which contains all the environment’s data to be observed by the agent
        self.observation_space = spaces.Box(min(self.data_source.min_values), max(self.data_source.max_values))
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # define one step through the environment. At each step we will take the specified action (chosen by our model),
    # calculate the reward, and return the next observation.
    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0])
        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]

    # The render method may be called periodically to print a rendition of the environment
    def render(self, mode='human'):
        """Not implemented"""
        pass
