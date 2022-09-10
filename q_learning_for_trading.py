#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning for Trading - Deep Q-learning & the stock market

# To train a trading agent, we need to create a market environment that provides price and other information, offers trading-related actions, and keeps track of the portfolio to reward the agent accordingly.

# ## How to Design an OpenAI trading environment

# The OpenAI Gym allows for the design, registration, and utilization of environments that adhere to its architecture, as described in its [documentation](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym). The [trading_env.py](trading_env.py) file implements an example that illustrates how to create a class that implements the requisite `step()` and `reset()` methods.
# 
# The trading environment consists of three classes that interact to facilitate the agent's activities:
#  1. The `DataSource` class loads a time series, generates a few features, and provides the latest observation to the agent at each time step. 
#  2. `TradingSimulator` tracks the positions, trades and cost, and the performance. It also implements and records the results of a buy-and-hold benchmark strategy. 
#  3. `TradingEnvironment` itself orchestrates the process. 

# The book chapter explains these elements in more detail.

# ## A basic trading game

# To train the agent, we need to set up a simple game with a limited set of options, a relatively low-dimensional state, and other parameters that can be easily modified and extended.
# 
# More specifically, the environment samples a stock price time series for a single ticker using a random start date to simulate a trading period that, by default, contains 252 days, or 1 year. The state contains the (scaled) price and volume, as well as some technical indicators like the percentile ranks of price and volume, a relative strength index (RSI), as well as 5- and 21-day returns. The agent can choose from three actions:
# 
# - **Buy**: Invest capital for a long position in the stock
# - **Flat**: Hold cash only
# - **Sell short**: Take a short position equal to the amount of capital
# 
# The environment accounts for trading cost, which is set to 10bps by default. It also deducts a 1bps time cost per period. It tracks the net asset value (NAV) of the agent's portfolio and compares it against the market portfolio (which trades frictionless to raise the bar for the agent).

# We use the same DDQN agent and neural network architecture that successfully learned to navigate the Lunar Lander environment. We let exploration continue for 500,000 time steps (~2,000 1yr trading periods) with linear decay of ε to 0.1 and exponential decay at a factor of 0.9999 thereafter.

# ## Imports & Settings

# ### Imports

# In[3]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import config


# get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam

import gym
from gym.envs.registration import register

from ddqn_agent import DDQNAgent

# ### Settings

# In[5]:
def run_rl_trading():
    np.random.seed(42)
    tf.random.set_seed(42)

    sns.set_style('whitegrid')

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')

    results_path = Path('results', 'trading_bot')
    if not results_path.exists():
        results_path.mkdir(parents=True)

    def format_time(t):
        m_, s = divmod(t, 60)
        h, m = divmod(m_, 60)
        return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

    trading_days = 252

    register(
        id='trading-v0',
        entry_point='trading_env:TradingEnvironment',
        max_episode_steps=trading_days
    )

    trading_cost_bps = 1e-3
    time_cost_bps = 1e-4

    f'Trading costs: {trading_cost_bps:.2%} | Time costs: {time_cost_bps:.2%}'


    trading_environment = gym.make('trading-v0', ticker=config.CRYPTO)
    # trading_environment = gym.make('trading-v0', ticker='DIS')
    trading_environment.env.trading_days = trading_days
    trading_environment.env.trading_cost_bps = trading_cost_bps
    trading_environment.env.time_cost_bps = time_cost_bps
    trading_environment.env.ticker = config.CRYPTO
    # trading_environment.env.ticker = 'DIS'
    trading_environment.seed(42)

    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n
    max_episode_steps = trading_environment.spec.max_episode_steps

    # ## Define hyperparameters
    gamma = .99,  # discount factor
    tau = 100  # target network update frequency


    # ### NN Architecture
    architecture = (256, 256)  # units per layer
    learning_rate = 0.0001  # learning rate
    l2_reg = 1e-6  # L2 regularization


    # ### Experience Replay
    replay_capacity = int(1e6)
    batch_size = 4096


    # ### $\epsilon$-greedy Policy
    epsilon_start = 1.0
    epsilon_end = .01
    epsilon_decay_steps = 250
    epsilon_exponential_decay = .99


    # ## Create DDQN Agent
    tf.keras.backend.clear_session()

    ddqn = DDQNAgent(state_dim=state_dim,
                     num_actions=num_actions,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     epsilon_start=epsilon_start,
                     epsilon_end=epsilon_end,
                     epsilon_decay_steps=epsilon_decay_steps,
                     epsilon_exponential_decay=epsilon_exponential_decay,
                     replay_capacity=replay_capacity,
                     architecture=architecture,
                     l2_reg=l2_reg,
                     tau=tau,
                     batch_size=batch_size)

    ddqn.online_network.summary()

    total_steps = 0
    max_episodes = 1000

    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    # ## Visualization
    def track_results(episode, nav_ma_100, nav_ma_10,
                      market_nav_100, market_nav_10,
                      win_ratio, total, epsilon):
        time_ma = np.mean([episode_time[-100:]])
        T = np.sum(episode_time)

        template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
        template += 'Market: {:>6.1%} ({:>6.1%}) | '
        template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
        print(template.format(episode, format_time(total),
                              nav_ma_100-1, nav_ma_10-1,
                              market_nav_100-1, market_nav_10-1,
                              win_ratio, epsilon))


    # ## Train Agent
    start = time()
    results = []
    for episode in range(1, max_episodes + 1):
        this_state = trading_environment.reset()
        for episode_step in range(max_episode_steps):
            action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            next_state, reward, done, _ = trading_environment.step(action)

            ddqn.memorize_transition(this_state,
                                     action,
                                     reward,
                                     next_state,
                                     0.0 if done else 1.0)
            if ddqn.train:
                # print("episode: ",episode ," - episode_step: ", episode_step)
                ddqn.experience_replay()
            if done:
                break
            this_state = next_state

        # get DataFrame with seqence of actions, returns and nav values
        result = trading_environment.env.simulator.result()

        # get results of last step
        final = result.iloc[-1]

        # apply return (net of cost) of last action to last starting nav
        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        # market nav
        market_nav = final.market_nav
        market_navs.append(market_nav)

        # track difference between agent an market NAV results
        diff = nav - market_nav
        diffs.append(diff)

        if episode % 10 == 0:
            track_results(episode,
                          # show mov. average results for 100 (10) periods
                          np.mean(navs[-100:]),
                          np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]),
                          np.mean(market_navs[-10:]),
                          # share of agent wins, defined as higher ending nav
                          np.sum([s > 0 for s in diffs[-100:]])/min(len(diffs), 100),
                          time() - start, ddqn.epsilon)
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break

    trading_environment.close()

    # ### Store Results
    results = pd.DataFrame({'Episode': list(range(1, episode+1)),
                            'Agent': navs,
                            'Market': market_navs,
                            'Difference': diffs}).set_index('Episode')

    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()

    results.to_csv(results_path / 'results.csv', index=False)

    with sns.axes_style('white'):
        sns.distplot(results.Difference)
        sns.despine()


    # ### Evaluate Results
    results.info()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    df1 = (results[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0],
             title='Annual Returns (Moving Average)',
             lw=1)

    df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=axes[1],
             title='Agent Outperformance (%, Moving Average)')

    for ax in axes:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    axes[1].axhline(.5, ls='--', c='k', lw=1)

    sns.despine()
    fig.tight_layout()
    fig.savefig(results_path / 'performance', dpi=300)
