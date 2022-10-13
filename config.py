CRITIC_ON = False
# COLAB = False

LST_CRYPTO = ['BTC_USDT']   # 'ETH_USDT',

DATA_COLUMNS = ['open',  'high', 'low', 'close', 'volume']

DIR_DATA = './data/'

START_DATE = '2021-09-29'

INTERVAL = '1h'

FDP_URL = 'https://fdp-ifxcxetwza-uc.a.run.app/'

# DDQN Agent
TRADING_DAYS = 250
MAX_EPISODES = 20
TRADING_COST_BPS = 1e-3
TIME_COST_BPS = 1e-4
