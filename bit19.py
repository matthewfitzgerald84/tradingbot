import ccxt
import numpy as np
import os
import time
import pandas as pd
import talib
import yfinance as yf
import logging
import configparser
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from typing import List, Tuple

MIN_TRADE_SIZE = {
    'XRP/BTC': 0.00020000,
    'LTC/BTC': 0.00020000,
    'ETH/BTC': 0.00020000,
    'BCH/BTC': 0.00020000,
    'XLM/BTC': 0.00020000,
}
DEFAULT_MIN_TRADE_SIZE = 0.00020000

bitstamp = ccxt.bitstamp()
symbol = 'BTC/USD'
short_window = 20
long_window = 50

def calculate_moving_averages(exchange, symbol, interval='1h', short_window=20, long_window=50):
    ohlcv = exchange.fetch_ohlcv(symbol, interval)
    close_prices = np.array([x[4] for x in ohlcv])

    short_mavg = np.convolve(close_prices, np.ones(short_window), 'valid') / short_window
    long_mavg = np.convolve(close_prices, np.ones(long_window), 'valid') / long_window

    return short_mavg[-1], long_mavg[-1]

short_mavg, long_mavg = calculate_moving_averages(bitstamp, symbol, '1h', short_window, long_window)

print(f"Short moving average: {short_mavg}")
print(f"Long moving average: {long_mavg}")

if short_mavg > long_mavg:
    print("Buy signal")
else:
    print("Sell signal")
short_mavg, long_mavg = calculate_moving_averages(bitstamp, symbol, '1h', short_window, long_window)
print(short_mavg, long_mavg)

logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    os.environ["BITSTAMP_API_KEY"] = config.get("bitstamp", "api_key")
    os.environ["BITSTAMP_SECRET_KEY"] = config.get("bitstamp", "secret_key")

def place_order(exchange, symbol, side, amount, price, order_type='limit'):
    order = exchange.create_order(symbol, order_type, side, amount, price)
    return order

    if balance >= (amount * price):
        order = place_order(bitstamp, symbol, 'buy', amount, price)

def initialize_bitstamp(api_key, secret_key):
    exchange = ccxt.bitstamp({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
    })
    return exchange

def get_symbols(exchange):
    return [symbol for symbol in exchange.load_markets() if symbol in ['XRP/BTC', 'LTC/BTC', 'ETH/BTC', 'BCH/BTC', 'XLM/BTC']]

def get_top_symbols(symbols, n=5):
    return symbols[:n]

def fetch_ohlcv(exchange, symbol, timeframe='1d', since=None):
    return exchange.fetch_ohlcv(symbol, timeframe, since=since)

def ohlcv_to_dataframe(ohlcv):
    header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return pd.DataFrame(ohlcv, columns=header)

def force_symmetric(matrix):
    return (matrix + matrix.T) / 2

def get_trading_fee(exchange, symbol):
    try:
        market = exchange.markets[symbol]
        return market['taker']
    except KeyError:
        logging.warning(f"Unable to fetch trading fee for {symbol}")
        return None

def get_available_markets(exchange):
    markets = exchange.load_markets()
    available_markets = [market['symbol'] for market in markets.values()]
    return available_markets

exchange = ccxt.bitstamp({
    'apiKey': 'KhokTjSYDBlBP2gxGjWhHv6GhwQqKJO9',
    'secret': 'WBiyDiRRR3BcahYLkaf7ugcCLzECAceM',
    'uid': 'cfya4237',
})

available_markets = get_available_markets(exchange)
print(available_markets)

def calculate_indicators(df):
    df['rsi'] = talib.RSI(df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    return df

def moving_average_crossover_strategy(symbol, interval='1h', short_window=20, long_window=50):
    short_mavg, long_mavg = calculate_moving_averages(symbol, interval, short_window, long_window)

    if short_mavg > long_mavg:
        return 'buy'
    elif short_mavg < long_mavg:
        return 'sell'
    else:
        return None

def drop_non_numeric_columns(df):
    return df[df.select_dtypes(include=[np.number]).columns.tolist()]

def calculate_returns(df, symbol):
    returns = np.log(df['close'] / df['close'].shift(1))
    return returns.dropna()

def optimize_portfolio(returns_df, symbols):
    missing_values = returns_df.isna().sum().sum()
    if missing_values > 0:
        print(f"There are {missing_values} missing values in the DataFrame.")
        # Drop missing values or fill with suitable values
        returns_df = returns_df.dropna()
        # Or
        # returns_df = returns_df.fillna(method='ffill')

    expected_returns_annual = expected_returns.mean_historical_return(returns_df)

    cov_matrix = returns_df.cov()

    cov_matrix = cov_matrix.round(8)

    ef = EfficientFrontier(expected_returns_annual, cov_matrix)

    weights = ef.min_volatility()

    cleaned_weights = ef.clean_weights()
    weight_dict = dict(zip(symbols, cleaned_weights.values()))

    return weight_dict

def get_trade_signal(df):
    last_row = df.iloc[-1]
    if last_row['rsi'] < 35 and last_row['macd_hist'] > 0:
        return 'buy'
    elif last_row['rsi'] > 65 and last_row['macd_hist'] < 0:
        return 'sell'
    else:
        return None

def get_precision(exchange, symbol):
    market_data = exchange.markets[symbol]
    return market_data['precision']['amount']

def get_minimum_trade_amount(exchange, symbol):
    markets_info = exchange.load_markets()
    market_info = markets_info[symbol]
    return market_info.get('limits', {}).get('amount', {}).get('min', 0)

def fetch_last_prices(exchange: ccxt.Exchange, symbols: List[str]) -> List[Tuple[str, float]]:
    last_prices = []
    for symbol in symbols:
        ticker = exchange.fetch_ticker(symbol)
        last_prices.append((symbol, ticker['last']))
    return last_prices

def fetch_balances(exchange):
    balances = exchange.fetch_balance()
    available_balances = {}

    for currency, balance in balances['free'].items():
        if balance > 0:
            available_balances[currency] = balance

    return available_balances

def round_amount(exchange, symbol, amount):
    market = exchange.market(symbol)
    precision = market['precision']['amount']
    return exchange.amount_to_precision(symbol, amount)

def execute_trade(exchange, trade_type, symbol, amount):
    asset = symbol.split('/')[0]
    balance = exchange.fetch_balance()
    available_balance = balance[asset]['free']

    if trade_type == 'sell' and amount > available_balance:
        amount = available_balance

    if amount <= 0:
        print(f"Skipping trade with zero amount: trade_type: {trade_type}, symbol: {symbol}")
        return

    market = exchange.market(symbol)
    min_cost = market['limits']['cost']['min']

    price = fetch_last_price(exchange, symbol)
    trade_cost = amount * price

    if trade_cost < min_cost:
        print(f"Skipping trade: trade_cost ({trade_cost}) is below the minimum required ({min_cost})")
        return

    rounded_amount = round_amount(exchange, symbol, amount)
    print(f"trade_type: {trade_type}, symbol: {symbol}, rounded_amount: {rounded_amount}")

    order = exchange.create_market_order(symbol, trade_type, rounded_amount)
    print(order)

def get_current_distribution(exchange, symbols):
    balance = exchange.fetch_balance()
    total_btc_value = sum([balance[symbol.split('/')[0]]['free'] for symbol in symbols]) + balance['BTC']['free']
    current_distribution = {symbol: balance[symbol.split('/')[0]]['free'] / total_btc_value for symbol in symbols}
    return current_distribution

def should_rebalance(current_distribution, optimized_distribution, threshold=0.01):
    for symbol in current_distribution.keys():
        if abs(current_distribution[symbol] - optimized_distribution[symbol]) > threshold:
            return True
    return False

def calculate_current_weight(exchange, symbol, balances):
    asset = symbol.split('/')[0]

    if asset not in balances:
        return 0

    asset_balance = balances[asset]
    btc_balance = balances['BTC']

    last_prices = fetch_last_prices(exchange, [symbol])
    asset_price = last_prices[0][1]

    if asset_price is None:
        return 0

    asset_value_in_btc = asset_balance * asset_price

    if btc_balance + asset_value_in_btc == 0:
        return 0

    current_weight = asset_value_in_btc / (btc_balance + asset_value_in_btc)
    return current_weight

def fetch_historical_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1d', limit: int = 500) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower

    df['SAR'] = talib.SAR(df['high'], df['low'])

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    return df

def calculate_trade_signal(exchange, symbol):
    ohlc = exchange.fetch_ohlcv(symbol, '1h', limit=200)
    close_prices = [x[4] for x in ohlc]

    short_sma = sum(close_prices[-20:]) / 20
    long_sma = sum(close_prices[-50:]) / 50

    return 'buy' if short_sma > long_sma else 'sell'

def rebalance_portfolio(exchange, top_symbols, optimized_portfolio):
    raw_balances = fetch_balances(exchange)
    balances = {k: v for k, v in raw_balances.items() if k in ['BTC'] + [s.split('/')[0] for s in top_symbols]}
    btc_balance = balances['BTC']

    for symbol, target_weight in optimized_portfolio.items():
        trade_signal = calculate_trade_signal(exchange, symbol)

        current_weight = calculate_current_weight(exchange, symbol, balances)
        asset_balance = balances.get(symbol.split('/')[0], 0)
        print(f"Symbol: {symbol}")
        print(f"New weight: {target_weight}")
        print(f"Current weight: {current_weight}")
        print(f"BTC balance: {btc_balance}")
        print(f"Asset balance: {asset_balance}")
        print(f"Last price: {fetch_last_price(exchange, symbol)}")
        print("\n")

        if trade_signal == 'buy':
            if target_weight > current_weight:
                amount_to_buy = (btc_balance * target_weight) - (btc_balance * current_weight)
                execute_trade(exchange, 'buy', symbol, amount_to_buy)
        elif trade_signal == 'sell':
            if target_weight < current_weight:
                amount_to_sell = (btc_balance * current_weight) - (btc_balance * target_weight)
                execute_trade(exchange, 'sell', symbol, amount_to_sell)

    print("Portfolio rebalanced.")

def fetch_technical_indicators(df):
    df['rsi'] = talib.RSI(df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'], timeperiod=20)
    return df

def updated_optimize_portfolio(returns_df, top_symbols):
    expected_returns_annual = expected_returns.mean_historical_return(returns_df)

    cov_matrix = returns_df.cov()

    cov_matrix = cov_matrix.round(8)

    ef = EfficientFrontier(expected_returns_annual, cov_matrix)

    weights = ef.min_volatility()

    cleaned_weights = ef.clean_weights()
    weight_dict = dict(zip(top_symbols, cleaned_weights.values()))

    pe_ratios = [fetch_fundamental_data(symbol).get('trailingPE', np.nan) if fetch_fundamental_data(symbol) is not None else np.nan for symbol in top_symbols]

    median_pe_ratio = np.nanmedian(pe_ratios)
    pe_ratios = [pe if not np.isnan(pe) else median_pe_ratio for pe in pe_ratios]

    ep_ratios = [1 / pe for pe in pe_ratios]

    ep_ratios_sum = sum(ep_ratios)
    normalized_ep_ratios = [ep / ep_ratios_sum for ep in ep_ratios]

    fundamental_weight_factor = 0.5  # Adjust this value to control the influence of fundamental factors
    combined_weights = [
        (1 - fundamental_weight_factor) * technical_weight + fundamental_weight_factor * fundamental_weight
        for technical_weight, fundamental_weight in zip(cleaned_weights.values(), normalized_ep_ratios)
    ]

    fundamental_data_list = [fetch_fundamental_data(symbol) for symbol in top_symbols]
    pe_ratios = [data.get('trailingPE', np.nan) if data is not None else np.nan for data in fundamental_data_list]
   
    combined_weights_sum = sum(combined_weights)
    normalized_combined_weights = [weight / combined_weights_sum for weight in combined_weights]

    weight_dict.update(dict(zip(top_symbols, normalized_combined_weights)))

    return weight_dict

def fetch_aseet_fundamental_data(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info

    fundamental_data = {
        "marketCap": info.get("marketCap", None),
        "dividendYield": info.get("dividendYield", None),
        "priceToEarningsRatio": info.get("trailingPE", None)
    }
    return fundamental_data

def fetch_fundamental_data(asset_symbol):
    try:
        ticker = yf.Ticker(asset_symbol)
        info = ticker.info
    except AttributeError:
        print(f"Error fetching fundamental data for {asset_symbol}")
        info = None

    return info

def fetch_last_price(exchange, symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

def main():
    selected_symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD']

    returns_df = pd.DataFrame()

    for symbol in selected_symbols:
        symbol_returns = calculate_returns(bitstamp, symbol)
        returns_df[symbol] = symbol_returns

    optimized_portfolio = optimize_portfolio(returns_df, selected_symbols)

    print("Optimized Portfolio:")
    print(optimized_portfolio)
    load_config("config.txt")

    api_key = os.environ.get("BITSTAMP_API_KEY")
    secret_key = os.environ.get("BITSTAMP_SECRET_KEY")

    exchange = initialize_bitstamp(api_key, secret_key)
    symbols = get_symbols(exchange)
    top_symbols = get_top_symbols(symbols, n=5)

    trading_pairs = ['BTC/USD', 'ETH/USD', 'XLM/USD']
    trading_interval = '1h'
    short_window = 20
    long_window = 50

exchange = ccxt.bitstamp({
    'apiKey': os.environ.get("BITSTAMP_API_KEY"),
    'secret': os.environ.get("BITSTAMP_SECRET_KEY"),
})

first_run = True

while True:
        returns_df = pd.DataFrame()
        optimized_portfolio = optimize_portfolio()

        for symbol in top_symbols:
            from ccxt.base.exchange import Exchange

            days_back = 200
            timeframe_sec = Exchange.parse_timeframe('1d')
            since = exchange.milliseconds() - days_back * timeframe_sec * 1000
            ohlcv = fetch_ohlcv(exchange, symbol, timeframe='1d', since=since)

            df = ohlcv_to_dataframe(ohlcv)
            df = drop_non_numeric_columns(df)
            
            df = fetch_technical_indicators(df)
            
            asset_symbol = symbol.split('/')[0]
            fundamental_data = fetch_fundamental_data(asset_symbol)
            
            returns = calculate_returns(df, symbol).dropna()
            returns.name = symbol
            returns_df = pd.concat([returns_df, returns], axis=1)

        optimized_portfolio = updated_optimize_portfolio(returns_df, top_symbols)

        if first_run:
            print("Initial optimized portfolio:")
            print(optimized_portfolio)
            first_run = False
        else:
            current_distribution = get_current_distribution(exchange, top_symbols)
            if should_rebalance(current_distribution, optimized_portfolio):
                rebalance_portfolio(exchange, top_symbols, optimized_portfolio)
                print("New optimized portfolio:")
                print(optimized_portfolio)
            else:
                print("No rebalancing needed.")
                print("Current distribution:")
                print(current_distribution)
                print("Optimized portfolio:")
                print(optimized_portfolio)

            time.sleep(60)

            for trading_pair in trading_pairs:
                signal = moving_average_crossover_strategy(trading_pair, trading_interval, short_window, long_window)

                ticker = bitstamp.fetch_ticker(trading_pair)
                current_price = ticker['ask']

                trade_amount_in_btc = 0.00020000
                trade_amount_in_asset = (trade_amount_in_btc / current_price) * optimized_portfolio[trading_pair]

                if signal == 'buy':
                    execute_buy_order(trading_pair, trade_amount_in_asset)
                elif signal == 'sell':
                    execute_sell_order(trading_pair, trade_amount_in_asset)

            time.sleep(60 * 60) 

if __name__ == "__main__":
    main()