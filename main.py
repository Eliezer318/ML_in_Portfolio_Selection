import os

import pandas as pd
import pickle
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import re
from datetime import datetime

START_DATE = '2017-08-01'
END_TRAIN_DATE = '2021-12-31'
END_TEST_DATE = '2022-01-31'


def get_data():
    wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp_tickers = wiki_table[0]
    tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
    data = yf.download(tickers, START_DATE, END_TEST_DATE)
    return data


def test_portfolio():
    os.makedirs('data_pickles', exist_ok=True)
    path = f'data_pickles/data_{START_DATE}_{END_TRAIN_DATE}_{END_TEST_DATE}.pkl'
    if os.path.isfile(path):
        full_train = pickle.load(open(path, 'rb'))
    else:
        full_train = get_data()
        pickle.dump(full_train, open(path, 'wb'))

    returns = []
    strategy = Portfolio()
    for test_date in pd.date_range(END_TRAIN_DATE, END_TEST_DATE):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    print(sharpe)


if __name__ == '__main__':
    test_portfolio()