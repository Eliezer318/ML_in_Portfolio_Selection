import os
import pandas as pd
import yfinance as yf
from portfolio import Portfolio
import numpy as np
from tqdm import tqdm
import pickle
import os

START_DATE0, END_TRAIN_DATE0, END_TEST_DATE0 = '2017-05-31', '2022-02-23', '2022-3-23'
START_DATE1, END_TRAIN_DATE1, END_TEST_DATE1 = '2017-05-31', '2022-04-30', '2022-05-31'
START_DATE2, END_TRAIN_DATE2, END_TEST_DATE2 = '2017-05-31', '2022-05-31', '2022-06-30'
START_DATE3, END_TRAIN_DATE3, END_TEST_DATE3 = '2017-06-30', '2022-06-30', '2022-07-31'
START_DATE4, END_TRAIN_DATE4, END_TEST_DATE4 = '2017-06-30', '2022-07-31', '2022-08-17'

for START_DATE, END_TRAIN_DATE, END_TEST_DATE in [(START_DATE0, END_TRAIN_DATE0, END_TEST_DATE0),
                                                  (START_DATE1, END_TRAIN_DATE1, END_TEST_DATE1),
                                                  (START_DATE2, END_TRAIN_DATE2, END_TEST_DATE2),
                                                  (START_DATE3, END_TRAIN_DATE3, END_TEST_DATE3),
                                                  (START_DATE4, END_TRAIN_DATE4, END_TEST_DATE4)]:

    def get_data():
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        data = yf.download(tickers, START_DATE, END_TEST_DATE)
        return data


    def test_portfolio():
        os.makedirs('data_pickles', exist_ok=True)
        path = f'data_pickles/file_{START_DATE}_{END_TRAIN_DATE}_{END_TEST_DATE}.pkl'
        if os.path.isfile(path):
            full_train = pickle.load(open(path, 'rb'))
        else:
            full_train = get_data()
            pickle.dump(full_train, open(path, 'wb'))

        returns = []
        strategy = Portfolio()
        # pbar = tqdm(pd.date_range(END_TRAIN_DATE, END_TEST_DATE))
        # for test_date in pbar:
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
            # print(f'{cur_return: .5f}', end=' ')
        returns = pd.DataFrame(returns).set_index('date')
        mean_return, std_returns = float(returns.mean()), float(returns.std())
        sharpe = mean_return / std_returns
        print(f'{sharpe: .5f}, {mean_return: .5f}, {std_returns: .5f}')

    test_portfolio()
