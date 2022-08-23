import os

import pandas as pd
import pickle
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import optuna

from utils import parse_args


def get_data(START_DATE, END_TEST_DATE):
    wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp_tickers = wiki_table[0]
    tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
    data = yf.download(tickers, START_DATE, END_TEST_DATE)
    return data


def test_portfolio(args, month_name, START_DATE, END_TRAIN_DATE, END_TEST_DATE):
    os.makedirs('data_pickles', exist_ok=True)
    path = f'data_pickles/data_{START_DATE}_{END_TRAIN_DATE}_{END_TEST_DATE}.pkl'
    if os.path.isfile(path):
        full_train = pickle.load(open(path, 'rb'))
    else:
        full_train = get_data(START_DATE, END_TEST_DATE)
        pickle.dump(full_train, open(path, 'wb'))

    returns = []
    strategy = Portfolio(args)
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
    print(month_name, sharpe)
    return sharpe


def objective(trial: optuna.trial.Trial):
    gamma = 0.94
    window_size = trial.suggest_int('window_size', 5, 60)
    n_epochs_train = trial.suggest_int('n_epochs_train', 30, 500)
    lr_train = 1e-2
    weight_decay_train = trial.suggest_loguniform('weight_decay_train', 1e-5, 1e-1)
    n_epochs_w = trial.suggest_int('n_epochs_w', 30, 1200)
    lr_w = trial.suggest_loguniform('lr_w', 1e-5, 1e-3)
    weight_decay_w = trial.suggest_loguniform('weight_decay_w', 1e-5, 1e-1)

    print('tested params', trial.params)
    test_months = {'May': ('2017-08-01', '2022-04-30', '2022-05-31'),
                   'June': ('2017-08-01', '2022-05-31', '2022-06-30'),
                   'July': ('2017-08-01', '2022-06-30', '2022-07-31'),
                   'August': ('2017-08-01', '2022-07-31', '2022-08-19')}
    results = {}
    for TRAIN_MONTH, (START_DATE, END_TRAIN_DATE, END_TEST_DATE) in test_months.items():
        args = parse_args(gamma, window_size, n_epochs_train, lr_train, weight_decay_train, n_epochs_w, lr_w, weight_decay_w)
        sharpe = test_portfolio(args, TRAIN_MONTH, START_DATE, END_TRAIN_DATE, END_TEST_DATE)
        results[TRAIN_MONTH] = sharpe
    print('Last Trial')
    print(trial.params, results, sum(results.values())/len(results))
    print('Best trial till now')
    print(study1.best_trial)
    print(study1.best_value, study1.best_params)
    return sum(results.values())/len(results)


if __name__ == '__main__':
    print('with minus on sharpe')
    study1 = optuna.create_study(direction='maximize', study_name='portfolio_optimization without minus on sharpe')
    study1.optimize(objective, n_trials=150, timeout=None, catch=(ValueError, ))
    print(study1.best_params, study1.best_value)
