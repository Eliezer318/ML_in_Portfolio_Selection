import numpy as np
import pandas as pd
import cvxpy as cp
import torch
from pypfopt import HRPOpt
from model import get_lstm, optimization_deep, return_cov_for_inference, device


def min_variance(daily_returns: pd.DataFrame, tau: float = 0):
    cov_matrix = daily_returns.iloc[-250:].cov().to_numpy()

    w = cp.Variable(cov_matrix.shape[0])
    tau_param = cp.Parameter(nonneg=True)  # w > 0

    objective = cp.Minimize(cp.quad_form(w, cov_matrix) + tau_param * cp.norm(w, 1))
    constraints = [cp.sum(w) == 1]

    prob = cp.Problem(objective, constraints)

    tau_param.value = tau
    prob.solve()  # Returns the optimal value.
    w_min_variance = prob.variables()[0].value
    return w_min_variance


def hrp(daily_returns: pd.DataFrame):
    """
    The HRP method works by finding subclusters of similar assets based on returns and constructing a hierarchy from
    these clusters to generate weights for each asset. Supposed to be less sensitive to outliers
    :param daily_returns: pct change, daily returns
    :return: weights
    """
    hrp_weights = HRPOpt(daily_returns).optimize()
    return np.array(list(hrp_weights.values()))


def deep_approach(portfolio, daily_returns: pd.DataFrame, pretrained=False) -> np.ndarray:
    """
    :param portfolio: the portfolio strategy class
    :param data: the dataframe with the full data statistics
    :param daily_returns: daily returns dataframe
    :param pretrained: whether try to get pretrained weights or to train the model
    :return:
    """
    window_size = 30
    if portfolio.model is None:
        portfolio.model = get_lstm(daily_returns, pretrained, window_size=window_size)
    portfolio.model = portfolio.model.to(device)
    portfolio.model.eval()
    data = torch.from_numpy(daily_returns.iloc[-window_size:].values).to(device, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        model_output = portfolio.model(data).squeeze(0)
        cov = return_cov_for_inference(daily_returns, window_size)[-1].to(device)
    weights = optimization_deep(cov_matrix=cov, returns=model_output)
    return weights/weights.sum()


def stock_picking(daily_returns: pd.DataFrame) -> np.ndarray:
    """
    :param daily_returns: take a uniform distribution on the k most successful stocks
    :return:
    """
    mean_return_per_stock: np.ndarray = daily_returns.values.mean(0)
    weights = np.zeros_like(mean_return_per_stock)
    indices_good = mean_return_per_stock.argsort()[-5:]
    weights[indices_good] = 1
    return weights/weights.sum()


def market(train_data: pd.DataFrame) -> np.ndarray:
    """
    :param train_data: train data
    :return: S&P500 market portfolio
    """
    return np.ones(503)/503


def uniform_approach(train_data: pd.DataFrame):
    n_cols = train_data.shape[1] // len({col for col, _ in train_data.columns})
    return np.ones(n_cols)/n_cols
