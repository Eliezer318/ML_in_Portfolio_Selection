import torch
import numpy as np
import pandas as pd

import argparse
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args(gamma=0.94, window_size=30, n_epochs_train=250, lr_train=8e-4, weight_decay_train=1e-4, n_epochs_w=500,
               lr_w=1e-2, weight_decay_w=1e-8, dropout=0.8, n_assets=10):
    parser = argparse.ArgumentParser(description='ML in Portfolio Optimization')
    parser.add_argument('--window_size', type=int, default=window_size, help='window size')
    parser.add_argument('--gamma', type=float, default=gamma, help='gamma')
    parser.add_argument('--portfolio', type=str, default='portfolio.pkl', help='portfolio')
    parser.add_argument('--n_assets', type=int, default=n_assets, help='window size')
    parser.add_argument('--dropout_t', type=float, default=dropout, help='dropout in lstm model')

    # train arguments
    parser.add_argument('--n_epochs_train', type=int, default=n_epochs_train, help='number of epochs to train the LSTM model')
    parser.add_argument('--lr_train', type=float, default=lr_train, help='learning rate to train the LSTM model')
    parser.add_argument('--weight_decay_train', type=float, default=weight_decay_train, help='lr decay to train the LSTM model')

    parser.add_argument('--n_epochs_w', type=int, default=n_epochs_w, help='number of epochs for sgd on weights sharpe')
    parser.add_argument('--lr_w', type=float, default=lr_w, help='learning rate for sgd on weights sharpe')
    parser.add_argument('--weight_decay_w', type=float, default=weight_decay_w, help='lr decay for the weights optimization')

    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
    return parser.parse_args()


def strided_axis0(data, window_size):
    nd0 = data.shape[0] - window_size + 1
    m, n = data.shape
    s0, s1 = data.strides
    return np.lib.stride_tricks.as_strided(data, shape=(nd0, window_size, n), strides=(s0, s0, s1))


def batch_cov(data: torch.Tensor):
    """
    return batch covariance matrix
    """
    B, N, D = data.size()
    mean = data.mean(dim=1).unsqueeze(1)
    diffs = (data - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D).sum(dim=1)
    cov_matrices = (prods / (N - 1))
    return cov_matrices


def vectorized_adjusted_covariance(cov_matrices: torch.Tensor, gamma=0.94):
    """
    return vectorized adjusted covariance matrices
    """
    B = cov_matrices.shape[0]
    current_gammas = gamma ** torch.arange(1, B + 1)
    sigma_gammas = current_gammas.cumsum(0)
    cov = ((current_gammas.reshape(-1, 1, 1) * cov_matrices.flip(0)).cumsum(dim=0)/ sigma_gammas.reshape(-1, 1, 1)).flip(0)
    return cov.to(dtype=torch.float)


def create_dataset(df: pd.DataFrame, window_size=30, gamma=0.94, inference=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.from_numpy(strided_axis0(df.values, window_size)).float()
    train_y = torch.from_numpy(df.values[window_size::]).float()
    # if not inference:
    #     train_x = train_x[:-1]
    # train_cov, cov_matrices = None, None
    cov_matrices = batch_cov(train_x)
    train_cov = vectorized_adjusted_covariance(cov_matrices, gamma=gamma)
    return train_x, train_y, train_cov, cov_matrices


def return_cov_for_inference(portfolio, df, window_size, gamma=0.94):
    if portfolio.cov_matrices is None:
        train_x, train_y, cov_matrix, cov_matrices = create_dataset(df, window_size, gamma=gamma, inference=True)
        portfolio.cov_matrices = cov_matrices
    else:
        cov_matrices = portfolio.cov_matrices
        cov_matrices = torch.cat([cov_matrices, torch.from_numpy(df.iloc[-window_size:].cov().values).float().unsqueeze(0)], dim=0)
        cov_matrix = vectorized_adjusted_covariance(cov_matrices, gamma=gamma)
    return cov_matrix
