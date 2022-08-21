import pandas as pd
from tqdm import trange
import torch
from torch import optim
from models import Weights, MyLSTM
from utils import create_dataset, get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_args()


def optimize_weights(cov_matrix: torch.Tensor, returns: torch.Tensor):
    """
    :param cov_matrix: relevant covariance matrix [D, D] for updated current date
    :param returns: estimated returns for the next day
    train_data: original data of stocks, including stock statistics
    :return: find optimal weights
    """

    model = Weights().to(device)
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_w, weight_decay=args.lr_decay_w)
    weights, sharpe = 0, 0
    for _ in range(args.n_epochs_w):
        weights, sharpe = model(cov_matrix, returns)
        optimizer2.zero_grad()
        (-sharpe).backward()
        optimizer2.step()
    print(f'Weights: {weights.max(): .5f}, Sharpe: {sharpe.item(): .5f}')
    return weights.detach().cpu().numpy()


def train_model(model: MyLSTM, daily_returns: pd.DataFrame, window_size, n_epochs=10):
    # split data
    train_x, train_y = create_dataset(daily_returns, window_size=window_size, gamma=args.gamma)[:2]
    train_x, train_y = [x.to(device, dtype=torch.float) for x in [train_x, train_y]]

    # train the model
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_train, weight_decay=args.lr_decay_train)
    tbar = trange(n_epochs)
    for _ in tbar:
    # for _ in range(n_epochs):
        pred = model(train_x)  # [B, 503]
        l2_loss = ((pred - train_y) ** 2).sum(1).mean(0)
        optimizer.zero_grad()
        l2_loss.backward()
        optimizer.step()
        tbar.set_description(f'Training model {l2_loss: .5f}')


def get_lstm(daily_returns: pd.DataFrame, pretrained=False, window_size=30) -> MyLSTM:
    """
    :param data: original data of stocks, including stock statistics
    :param daily_returns: changes in pct, daily returns
    :param pretrained: try to get pretrained weights, if not found will raise error
    :param window_size:
    :return:
    """
    path_weights = 'weights.pkl'
    model = MyLSTM(input_dim=503, hidden_size=503, num_layers=2, dropout=0.5)
    if pretrained:
        model.load_weights(path_weights)
    else:
        train_model(model, daily_returns, window_size, n_epochs=args.n_epochs_train)
        torch.save(model.state_dict(), path_weights)

    return model
