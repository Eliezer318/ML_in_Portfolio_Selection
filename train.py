import pandas as pd
from tqdm import trange
import torch
from torch import optim
from torch.nn import functional as f
from models import Weights, MyLSTM
from utils import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimize_weights(cov_matrix: torch.Tensor, returns: torch.Tensor, args):
    """
    :param cov_matrix: relevant covariance matrix [D, D] for updated current date
    :param returns: estimated returns for the next day
    train_data: original data of stocks, including stock statistics
    :return: find optimal weights
    """

    model = Weights().to(device)
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr_w, weight_decay=args.weight_decay_w)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=5)
    weights, sharpe = 0, 0
    for _ in range(args.n_epochs_w):
        weights, sharpe = model(cov_matrix, returns)
        optimizer2.zero_grad()
        (-sharpe).backward()  # TODO add minus
        optimizer2.step()
        # scheduler2.step(-sharpe)
    # print(f'Weights: {weights.max(): .5f}, Sharpe: {sharpe.item(): .5f}')
    return weights.detach().cpu().numpy()


def train_model(model: MyLSTM, daily_returns: pd.DataFrame, args):
    # split data
    train_x, train_y = create_dataset(daily_returns, window_size=args.window_size, gamma=args.gamma)[:2]
    train_x, train_y = [x.to(device, dtype=torch.float) for x in [train_x, train_y]]

    # train the model
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_train, weight_decay=args.weight_decay_train)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    tbar = trange(args.n_epochs_train)
    # labels = (train_y > 0.001).float()
    for _ in tbar:
    # for _ in range(args.n_epochs_train):
        pred = model(train_x)  # [B, 503]
        l2_loss = ((pred - train_y) ** 2).sum(1).mean(0)
        # bce_loss = f.binary_cross_entropy(pred, labels)
        optimizer.zero_grad()
        # bce_loss.backward()
        l2_loss.backward()
        optimizer.step()
        tbar.set_description(f'Training model {l2_loss: .5f}')
        # scheduler.step(l2_loss)


def get_lstm(daily_returns: pd.DataFrame, args) -> MyLSTM:
    """
    :param daily_returns: changes in pct, daily returns
    :param pretrained: try to get pretrained weights, if not found will raise error
    :param args: arguments for the model
    :return:
    """
    path_weights = 'weights.pkl'
    model = MyLSTM(input_dim=503, hidden_size=503, num_layers=2, dropout=args.dropout_t)
    if args.pretrained:
        model.load_weights(path_weights)
    else:
        train_model(model, daily_returns, args=args)
        torch.save(model.state_dict(), path_weights)

    return model
