import numpy as np
import pandas as pd


class Portfolio:

    def __init__(self, ):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        from utils import parse_args
        self.model = None
        self.weights = None
        self.cov_matrices = None
        self.args = parse_args()
        self.count = 0

    def train(self, train_data: pd.DataFrame):
        pass

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        # try:
        from general_methods import uniform_approach, stock_picking, min_variance, hrp, deep_approach
        # return uniform_approach(train_data)
        adj_train_data = train_data['Adj Close']
        daily_returns = adj_train_data.select_dtypes(include=np.number).pct_change().iloc[1:].fillna(0)
        daily_returns = daily_returns.dropna(axis=0, inplace=False).select_dtypes(include=np.number)

        if self.count % 1 == 0:
            self.count += 1
            self.weights = None
        if self.weights is not None:  # we don't change strategy
            return self.weights
        self.weights = deep_approach(self, daily_returns, self.args).reshape(-1)
        self.weights = self.weights / self.weights.sum()
        weights = self.weights
        for i in range(10, 3, -1):
            if not np.isclose(weights.sum(), 1):
                print(f'difference {weights.sum().round(i)}')
                weights[weights.argmax()] += 1 - weights.sum().round(i)
        return weights




        # except Exception as e:
        #     print(f'Exception raised! {e}')
        #     try:
        #         n_cols = train_data.shape[1] // len({col for col, _ in train_data.columns})
        #         return np.ones(n_cols)/n_cols
        #     except Exception as e:
        #         n_cols = train_data['Adj Close'].shape[1]
        #         return np.ones(n_cols)/n_cols

