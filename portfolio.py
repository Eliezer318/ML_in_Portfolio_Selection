import numpy as np
import pandas as pd


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.model = None
        self.weights = None
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
        from methods import uniform_approach, stock_picking, min_variance, hrp, deep_approach
        # return uniform_approach(train_data)
        adj_train_data = train_data['Adj Close']
        daily_returns = adj_train_data.select_dtypes(include=np.number).pct_change().iloc[1:].fillna(0)
        daily_returns = daily_returns.dropna(axis=0, inplace=False).select_dtypes(include=np.number)

        if self.count % 1 == 0:
            self.count += 1
            self.weights = None

        if self.weights is not None:  # we don't change strategy
            return self.weights

        self.weights = deep_approach(self, daily_returns)
        self.weights = self.weights / self.weights.sum()
        return self.weights
        # except Exception as e:
        #     print(f'Exception raised! {e}')
        #     try:
        #         n_cols = train_data.shape[1] // len({col for col, _ in train_data.columns})
        #         return np.ones(n_cols)/n_cols
        #     except Exception as e:
        #         n_cols = train_data['Adj Close'].shape[1]
        #         return np.ones(n_cols)/n_cols


""" 
UNIFORM
 0.14097,  0.00222,  0.01577
 0.06659,  0.00128,  0.01917
-0.25068, -0.00460,  0.01834
 0.33183,  0.00378,  0.01140
 0.44415,  0.00378,  0.00852

MIN VAR
 0.13134,  0.00171,  0.01302
 0.15129,  0.00189,  0.01252
-0.34364, -0.00305,  0.00887
 0.35971,  0.00323,  0.00899
 0.24235,  0.00278,  0.01148
 
MIN_Var + 2 * Winning Team
0.07061,  0.00262,  0.03714
-0.11980, -0.00343,  0.02861
0.51756,  0.01018,  0.01967
0.19138,  0.00311,  0.01626
0.27817,  0.00845,  0.03036

Winning Team
0.05194,  0.00276,  0.05305
-0.09316, -0.00368,  0.03947
0.47675,  0.01353,  0.02838
0.19881,  0.00436,  0.02194
0.27179,  0.01215,  0.04471
"""


"""
Min Variance 0.3506356268530881
Uniform -0.3194586438512036


"""