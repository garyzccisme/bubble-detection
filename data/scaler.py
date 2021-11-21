from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Some commonly used scaling methods for Stock Data:
# https://quant.stackexchange.com/questions/9192/how-to-normalize-stock-data


class HybridScaler(TransformerMixin, BaseEstimator):
    def __init__(self, target: Dict = None):
        # target must have keys 'price', 'return', 'diff', 'volume'.
        if target is None:
            target = {
                'price': 'minmax',
                'return': 'standard',
                'diff': 'standard',
                'volume': 'standard',
            }
        self.target = target
        # Define sub scalers by target names
        self.price_scaler = self.get_scaler(target['price'])
        self.return_scaler = self.get_scaler(target['return'])
        self.diff_scaler = self.get_scaler(target['diff'])
        self.volume_scaler = self.get_scaler(target['volume'])
        self.columns = None

    def fit(self, df: pd.DataFrame, y=None):
        self.classify_cols(df.columns)
        self.price_scaler.fit(df[self.columns[0]])
        self.return_scaler.fit(df[self.columns[1]])
        self.diff_scaler.fit(df[self.columns[2]])
        self.volume_scaler.fit(df[self.columns[3]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        price_df = pd.DataFrame(self.price_scaler.transform(df[self.columns[0]]), columns=self.columns[0])
        return_df = pd.DataFrame(self.return_scaler.transform(df[self.columns[1]]), columns=self.columns[1])
        diff_df = pd.DataFrame(self.diff_scaler.transform(df[self.columns[2]]), columns=self.columns[2])
        volume_df = pd.DataFrame(self.volume_scaler.transform(df[self.columns[3]]), columns=self.columns[3])

        scaled_df = pd.concat([price_df, return_df, diff_df, volume_df], axis=1)[df.columns]
        return scaled_df

    def classify_cols(self, cols):
        price_cols, return_cols, diff_cols, volume_cols = [], [], [], []
        for col in cols:
            if 'return' in col:
                return_cols.append(col)
            elif 'diff' in col:
                diff_cols.append(col)
            elif 'volume' in col:
                volume_cols.append(col)
            else:
                price_cols.append(col)
        self.columns = [price_cols, return_cols, diff_cols, volume_cols]

    # TODO: more scalers to be added.
    @staticmethod
    def get_scaler(name):
        if name == 'minmax':
            return MinMaxScaler()
        elif name == 'standard':
            return StandardScaler()
        else:
            raise ValueError("Unknown scaler name.")






