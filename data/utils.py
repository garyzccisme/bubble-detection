import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from data.scaler import HybridScaler


def sliding_window(df: pd.DataFrame, window_len: int, step_size: int = 1):
    indices, windows = [], []
    for i in range(0, df.shape[0] - window_len + 1, step_size):
        indices.append(df.index[i + window_len - 1])
        windows.append(df.iloc[i:i + window_len].values)
    return indices, np.stack(windows)

