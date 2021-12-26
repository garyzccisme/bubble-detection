import numpy as np
import pandas as pd


# https://stats.stackexchange.com/questions/163054/compare-several-binary-time-series
def moving_average(true, pred, window=10, weights=None):
    df = pd.DataFrame({'true': true, 'pred': pred})
    if weights is None:
        weights = np.ones(window)
    ma_df = df.rolling(window).apply(lambda x: (x * weights).sum() / weights.sum())
    return ma_df[window:]
