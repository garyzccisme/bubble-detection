from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def get_order(p_values: Iterable, d_values: Iterable, q_values: Iterable):
    orders = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                orders.append((p, d, q))
    return orders


def grid_search(series: pd.Series, search_range: Dict = None, train_ratio=0.8):
    if search_range is None:
        search_range = {
            'p_values': [0, 1, 2, 3, 4, 5, 10],
            'd_values': range(3),
            'q_values': range(5),
        }
    orders = get_order(**search_range)
    train_valid_split = int(len(series) * train_ratio)
    print(f"Train Valid Split at {series.index[train_valid_split]}")
    train_series, valid_series = series.values[:train_valid_split], series.values[train_valid_split:]

    # Start Search
    best_order, best_error = (0, 0, 0), np.inf
    for order in orders:
        error = evaluate(train_series, valid_series, order)
        print(f"ARIMA({order}), MSE = {error}")
        if error < best_error:
            best_order = order
            best_error = error
    return best_order


def evaluate(train, valid, order):
    # prepare training dataset
    history, predictions = list(train), []
    for t in range(len(valid)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = np.array(model_fit.forecast())[0]
        predictions.append(yhat)
        history.append(valid[t])
    # calculate out of sample error
    error = mean_squared_error(valid, predictions)
    return error
