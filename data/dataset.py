from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import yahoofinancials


class StockDataset:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yahoofinancials.YahooFinancials(ticker)
        self.hist = None
        self.arima_dict = {}

    def get_hist(self, start_date, end_date, time_interval='daily'):
        df = pd.DataFrame(
            self.data.get_historical_price_data(start_date, end_date, time_interval)[self.ticker]['prices']
        )
        df = df[['formatted_date', 'high', 'low', 'open', 'adjclose', 'volume']]
        df.rename(columns={'formatted_date': 'date', 'adjclose': 'close'}, inplace=True)
        df.set_index('date', inplace=True)
        df['diff'] = df['close'].diff().fillna(0)
        df['return'] = df['close'].pct_change().fillna(0)
        self.hist = df
        return df.copy(deep=True)

    def get_change(self, is_up: bool):
        """
        Get all change period dataframe given up or down.
        Args:
            is_up: If True then choose all drawup periods, else then choose all drawdown periods.

        Returns: Dataframe with all either drawup or drawdown.

        """
        pmin_pmax = (self.hist['close'].diff(-1) > 0).astype(int).diff()
        change_point = pmin_pmax[pmin_pmax != 0].index

        change_df = self.hist.loc[change_point, 'close'].reset_index()
        change_df['start_date'] = change_df['date'].shift(1)
        change_df['return'] = change_df['close'].pct_change()
        change_df.rename(columns={'date': 'end_date'}, inplace=True)
        change_df = change_df[['start_date', 'end_date', 'close', 'return']].drop(index=0)
        change_df['duration'] = pd.to_datetime(change_df['end_date']) - pd.to_datetime(change_df['start_date'])

        if is_up:
            return change_df[change_df['return'] >= 0].reset_index(drop=True)
        else:
            return change_df[change_df['return'] < 0].reset_index(drop=True)

    def get_massive_change(self, is_up: bool, method, pct=0.05, past_period=120):
        """
        Filter out massive changes among all change periods by quantile.

        Args:
            is_up: If True then choose drawup, else then choose drawdown periods.
            method: The way to define a massive change. If the change return falls outside of give quantile, then it's massive.
                Options:
                    'all': Use all history return as distribution.
                    'past_all': Use all history return before corresponding date as distribution.
                    'past_period': Use past_period history return before corresponding date as distribution.
            pct: The quantile threshold.
            past_period: number of time steps to lookback for past_period method.

        Returns: Dataframe with either massive drawup or massive drawdown.

        """
        change_df = self.get_change(is_up)
        abs_return = change_df['return'].abs()
        if method == 'all':
            change_df = change_df[abs_return >= abs_return.quantile(1 - pct)]
        elif method == 'past_all':
            change_df = change_df[abs_return >= abs_return.expanding().quantile(1 - pct)]
        elif method == 'past_period':
            change_df = change_df[abs_return >= abs_return.rolling(window=past_period, min_periods=1).quantile(1 - pct)]
        return change_df.reset_index(drop=True)

    def get_change_forecast_label(self, forecast_len: int, **kwargs):
        """
        Given massive change start date, mark previous `forecast_days` trade days as 1.
        Thus a date label is 1, indicating there will be a massive change(drawup or drawdown) in `forecast_days`.

        Args:
            forecast_len: The number of trade days to forecast massive ahead.
            **kwargs: parameters of self.get_massive_change()

        Returns: Binary label series.

        """
        massive_change = self.get_massive_change(**kwargs)
        labels = pd.Series(0, index=self.hist.index)
        for date in massive_change['start_date']:
            iloc = labels.index.get_loc(date)
            if iloc < forecast_len:
                labels[:iloc] = 1
            else:
                labels[iloc - forecast_len:iloc] = 1
        return labels

    def lookback_agg(self, lookback_len, agg_func: Dict = None, new_col_name: List = None):
        """
        Apply aggregation functions on lookback period.

        Args:
            lookback_len: The number of trade days to lookback.
            agg_func: Aggregation functions, in structure {'column name': ['function to apply']}.
            new_col_name: List of column names of aggregated functions.

        Returns: Lookback aggregated dataframe.

        """
        if agg_func is None:
            agg_func = {
                'high': max,
                'low': min,
                'close': [np.mean, np.std],
                'diff': [np.mean, np.std],
                'return': [np.mean, np.std],
                'volume': [np.mean, np.std],
            }
            new_col_name = [
                f'past_{lookback_len}_max', f'past_{lookback_len}_min',
                f'past_{lookback_len}_avg', f'past_{lookback_len}_std',
                f'past_{lookback_len}_diff_avg', f'past_{lookback_len}_diff_std',
                f'past_{lookback_len}_return_avg', f'past_{lookback_len}_return_std',
                f'past_{lookback_len}_volume_avg', f'past_{lookback_len}_volume_std',
            ]
        agg_df = self.hist.rolling(lookback_len).agg(agg_func)
        agg_df.columns = new_col_name
        return agg_df
