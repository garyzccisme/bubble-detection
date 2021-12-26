import quandl
import pandas as pd
import numpy as np


class Macro:
    def __init__(self, token):
        self.token = token
        self.macro = None

    def quarter_lag(self, df: pd.DataFrame, freq: str, steps: int = 4):
        if freq == 'Q':
            multiplier = 1
        elif freq == 'M':
            multiplier = 3
        elif freq == 'D':
            multiplier = 90
        else:
            raise ValueError("Invalid freq input, only ['Q', 'M', 'D'] are allowed.")

        lags = [df]
        for shift in np.arange(steps) + 1:
            columns = {col: f'{col}_lag_Q{int(shift)}' for col in df.columns}
            lags.append(df.shift(shift * multiplier).rename(columns=columns))
        return pd.concat(lags, axis=1)

    def get_macro(self):
        quandl.ApiConfig.api_key = self.token
        # https://data.nasdaq.com/data/FRED/GDP-gross-domestic-product
        gdp = quandl.get("FRED/GDP").rename(columns={'Value': 'GDP'})
        gdp = self.quarter_lag(gdp, freq='Q')
        # https://data.nasdaq.com/data/FRED/CPIAUCSL-consumer-price-index-for-all-urban-consumers-all-items
        cpi = quandl.get("FRED/CPIAUCSL").rename(columns={'Value': 'CPI'})
        cpi = self.quarter_lag(cpi, freq='M')
        # https://data.nasdaq.com/data/FRED/DFF-effective-federal-funds-rate
        ffr = quandl.get("FRED/DFF").rename(columns={'Value': 'FFR'})
        ffr = self.quarter_lag(ffr, freq='D')
        # https://data.nasdaq.com/data/FRED/DGS5-5year-treasury-constant-maturity-rate
        treasury_5 = quandl.get("FRED/DGS5").rename(columns={'Value': 'treasury_5'})
        treasury_5 = self.quarter_lag(treasury_5, freq='D')
        # https://data.nasdaq.com/data/FRED/DGS10-10year-treasury-constant-maturity-rate
        treasury_10 = quandl.get("FRED/DGS10").rename(columns={'Value': 'treasury_10'})
        treasury_10 = self.quarter_lag(treasury_10, freq='D')
        # https://data.nasdaq.com/data/FRED/DGS30-30year-treasury-constant-maturity-rate
        treasury_30 = quandl.get("FRED/DGS30").rename(columns={'Value': 'treasury_30'})
        treasury_30 = self.quarter_lag(treasury_30, freq='D')
        # https://data.nasdaq.com/data/FRED/UNRATE-civilian-unemployment-rate
        unemployment = quandl.get("FRED/UNRATE").rename(columns={'Value': 'unemployment_rate'})
        unemployment = self.quarter_lag(unemployment, freq='M')
        # https://data.nasdaq.com/data/FRED/INDPRO-industrial-production-index
        ipi = quandl.get("FRED/INDPRO").rename(columns={'Value': 'IPI'})
        ipi = self.quarter_lag(ipi, freq='M')
        # https://data.nasdaq.com/data/FRED/DCOILWTICO-crude-oil-prices-west-texas-intermediate-wti-cushing-oklahoma
        oil = quandl.get("FRED/DCOILWTICO").rename(columns={'Value': 'oil'})
        oil = self.quarter_lag(oil, freq='D')
        # https://data.nasdaq.com/data/FRED/DTWEXM-trade-weighted-us-dollar-index-major-currencies
        # currency = quandl.get("FRED/DTWEXM").rename(columns={'Value': 'currency'})
        # https://data.nasdaq.com/data/LBMA/GOLD-gold-price-london-fixing
        gold = quandl.get("LBMA/GOLD")[['USD (AM)']].rename(columns={'USD (AM)': 'gold'})
        gold = self.quarter_lag(gold, freq='D')

        # Merge variables
        var = [gdp, cpi, ffr, treasury_5, treasury_10, treasury_30, unemployment, ipi, oil, gold]
        macro = pd.concat(var, axis=1).interpolate(method='time')
        macro.index = macro.index.astype(str)
        macro.index.name = 'date'
        return macro
