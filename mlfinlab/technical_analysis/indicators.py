"""
Implementation of technical indicators introduced in paper,
Predicting direction of stock price index movement using artificial neural networks and support vector machines
by Yakup Kara, Melek Acar Boyacioglu, Ömer Kaan Baykan.
"""

from functools import partial

import pandas as pd
import numpy as np


def moving_average_ndays(price_series: pd.Series, ndays: int):
    """
    Get Simple Moving Average(sma) of price over n-days.
    This method computes the average of price over past n-days.

    If current price is above the sma of itself, we expect the trend is up.

    :param price_series: (pd.Series)  to get simple moving average for
    :param ndays: (int) the number of period to calculate simple moving average
    :return: (pd.Series) simple moving average of the price series.
    """
    mavg_ndays = price_series.rolling(window=ndays).mean()
    return mavg_ndays


def weighted_moving_average(price_series: pd.Series, ndays: int):
    """
    Get Weighted Moving Average(wma) of price over n-days.
    This method computes the linearly weighted average of price over past n-days.

    If current price is above the sma of itself, we expect the trend is up.

    :param price_series: (pd.Series)  to get weighted moving average of for
    :param ndays: (int) the number of period to calculate wma
    :return: (pd.Series) weighted moving average of the price series.
    """
    denominator = (ndays) * (ndays + 1) / 2
    weights = np.arange(1, ndays + 1)
    weights = weights / denominator

    weighted_avg_ndays = price_series.rolling(window=ndays).apply(lambda prices: np.dot(weights, prices), raw=False)
    return weighted_avg_ndays


def momentum_ndays(price_series: pd.Series, ndays: int):
    """
    Get Momentum of prices. The momentum indicates the degree of rise and fall in prices.
    Here, the momentum is difference between the current price and the price n-days ago.

    If momentum is positive, we expect the trend is up.

    :param price_series: (pd.Series)  to get momentum of for
    :param ndays: (int) the number of period to calculate the momentum
    :return: (pd.Series) momentum of the price series.
    """
    momentum = price_series - price_series.shift(ndays)
    return momentum


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, ndays: int):
    """
    Get Stochastic K%(stck) of prices. This is one of the stochastic oscillators which show the trend.
    It indicates where the price is located in the range of price fluctuation as a percentage(%).

    If stck is increasing, there is a possibility that prices will rise.

    :param high: (pd.Series) the highest price in time interval.
    :param low: (pd.Series) the lowest price in time interval.
    :param close: (pd.Series) the price at end of the time interval.
    :param ndays: (int) periods for computing moving average of 'high' and 'low'.
    :return: (pd.Series) stochastic k% of the price series.
    """
    ndays_high = high.rolling(window=ndays).max()
    ndays_low = low.rolling(window=ndays).min()

    stck = (close - ndays_low) * 100 / (ndays_high - ndays_low)

    return stck


def stochastic_d(high: pd.Series, low: pd.Series, close: pd.Series, ndays: pd.Series):
    """
    Get Stochastic D%(stcd) of prices. This is simple moving average of Stochastic k%(stck) over n-days.
    You can think it as smoothed version of stck.

    If stcd is increasing, there is a possibility that prices will rise.

    :param high: (pd.Series) the highest price in time interval.
    :param low: (pd.Series) the lowest price in time interval.
    :param close: (pd.Series) the price at end of the time interval.
    :param ndays: (int) periods for computing the stochastic k% and periods of moving average of it..
    :return: (pd.Series) stochastic d% of the price series.
    """
    stck = stochastic_k(high, low, close, ndays)
    stcd = stck.rolling(window=ndays).mean()

    return stcd


def ad_oscillator(high: pd.Series, low: pd.Series, close: pd.Series):
    """
    Get Accumulation/Distribution(A/D) oscillator. It is an indicator fo the trend of price, such as
    Stochastic K%, Williams R%. But, it doesn't use percentage.

    If this value is increasing, there is a possibility that prices will go up.

    :param high: (pd.Series) the highest price in time interval.
    :param low: (pd.Series) the lowest price in time interval.
    :param close: (pd.Series) the price at end of the time interval.
    :return: (pd.Series) a/d oscillator of the price series.
    """
    ad_oscil = (high - close.shift(1)) / (high - low)

    return ad_oscil


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series):
    """
    Get Williams R% of prices. This is one of the stochastic oscillators which show the trend.
    It indicates where the price is located in the range of price fluctuation as a percentage(%).

    If this value is increasing, there is a possibility that prices will rise.

    :param high: (pd.Series) the highest price in time interval.
    :param low: (pd.Series) the lowest price in time interval.
    :param close: (pd.Series) the price at end of the time interval.
    :return: (pd.Series) williams r% of the price series.
    """
    williams_r_ = (high - close)*100/(high-low)
    return williams_r_


def commodity_chanel_index(high: pd.Series, low: pd.Series, close: pd.Series, ndays: int):
    """
    Get Commodity Channel Index(CCI) of price series. This computes how far the mean is from the moving average.
    It can indicate both the direction of price and its strength.

    :param high: (pd.Series) the highest price in time interval.
    :param low: (pd.Series) the lowest price in time interval.
    :param close: (pd.Series) the price at end of the time interval.
    :param ndays: (int) periods for computing moving average of typical price and its variance.
    :return: (pd.Series) cci of the price series.
    """
    typical_price = (high + low + close) / 3
    avg_tp = typical_price.rolling(ndays).mean()
    deviation = typical_price.rolling(ndays).std()

    cci = (typical_price - avg_tp) / (0.015 * deviation)

    return cci


def _rs_filter(series: pd.Series, length: int):
    """
    This method is used for computing RSI in the 'rsi' method.

    :param series: (pd.Series) to get relative strength index for
    :param length: (int) window size of filter
    :return: (np.float64) rsi value
    """
    up_mean = series[series > 0].sum() / length
    down_mean = abs(series[series < 0].sum()) / length
    if up_mean == 0:
        return 0
    if down_mean == 0:
        return 100.

    rsi_value = 100 - (100/(1+(up_mean/down_mean)))

    return rsi_value


def rsi(price_series: pd.Series, ndays: int):
    """
    Get Relative Strength Index(RSI). This index calculates the frequency of price up and down and
    indicates the strength of the trend.

    :param price_series: (pd.Series) to get relative strength index for
    :param ndays: (int) periods used for rsi
    :return: (pd.Series) rsi of the price series.
    """
    diff = price_series.diff(1).dropna()
    rs_filter = partial(_rs_filter, length=ndays)
    rsi_series = diff.rolling(window=ndays).apply(rs_filter, raw=False)
    return rsi_series


def macd(price_series: pd.Series, short: int, long: int, ndays: int, adjust: bool = False):
    """
    Get Moving Average Convergence Divergence(MACD). It shows a trend through the differences between moving averages
    of different lengths.
    """
    ewm_short = price_series.ewm(span=short, adjust=adjust).mean()
    ewm_long = price_series.ewm(span=long, adjust=adjust).mean()
    ewm_diff = ewm_short - ewm_long
    macd_ = ewm_diff.ewm(span=ndays, adjust=adjust).mean()

    return macd_
