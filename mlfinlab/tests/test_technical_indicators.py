"""
Test techinical indicators
"""

import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from mlfinlab.technical_analysis import indicators


class TestTechnicalIndicators(unittest.TestCase):
    """
    Test multiple indicators in indicators module.
    """
    def setUp(self):
        num_rows = 10

        # test index
        test_index = pd.util.testing.makeDateIndex(num_rows, freq="D")
        # test series
        self.test_series = pd.Series(np.arange(1, 11), index=test_index)

        # test dataframe
        self.test_df = pd.DataFrame(
            {'open': [5, 14, 39, 34, 30, 38, 8, 2, 2, 23],
             'high': [19, 31, 9, 1, 38, 13, 28, 7, 36, 36],
             'low': [32, 33, 36, 3, 30, 15, 6, 11, 5, 36],
             'close': [28, 13, 37, 13, 7, 22, 8, 31, 17, 12]
             },
            index=test_index
        )

        self.addTypeEqualityFunc(pd.Series, self.assert_series_equal)

    def assert_series_equal(self, a, b, msg):
        """
        This method is used for integrating testing method of pandas in python unittest.
        """
        try:
            pdt.assert_series_equal(a, b, check_less_precise=4)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_moving_average_ndays(self):
        """
        Assert that moving_average_ndays return wrong values.
        """
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8], index=self.test_series.index)

        ndays = 5
        mavg = indicators.moving_average_ndays(self.test_series, ndays=ndays)

        pdt.assert_series_equal(mavg, answer)

    def test_weighted_moving_average(self):
        """
        Assert that weighted_moving_average return wrong values.
        """
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan, 3.666667, 4.666667, 5.666667, 6.666667, 7.666667, 8.666667],
                           index=self.test_series.index
                           )
        ndays = 5
        wmavg = indicators.weighted_moving_average(self.test_series, ndays=ndays)

        self.assertEqual(wmavg, answer)

    def test_momentum_ndays(self):
        """
        Assert that momentum_ndays return wrong values.
        """
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5., 5., 5., 5., 5.],
                           index=self.test_series.index
                           )
        ndays = 5
        momentum = indicators.momentum_ndays(self.test_series, ndays=ndays)

        self.assertEqual(momentum, answer)

    def test_stochastic_k(self):
        """
        Assert that stochastic_k return wrong values.
        """
        answer = pd.Series([
            np.nan, np.nan, np.nan, np.nan,
            11.428571, 54.285714, 14.285714, 80.000000, 36.363636, 22.580645
        ],
            index=self.test_series.index
        )
        ndays = 5
        high, low, close = self.test_df.high, self.test_df.low, self.test_df.close
        stochastic_k = indicators.stochastic_k(high, low, close, ndays=ndays)

        self.assertEqual(stochastic_k, answer)

    def test_stochastic_d(self):
        """
        Assert that stochastic_d return wrong values.
        """
        answer = pd.Series([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            39.272727, 41.503142
        ],
            index=self.test_series.index
        )
        ndays = 5
        high, low, close = self.test_df.high, self.test_df.low, self.test_df.close
        stochastic_d = indicators.stochastic_d(high, low, close, ndays=ndays)

        self.assertEqual(stochastic_d, answer)

    def test_ad_oscillator(self):
        """
        Assert that ad_oscillator return wrong values.
        """
        answer = pd.Series([np.nan, -1.5, 0.148148, 18., 3.125, -3., 0.272727, 0.250000, 0.161290, np.inf],
                           index=self.test_series.index
                           )
        high, low, close = self.test_df.high, self.test_df.low, self.test_df.close
        ad_oscillator = indicators.ad_oscillator(high, low, close)

        self.assertEqual(ad_oscillator, answer)

    def test_williams_r(self):
        """
        Assert that williams_r return wrong values.
        """
        answer = pd.Series([69.230769, -900., 103.703704, 600., 387.5, 450., 90.909091, 600., 61.290323, np.inf],
                           index=self.test_series.index
                           )
        high, low, close = self.test_df.high, self.test_df.low, self.test_df.close
        williams_r = indicators.williams_r(high, low, close)

        self.assertEqual(williams_r, answer)

    def test_commodity_chanel_index(self):
        """
        Assert that commodity_chanel_index return wrong values.
        """
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan,
                            21.807371, -25.053221, -28.472608, 7.713195, 16.881315, 111.832077
                            ],
                           index=self.test_series.index
                           )
        ndays = 5
        high, low, close = self.test_df.high, self.test_df.low, self.test_df.close
        cci = indicators.commodity_chanel_index(high, low, close, ndays=ndays)

        self.assertEqual(cci, answer)

    def test_rsi_filter(self):
        """
        Assert that rsi_filter return wrong value.
        """
        answer = 100.0
        ndays = 10

        rsi_value = indicators._rs_filter(self.test_series, length=ndays)

        self.assertEqual(rsi_value, answer)

    def test_rsi(self):
        """
        Assert that rsi return wrong values.
        """
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0],
                           index=self.test_series.index[1:]
                           )
        ndays = 5
        rsi = indicators.rsi(self.test_series, ndays=ndays)

        self.assertEqual(rsi, answer)

    def test_macd(self):
        """
        Assert that macd return wrong values.
        """
        answer = pd.Series([0., 0.083333, 0.222222, 0.377315, 0.522377,
                            0.645126, 0.742584, 0.816671, 0.871270, 0.910599],
                           index=self.test_series.index
                           )

        answer_adjust = pd.Series([0., 0.044444, 0.113570, 0.203427, 0.306430,
                                   0.413790, 0.517684, 0.612488, 0.695030, 0.764232],
                                  index=self.test_series.index
                                  )

        short, long, ndays = 3, 5, 3

        macd = indicators.macd(self.test_series, short, long, ndays, adjust=False)
        self.assertEqual(macd, answer)

        macd_adjust = indicators.macd(self.test_series, short, long, ndays, adjust=True)
        self.assertEqual(macd_adjust, answer_adjust)
