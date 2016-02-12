from unittest import TestCase

from testfixtures import TempDirectory
import pandas as pd
import numpy as np

from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import BcolzMinuteBarWriter, \
    US_EQUITIES_MINUTES_PER_DAY, BcolzMinuteBarReader
from zipline.finance.trading import TradingEnvironment
from zipline.protocol import BarData
from zipline.utils.test_utils import write_minute_data_for_asset


OHLC = ["open", "high", "low", "close"]
OHLCP = OHLC + ["price"]
ALL_FIELDS = OHLCP + ["volume", "last_traded"]

field_info = {
    "open": 1,
    "high": 2,
    "low": -1,
    "close": 0
}


class TestBarData(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        # asset1 has trades every minute
        # asset2 has trades every 10 minutes

        cls.env = TradingEnvironment()

        cls.days = cls.env.days_in_range(
            start=pd.Timestamp("2016-01-05", tz='UTC'),
            end=pd.Timestamp("2016-01-07", tz='UTC')
        )

        cls.env.write_data(equities_data={
            sid: {
                'start_date': cls.days[0],
                'end_date': cls.days[-1],
                'symbol': "ASSET{0}".format(sid)
            } for sid in [1, 2]
        })

        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=cls.build_minute_data()
        )

        cls.ASSET1 = cls.env.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.env.asset_finder.retrieve_asset(2)

        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def build_minute_data(cls):
        market_opens = cls.env.open_and_closes.market_open.loc[cls.days]

        writer = BcolzMinuteBarWriter(
            cls.days[0],
            cls.tempdir.path,
            market_opens,
            US_EQUITIES_MINUTES_PER_DAY
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.days[0],
            cls.days[-2],
            1
        )

        write_minute_data_for_asset(
            cls.env,
            writer,
            cls.days[0],
            cls.days[-2],
            2,
            10
        )

        return BcolzMinuteBarReader(cls.tempdir.path)

    def check_internal_consistency(self, bar_data):
        df = bar_data.spot_value([self.ASSET1, self.ASSET2], ALL_FIELDS)

        asset1_multi_field = bar_data.spot_value(self.ASSET1, ALL_FIELDS)
        asset2_multi_field = bar_data.spot_value(self.ASSET2, ALL_FIELDS)

        for field in ALL_FIELDS:
            asset1_value = bar_data.spot_value(self.ASSET1, field)
            asset2_value = bar_data.spot_value(self.ASSET2, field)

            multi_asset_series = bar_data.spot_value(
                [self.ASSET1, self.ASSET2], field
            )

            # make sure all the different query forms are internally
            # consistent
            self.assert_equal_or_both_nan(multi_asset_series[self.ASSET1],
                                          asset1_value)
            self.assert_equal_or_both_nan(multi_asset_series[self.ASSET2],
                                          asset2_value)

            self.assert_equal_or_both_nan(df.loc[self.ASSET1][field],
                                          asset1_value)
            self.assert_equal_or_both_nan(df.loc[self.ASSET2][field],
                                          asset2_value)

            self.assert_equal_or_both_nan(asset1_multi_field[field],
                                          asset1_value)
            self.assert_equal_or_both_nan(asset2_multi_field[field],
                                          asset2_value)

    def test_minute_value_before_assets_trading(self):
        # grab minutes that include the day before the asset start
        minutes = self.env.market_minutes_for_day(
            self.env.previous_trading_day(self.days[0])
        )

        # this entire day is before either asset has started trading
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.check_internal_consistency(bar_data)

            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))

            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.spot_value(asset, field)

                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == "volume":
                        self.assertEqual(0, asset_value)
                    elif field == "last_traded":
                        self.assertTrue(asset_value is pd.NaT)

    def test_regular_minute_value(self):
        minutes = self.env.market_minutes_for_day(self.days[0])

        for idx, minute in enumerate(minutes):
            # day2 has prices
            # (every minute for asset1, every 10 minutes for asset2)

            # asset1:
            # opens: 2-391
            # high: 3-392
            # low: 0-389
            # close: 1-390
            # volume: 100-3900 (by 100)

            # asset2 is the same thing, but with only every 10th minute
            # populated.

            # this test covers the "IPO morning" case, because asset2 only
            # has data starting on the 10th minute.

            bar_data = BarData(self.data_portal, lambda: minute, "minute")
            self.check_internal_consistency(bar_data)
            asset2_has_data = (((idx + 1) % 10) == 0)

            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET1))

            if idx < 9:
                self.assertFalse(bar_data.can_trade(self.ASSET2))
                self.assertFalse(bar_data.is_stale(self.ASSET2))
            else:
                self.assertTrue(bar_data.can_trade(self.ASSET2))

                if asset2_has_data:
                    self.assertFalse(bar_data.is_stale(self.ASSET2))
                else:
                    self.assertTrue(bar_data.is_stale(self.ASSET2))

            for field in ALL_FIELDS:
                asset1_value = bar_data.spot_value(self.ASSET1, field)
                asset2_value = bar_data.spot_value(self.ASSET2, field)

                # now check the actual values
                if idx == 0 and field == "low":
                    # first low value is 0, which is interpreted as NaN
                    self.assertTrue(np.isnan(asset1_value))
                else:
                    if field in OHLC:
                        self.assertEqual(
                            idx + 1 + field_info[field],
                            asset1_value
                        )

                        if asset2_has_data:
                            self.assertEqual(
                                idx + 1 + field_info[field],
                                asset2_value
                            )
                        else:
                            self.assertTrue(np.isnan(asset2_value))
                    elif field == "volume":
                        self.assertEqual((idx + 1) * 100, asset1_value)

                        if asset2_has_data:
                            self.assertEqual((idx + 1) * 100, asset2_value)
                        else:
                            self.assertEqual(0, asset2_value)
                    elif field == "price":
                        self.assertEqual(idx + 1, asset1_value)

                        if asset2_has_data:
                            self.assertEqual(idx + 1, asset2_value)
                        elif idx < 9:
                            # no price to forward fill from
                            self.assertTrue(np.isnan(asset2_value))
                        else:
                            # forward-filled price
                            self.assertEqual((idx // 10) * 10, asset2_value)
                    elif field == "last_traded":
                        self.assertEqual(minute, asset1_value)

                        if idx < 9:
                            self.assertTrue(asset2_value is pd.NaT)
                        elif asset2_has_data:
                            self.assertEqual(minute, asset2_value)
                        else:
                            last_traded_minute = minutes[(idx // 10) * 10]
                            self.assertEqual(last_traded_minute - 1,
                                             asset2_value)

    def test_minute_value_after_assets_stopped(self):
        minutes = self.env.market_minutes_for_day(self.days[-1])

        last_trading_minute = \
            self.env.market_minutes_for_day(self.days[-2])[-1]

        # this entire day is after both assets have stopped trading
        for idx, minute in enumerate(minutes):
            bar_data = BarData(self.data_portal, lambda: minute, "minute")

            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))

            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))

            self.check_internal_consistency(bar_data)

            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.spot_value(asset, field)

                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == "volume":
                        self.assertEqual(0, asset_value)
                    elif field == "last_traded":
                        self.assertEqual(last_trading_minute, asset_value)

    def assert_equal_or_both_nan(self, val1, val2):
        try:
            self.assertEqual(val1, val2)
        except AssertionError:
            if val1 is pd.NaT:
                self.assertTrue(val2 is pd.NaT)
            elif np.isnan(val1):
                self.assertTrue(np.isnan(val2))
            else:
                raise
