import tempfile
import unittest
from pathlib import Path

import pandas as pd

from zhquant.plotting import plot_candlestick_with_orders


class PlottingTest(unittest.TestCase):
    def test_plot_candlestick_with_orders_writes_png(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        market_data = pd.DataFrame(
            {
                "open": [10, 11, 12, 11, 13],
                "high": [11, 12, 13, 12, 14],
                "low": [9, 10, 11, 10, 12],
                "close": [11, 12, 11, 13, 14],
                "volume": [1000, 1200, 900, 1500, 1800],
            },
            index=dates,
        )
        order_log = pd.DataFrame(
            [
                {"symbol": "AAA", "date": dates[1], "action": "BUY", "price": 12.0},
                {"symbol": "AAA", "date": dates[3], "action": "SELL", "price": 13.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chart.png"

            written = plot_candlestick_with_orders(market_data, order_log, "AAA", output)

            self.assertEqual(written, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
