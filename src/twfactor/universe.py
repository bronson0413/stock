"""Point-in-time investable universe.

Every filter at date t uses only data available at the close of t. This keeps
the universe free of look-ahead (e.g. we never exclude a stock *because* it
later delisted).
"""

from __future__ import annotations

import pandas as pd


def investable(
    panel,
    min_history: int = 252,
    min_avg_value: float = 2e7,
    value_window: int = 20,
    min_price: float = 10.0,
) -> pd.DataFrame:
    """Boolean (date x stock) mask of stocks eligible to be ranked/held.

    min_history    trading days with a return observed (avoids IPO noise and
                   lets 12-month factors exist).
    min_avg_value  20-day average traded value in NTD. 2e7 = NT$20M/day; below
                   that, a strategy of any size moves the price.
    min_price      excludes very low-priced stocks (often distressed / 全額交割).
    """
    has_hist = panel.ret.notna().cumsum() >= min_history
    liquid = panel.value.rolling(value_window, min_periods=value_window).mean() >= min_avg_value
    priced = panel.close >= min_price
    traded_today = panel.ret.notna()
    return has_hist & liquid & priced & traded_today
