"""Price/volume factors.

Convention: a factor value at row t uses data up to and including the close of
t, and "higher = expected to outperform". Execution happens later (see
backtest.py), so no factor here needs to be shifted by the caller.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _logret(panel) -> pd.DataFrame:
    return np.log1p(panel.ret)


def momentum_12_1(panel) -> pd.DataFrame:
    """Return from t-252 to t-21: classic 12-1 momentum.

    The most recent month is skipped because 1-month returns tend to reverse,
    which would contaminate the momentum signal.
    """
    lr = _logret(panel)
    return lr.rolling(231, min_periods=200).sum().shift(21)


def reversal_1m(panel) -> pd.DataFrame:
    """Negative of the last 21-day return (short-term reversal)."""
    return -_logret(panel).rolling(21, min_periods=15).sum()


def low_volatility(panel) -> pd.DataFrame:
    """Negative 60-day return volatility (low-vol anomaly)."""
    return -panel.ret.rolling(60, min_periods=40).std()


def illiquidity(panel) -> pd.DataFrame:
    """log Amihud illiquidity: mean(|r| / traded value) over 20 days.

    Higher = less liquid = (in theory) higher required return. Log because the
    raw ratio spans orders of magnitude across stocks.
    """
    amihud = (panel.ret.abs() / panel.value).rolling(20, min_periods=15).mean()
    return np.log(amihud.where(amihud > 0))


def max_return(panel) -> pd.DataFrame:
    """Negative of the largest daily return in the last 21 days.

    'Lottery' effect: stocks with extreme recent up-days attract retail demand
    and subsequently underperform. Plausibly relevant in a retail-heavy market
    like Taiwan, which is why it is included -- not because it is known to work.
    """
    return -panel.ret.rolling(21, min_periods=15).max()


FACTORS: dict[str, Callable] = {
    "momentum_12_1": momentum_12_1,
    "reversal_1m": reversal_1m,
    "low_volatility": low_volatility,
    "illiquidity": illiquidity,
    "max_return": max_return,
}


def compute_all(panel, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list(FACTORS)
    return {n: FACTORS[n](panel) for n in names}


def standardize(factor: pd.DataFrame, mask: pd.DataFrame, clip_pct: float = 0.01) -> pd.DataFrame:
    """Cross-sectional winsorize + z-score, using only stocks in `mask` each day.

    Done per row (per date), so it never mixes information across time.
    """
    f = factor.where(mask)
    lo = f.quantile(clip_pct, axis=1)
    hi = f.quantile(1 - clip_pct, axis=1)
    f = f.clip(lower=lo, upper=hi, axis=0)
    mu = f.mean(axis=1)
    sd = f.std(axis=1).replace(0, np.nan)
    return f.sub(mu, axis=0).div(sd, axis=0)


def neutralize(z: pd.DataFrame, industry: pd.Series, dates: pd.DatetimeIndex | None = None,
               min_group: int = 5) -> pd.DataFrame:
    """Remove each industry's mean from a cross-sectional score, then re-z-score.

    Afterwards the factor ranks stocks *within* their industry, so a "value"
    or "low-vol" factor cannot simply be a bet on, say, financials versus
    semiconductors. Per date only, so no look-ahead. Industries with fewer
    than `min_group` names on a date are pooled together.

    dates: only compute these rows (e.g. rebalance dates); others become NaN.
    """
    labels = industry.reindex(z.columns).fillna("(unknown)")
    rows = z if dates is None else z.loc[z.index.intersection(dates)]
    out = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    for d, row in rows.iterrows():
        ok = row.notna()
        if ok.sum() < 2 * min_group:
            continue
        lab = labels[ok]
        counts = lab.map(lab.value_counts())
        lab = lab.where(counts >= min_group, "(small groups)")
        x = row[ok]
        resid = x - x.groupby(lab).transform("mean")
        sd = resid.std()
        out.loc[d, ok] = (resid - resid.mean()) / sd if sd > 0 else np.nan
    return out
