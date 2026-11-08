"""Walk-forward factor combination with purged training windows.

At each rebalance date d we fit a model on past (features, label) pairs and
score stocks at d. The pitfall this module exists to demonstrate:

    The label for a sample at date s is the forward return over
    s+lag+1 ... s+lag+horizon. If that window ends after d, the label was not
    known at d. Training on it leaks the future into the model.

So training samples are *purged*: only dates s whose label window has fully
ended on or before d are used. Without purging, an overlapping 21-day label
leaks ~3 weeks of future returns into each fit and in-sample IC looks great.

Model: LightGBM regressor if installed (pip install lightgbm), otherwise a
ridge regression in numpy. Labels are cross-sectional ranks scaled to [-0.5,
0.5] so the model learns *relative* performance and market-wide moves (which
the strategy cannot exploit) do not dominate the loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import forward_returns
from .factors import neutralize, standardize


def _stack(features: dict[str, pd.DataFrame], dates) -> pd.DataFrame:
    parts = {k: v.loc[dates].stack(future_stack=True) for k, v in features.items()}
    return pd.DataFrame(parts)


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, Xp: np.ndarray, lam: float = 10.0) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X)), X])
    I = np.eye(Xa.shape[1])
    I[0, 0] = 0
    beta = np.linalg.solve(Xa.T @ Xa + lam * I, Xa.T @ y)
    return np.column_stack([np.ones(len(Xp)), Xp]) @ beta


def walk_forward_signal(panel, raw_factors: dict[str, pd.DataFrame], mask: pd.DataFrame,
                        rebal_dates: pd.DatetimeIndex, horizon: int = 21, lag: int = 1,
                        train_periods: int = 36, min_train_periods: int = 24,
                        model: str = "auto", purge: bool = True, seed: int = 0,
                        industry: pd.Series | None = None) -> pd.DataFrame:
    """Return a signal DataFrame (rebal_dates x stocks) of out-of-sample scores.

    purge=False exists ONLY so the tests can show what leakage does. Never use
    it for research.
    """
    feats = {k: standardize(v, mask) for k, v in raw_factors.items()}
    if industry is not None:
        feats = {k: neutralize(v, industry, rebal_dates) for k, v in feats.items()}
    fwd = forward_returns(panel.ret, horizon, lag).where(mask)
    label = fwd.rank(axis=1, pct=True) - 0.5

    idx = panel.ret.index
    pos_of = {d: i for i, d in enumerate(idx)}
    use_lgb = False
    if model in ("auto", "lightgbm"):
        try:
            import lightgbm as lgb  # noqa: F401
            use_lgb = True
        except ImportError:
            if model == "lightgbm":
                raise

    out = pd.DataFrame(np.nan, index=rebal_dates, columns=panel.ret.columns)
    rebal_list = list(rebal_dates)
    for j, d in enumerate(rebal_list):
        p = pos_of[d]
        past = rebal_list[max(0, j - train_periods):j]
        if purge:
            past = [s for s in past if pos_of[s] + lag + horizon <= p]
        if len(past) < min_train_periods:
            continue
        train = _stack(feats, past).join(label.loc[past].stack(future_stack=True).rename("y")).dropna()
        test = _stack(feats, [d]).dropna()
        if len(train) < 200 or test.empty:
            continue
        X, y = train.drop(columns="y").to_numpy(), train["y"].to_numpy()
        Xp = test.to_numpy()
        if use_lgb:
            # Native API: the sklearn wrapper (LGBMRegressor) needs scikit-learn installed.
            import lightgbm as lgb
            params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15,
                      "min_data_in_leaf": 200, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "feature_fraction": 0.8, "seed": seed, "deterministic": True,
                      "num_threads": 1, "verbose": -1}
            booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=200)
            pred = booster.predict(Xp)
        else:
            pred = _ridge_fit_predict(X, y, Xp)
        stocks = test.index.get_level_values(1)
        out.loc[d, stocks] = pred
    return out
