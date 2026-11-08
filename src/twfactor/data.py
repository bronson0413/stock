"""Data layer.

Everything downstream works on a `Panel`: wide DataFrames indexed by trading
date (rows) and stock id (columns). Two sources are provided:

* `make_synthetic` -- simulated market with a *known* predictive signal planted
  in it. Used by the tests to prove the engine can find real alpha and does
  not hallucinate alpha from noise.
* `load_finmind`  -- real TWSE/TPEx daily prices from the FinMind API, cached
  to disk per stock so an interrupted download can resume.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# TWSE/TPEx daily price limit is +/-10% of the reference price.
PRICE_LIMIT = 0.10


@dataclass
class Panel:
    """Wide (date x stock) market data.

    ret     daily return used for P&L. For real data, corporate-action days are
            priced off the exchange reference price (see `build_returns`).
    close   raw close price (used for the minimum-price filter).
    value   traded value in NTD (used for the liquidity filter and Amihud).
    true_signal  synthetic data only: the planted predictive signal.
    industry     optional stock_id -> industry label, for industry neutralisation.
    """

    ret: pd.DataFrame
    close: pd.DataFrame
    value: pd.DataFrame
    true_signal: pd.DataFrame | None = None
    industry: pd.Series | None = None

    def __post_init__(self) -> None:
        for name in ("close", "value"):
            other = getattr(self, name)
            if not other.index.equals(self.ret.index) or not other.columns.equals(self.ret.columns):
                raise ValueError(f"Panel.{name} is not aligned with Panel.ret")

    def truncate(self, end) -> "Panel":
        """Return a copy containing data up to and including `end`."""
        sig = None if self.true_signal is None else self.true_signal.loc[:end]
        return Panel(self.ret.loc[:end], self.close.loc[:end], self.value.loc[:end], sig, self.industry)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic(
    n_stocks: int = 300,
    n_days: int = 1500,
    alpha_per_sd: float = 0.0004,
    signal_persistence: float = 0.995,
    seed: int = 0,
    start: str = "2018-01-02",
) -> Panel:
    """Simulate a market where `true_signal` predicts next-day returns.

    r[t, i] = beta_i * market[t] + alpha_per_sd * z[t-1, i] + idio[t, i]

    z is a slowly mean-reverting (AR(1)) cross-sectional signal, standardised
    to unit variance. With the defaults the monthly rank IC of z is roughly
    0.05-0.10, which is in the range of a decent real-world factor -- i.e. a
    weak signal buried in noise, not an obvious one.

    Realism details the engine has to cope with:
    * returns are clipped to the +/-10% TW price limit;
    * ~10% of stocks list late and ~10% delist early (NaN outside their life);
    * stocks have very different liquidity.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    ids = [f"S{i:04d}" for i in range(n_stocks)]

    beta = rng.uniform(0.6, 1.4, n_stocks)
    idio_vol = rng.uniform(0.012, 0.03, n_stocks)
    market = rng.normal(0.0003, 0.01, n_days)

    rho = signal_persistence
    z = np.empty((n_days, n_stocks))
    z[0] = rng.standard_normal(n_stocks)
    shocks = rng.standard_normal((n_days, n_stocks)) * np.sqrt(1 - rho**2)
    for t in range(1, n_days):
        z[t] = rho * z[t - 1] + shocks[t]

    expected = np.zeros((n_days, n_stocks))
    expected[1:] = alpha_per_sd * z[:-1]  # signal known at t-1 predicts return at t
    ret = beta * market[:, None] + expected + rng.standard_normal((n_days, n_stocks)) * idio_vol
    ret = np.clip(ret, -PRICE_LIMIT, PRICE_LIMIT)

    # Listing / delisting.
    alive = np.ones((n_days, n_stocks), dtype=bool)
    late = rng.choice(n_stocks, n_stocks // 10, replace=False)
    for i in late:
        alive[: rng.integers(50, n_days // 2), i] = False
    early = rng.choice(np.setdiff1d(np.arange(n_stocks), late), n_stocks // 10, replace=False)
    for i in early:
        alive[rng.integers(n_days // 2, n_days - 20):, i] = False

    ret = np.where(alive, ret, np.nan)
    start_px = rng.uniform(15, 300, n_stocks)
    close = start_px * np.exp(np.nancumsum(np.log1p(np.nan_to_num(ret)), axis=0))
    close = np.where(alive, close, np.nan)
    shares_traded = np.exp(rng.normal(np.log(rng.uniform(2e4, 3e6, n_stocks)), 0.5, (n_days, n_stocks)))
    value = np.where(alive, shares_traded * close, np.nan)

    mk = lambda a: pd.DataFrame(a, index=dates, columns=ids)
    # Separate RNG so adding labels does not change any of the draws above.
    labels = np.random.default_rng(seed + 10_000).integers(0, 10, n_stocks)
    industry = pd.Series([f"IND{k}" for k in labels], index=ids)
    return Panel(mk(ret), mk(close), mk(value), mk(np.where(alive, z, np.nan)), industry)




# ---------------------------------------------------------------------------
# Real data: FinMind
# ---------------------------------------------------------------------------
#
# Why returns need corporate-action data (verified on real rows, see
# tests/fixtures/README.md):
#
#   On ex-dividend, capital-reduction and par-value-change days TWSE does not
#   compare the close with the previous close; it shows the change as "X0.00"
#   (不比價). FinMind keeps the 0 and drops the X. So on exactly the days that
#   matter, neither `close.pct_change()` nor `close - spread` gives the
#   holder's return. The exchange's *reference price* for that day does:
#
#       ret = close / reference * before / prev_close - 1
#
#   (before == prev_close in practice; the ratio guards against odd cases.)

EVENT_COLUMNS = ["date", "stock_id", "before", "reference", "kind"]


def events_from_finmind(dividend: pd.DataFrame | None = None,
                        capital_reduction: pd.DataFrame | None = None,
                        par_value: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalise FinMind corporate-action tables to (date, stock_id, before, reference, kind)."""
    parts = []
    spec = [
        (dividend, "before_price", "reference_price", "dividend"),
        (capital_reduction, "ClosingPriceonTheLastTradingDay", "PostReductionReferencePrice", "capital_reduction"),
        (par_value, "before_close", "after_ref_close", "par_value"),
    ]
    for df, b, r, kind in spec:
        if df is None or df.empty:
            continue
        x = df[["date", "stock_id", b, r]].rename(columns={b: "before", r: "reference"})
        x["kind"] = kind
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev = pd.concat(parts, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["stock_id"] = ev["stock_id"].astype(str)
    ev = ev[(ev["reference"] > 0) & (ev["before"] > 0)]
    # Two different events on one day: which reference price already includes
    # the other is not knowable from these tables, so drop both and let the day
    # surface as "unresolved" instead of guessing.
    dup = ev.duplicated(["date", "stock_id"], keep=False)
    return ev[~dup].reset_index(drop=True)[EVENT_COLUMNS]


def build_returns(prices: pd.DataFrame, events: pd.DataFrame | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Long FinMind price rows -> (ret, close, value, audit) wide frames.

    prices needs: date, stock_id, close, spread, Trading_Volume, Trading_money.

    audit has one row per day that needed attention:
      kind = dividend | capital_reduction | par_value   corrected with a reference price
      kind = unresolved   spread == 0 but close != prev close and no reference
                          price was found -> return set to NaN (not guessed)
    """
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["stock_id"] = px["stock_id"].astype(str)
    # Zero-volume rows are suspended / no-trade days: the close is not a traded price.
    px = px[px["Trading_Volume"] > 0].sort_values(["stock_id", "date"])
    # A duplicated (stock, date) row would make shift(1) pair a day with itself.
    px = px.drop_duplicates(["stock_id", "date"], keep="last")
    px["prev_close"] = px.groupby("stock_id")["close"].shift(1)
    px["ret"] = px["close"] / px["prev_close"] - 1

    ev = events if events is not None else pd.DataFrame(columns=EVENT_COLUMNS)
    ev = ev.astype({"stock_id": str, "before": float, "reference": float, "kind": object})
    ev["date"] = pd.to_datetime(ev["date"])
    px = px.merge(ev, on=["date", "stock_id"], how="left")
    has_ev = px["reference"].notna() & px["prev_close"].notna()
    adj = px["close"] / px["reference"] * px["before"] / px["prev_close"] - 1
    px["ret"] = np.where(has_ev, adj, px["ret"])

    # X-day signature without a matching event: refuse to guess.
    moved = (px["close"] - px["prev_close"]).abs() > 1e-9
    unresolved = (px["spread"] == 0) & moved & px["prev_close"].notna() & ~has_ev
    px.loc[unresolved, "ret"] = np.nan
    px.loc[unresolved, "kind"] = "unresolved"

    audit = px.loc[has_ev | unresolved, ["date", "stock_id", "kind", "prev_close", "before",
                                         "reference", "close", "ret"]].copy()
    audit["naive_pct_change"] = (px.loc[audit.index, "close"] / px.loc[audit.index, "prev_close"] - 1)
    audit = audit.sort_values(["date", "stock_id"]).reset_index(drop=True)

    wide = lambda col: px.pivot_table(index="date", columns="stock_id", values=col,
                                      aggfunc="last", dropna=False).sort_index()
    close = wide("close")
    ret = wide("ret").reindex_like(close)
    value = wide("Trading_money").reindex_like(close)
    ret.columns.name = close.columns.name = value.columns.name = None
    return ret, close, value, audit


class FinMindError(RuntimeError):
    pass


class FinMindClient:
    """Minimal FinMind v4 client with a per-request disk cache.

    The free tier has an hourly request quota. Every successful response is
    cached as CSV, so when the quota is hit you can simply re-run later and the
    download resumes. Register for a free token to raise the quota and export
    it as FINMIND_TOKEN.
    """

    # FinMind policy (finmind.github.io, checked 2026-11): free 300 req/h,
    # registered 600 req/h; over quota -> HTTP 402; many 4xx in a short time
    # -> temporary IP ban (403). So: wait on 402, back off on 5xx, and stop
    # immediately on any other error instead of retrying it.
    QUOTA_WAIT_S = 15 * 60
    MAX_QUOTA_WAITS = 12
    MAX_5XX_RETRIES = 3

    def __init__(self, cache_dir: str | Path = "data/cache", token: str | None = None,
                 sleep: float = 0.3, verbose: bool = True, wait_on_quota: bool = True):
        self.cache = Path(cache_dir)
        self.token = token or os.environ.get("FINMIND_TOKEN")
        self.sleep = sleep
        self.verbose = verbose
        self.wait_on_quota = wait_on_quota
        self.requests_made = 0

    def _request(self, dataset: str, params: dict) -> tuple[int, dict]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        resp = requests.get(FINMIND_URL, params={"dataset": dataset, **params},
                            headers=headers, timeout=60)
        self.requests_made += 1
        try:
            payload = resp.json()
        except ValueError:
            payload = {"status": resp.status_code, "msg": "non-JSON response"}
        return resp.status_code, payload

    def get(self, dataset: str, **params) -> pd.DataFrame:
        key = "_".join(f"{k}={v}" for k, v in sorted(params.items())) or "all"
        path = self.cache / dataset / f"{key}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"stock_id": str})

        quota_waits = server_retries = 0
        while True:
            http, payload = self._request(dataset, params)
            status = payload.get("status", http)
            if status == 200:
                break
            if status == 402 and self.wait_on_quota and quota_waits < self.MAX_QUOTA_WAITS:
                quota_waits += 1
                if self.verbose:
                    print(f"  quota reached after {self.requests_made} requests; "
                          f"waiting {self.QUOTA_WAIT_S // 60} min ({quota_waits}/{self.MAX_QUOTA_WAITS})")
                time.sleep(self.QUOTA_WAIT_S)
                continue
            if http >= 500 and server_retries < self.MAX_5XX_RETRIES:
                server_retries += 1
                time.sleep(2 ** server_retries)
                continue
            raise FinMindError(
                f"{dataset} {params}: HTTP {http}, {payload.get('msg')!r}. "
                "Not retried (repeated 4xx can get the IP banned). Cached data is kept; re-run to resume.")
        df = pd.DataFrame(payload.get("data", []))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        time.sleep(self.sleep)
        return df


# Industry categories that are not ordinary common stock.
_EXCLUDED_CATEGORIES = {"ETF", "存託憑證", "創新板股票", "創新版股票", "受益證券", "Index"}


def stock_universe(client: FinMindClient, start: str) -> pd.DataFrame:
    """All common stocks that were listed on TWSE/TPEx at any point since `start`.

    = currently listed (TaiwanStockInfo) + delisted after `start`
      (TaiwanStockDelisting). Including the delisted ones is what removes
    survivorship bias: companies that went bust are in the sample.

    Returns columns stock_id, stock_name, industry_category, status
    (listed/delisted), delist_date. industry_category is today's
    classification (FinMind has no history of it), and unknown for delisted
    stocks.
    """
    info = client.get("TaiwanStockInfo")
    info["stock_id"] = info["stock_id"].astype(str)
    info = info[info["type"].isin(["twse", "tpex"]) & info["stock_id"].str.fullmatch(r"[1-9]\d{3}")]
    bad = info.groupby("stock_id")["industry_category"].agg(lambda s: bool(set(s) & _EXCLUDED_CATEGORIES))
    info = info[~info["stock_id"].map(bad)]
    listed = info.drop_duplicates("stock_id")[["stock_id", "stock_name", "industry_category"]]
    listed = listed.assign(status="listed", delist_date=pd.NaT)

    dl = client.get("TaiwanStockDelisting")
    dl["stock_id"] = dl["stock_id"].astype(str)
    dl["date"] = pd.to_datetime(dl["date"])
    dl = dl[dl["stock_id"].str.fullmatch(r"[1-9]\d{3}") & (dl["date"] >= pd.Timestamp(start))
            & ~dl["stock_name"].str.contains("-DR", na=False) & ~dl["stock_id"].str.startswith("91")
            & ~dl["stock_id"].isin(listed["stock_id"])]  # code later reused by a new listing
    delisted = dl.sort_values("date").drop_duplicates("stock_id", keep="last")
    delisted = delisted.rename(columns={"date": "delist_date"})[["stock_id", "stock_name", "delist_date"]]
    delisted["status"] = "delisted"
    delisted["industry_category"] = "(delisted, unknown)"
    return pd.concat([listed, delisted], ignore_index=True).sort_values("stock_id").reset_index(drop=True)


def load_finmind(stock_ids: list[str], start: str, end: str,
                 client: FinMindClient | None = None,
                 with_corporate_actions: bool = True,
                 industry: pd.Series | None = None) -> tuple[Panel, pd.DataFrame]:
    """Download prices (+ corporate actions) and build a Panel.

    Requests: 1 price + 1 dividend per stock, 1 bulk par-value call, and a
    capital-reduction call only for stocks that still have unexplained X-days
    after dividends are applied. Returns (panel, audit).
    """
    client = client or FinMindClient()
    prices, divs = [], []
    for n, sid in enumerate(stock_ids, 1):
        p = client.get("TaiwanStockPrice", data_id=sid, start_date=start, end_date=end)
        if not p.empty:
            prices.append(p)
            if with_corporate_actions:
                d = client.get("TaiwanStockDividendResult", data_id=sid, start_date=start, end_date=end)
                if not d.empty:
                    divs.append(d)
        if client.verbose and n % 50 == 0:
            print(f"  {n}/{len(stock_ids)} stocks ({client.requests_made} new requests)")
    if not prices:
        raise FinMindError("No price data returned")
    prices = pd.concat(prices, ignore_index=True)

    if not with_corporate_actions:
        ret, close, value, audit = build_returns(prices)
        return Panel(ret, close, value, industry=industry), audit

    par = client.get("TaiwanStockParValueChange", start_date=start)
    div = pd.concat(divs, ignore_index=True) if divs else None
    ev = events_from_finmind(div, None, par)
    _, _, _, audit = build_returns(prices, ev)
    need = sorted(audit.loc[audit["kind"] == "unresolved", "stock_id"].unique())
    if need:
        if client.verbose:
            print(f"  fetching capital reductions for {len(need)} stocks with unexplained gaps")
        caps = [client.get("TaiwanStockCapitalReductionReferencePrice", data_id=s, start_date=start)
                for s in need]
        caps = [c for c in caps if not c.empty]
        cap = pd.concat(caps, ignore_index=True) if caps else None
        ev = events_from_finmind(div, cap, par)
    ret, close, value, audit = build_returns(prices, ev)
    return Panel(ret, close, value, industry=industry), audit
