"""Offline test of the FinMind download path, with the HTTP layer faked.

Checks the logic that cannot be exercised without the network otherwise:
caching / resume, which requests are made, universe filtering, and that the
real-data fixtures come out with the right returns end to end.
"""

from pathlib import Path

import pandas as pd
import pytest

import twfactor.data as D

FIX = Path(__file__).parent / "fixtures"


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


def _fake_api(calls):
    prices = pd.read_csv(FIX / "prices.csv", dtype={"stock_id": str})
    div = pd.read_csv(FIX / "dividend_result.csv", dtype={"stock_id": str})
    cap = pd.read_csv(FIX / "capital_reduction.csv", dtype={"stock_id": str})
    par = pd.read_csv(FIX / "par_value_change.csv", dtype={"stock_id": str})
    info = pd.DataFrame({
        "stock_id": ["2327", "2412", "2412", "0050", "9105", "6919"],
        "stock_name": ["國巨", "中華電", "中華電", "元大台灣50", "泰金寶-DR", "康霈*"],
        "type": ["twse", "twse", "twse", "twse", "twse", "tpex"],
        "industry_category": ["電子零組件業", "通信網路業", "電子工業", "ETF", "存託憑證", "創新板股票"],
        "date": "2024-12-04",
    })
    delist = pd.DataFrame({
        "date": ["2023-04-01", "2005-10-04", "2022-03-18", "2019-01-02"],
        "stock_id": ["6594", "1204", "9188", "2412"],
        "stock_name": ["展匯科", "津津", "精熙-DR", "舊代號"],
    })

    def get(url, params, headers, timeout):
        calls.append(dict(params))
        ds, sid = params["dataset"], params.get("data_id")
        table = {"TaiwanStockPrice": prices, "TaiwanStockDividendResult": div,
                 "TaiwanStockCapitalReductionReferencePrice": cap,
                 "TaiwanStockParValueChange": par, "TaiwanStockInfo": info,
                 "TaiwanStockDelisting": delist}[ds]
        if sid is not None:
            table = table[table["stock_id"] == sid]
        return FakeResp({"status": 200, "msg": "success", "data": table.to_dict("records")})

    return get


@pytest.fixture
def client(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(D.requests, "get", _fake_api(calls))
    c = D.FinMindClient(tmp_path / "cache", token=None, sleep=0, verbose=False)
    c.calls = calls
    return c


def test_end_to_end_returns_and_request_plan(client):
    panel, audit = D.load_finmind(["2412", "2327"], "2017-01-01", "2025-12-31", client)
    r = panel.ret
    assert r.loc["2024-07-04", "2412"] == pytest.approx(118.5 / 120.74 - 1)
    assert r.loc["2025-08-25", "2327"] == pytest.approx(143 / 136.5 - 1)
    assert r.loc["2017-08-18", "2327"] == pytest.approx(177 / 168.13 - 1)
    assert (audit["kind"] == "unresolved").sum() == 0

    datasets = [c["dataset"] for c in client.calls]
    assert datasets.count("TaiwanStockPrice") == 2
    assert datasets.count("TaiwanStockDividendResult") == 2
    assert datasets.count("TaiwanStockParValueChange") == 1
    # Only 2327 has a gap that dividends + par value don't explain.
    cap_calls = [c for c in client.calls if c["dataset"] == "TaiwanStockCapitalReductionReferencePrice"]
    assert [c["data_id"] for c in cap_calls] == ["2327"]


def test_second_run_is_served_from_cache(client):
    D.load_finmind(["2412", "2327"], "2017-01-01", "2025-12-31", client)
    n = len(client.calls)
    D.load_finmind(["2412", "2327"], "2017-01-01", "2025-12-31", client)
    assert len(client.calls) == n


def test_universe_keeps_delisted_and_drops_non_common(client):
    uni = D.stock_universe(client, start="2016-01-01")
    got = dict(zip(uni["stock_id"], uni["status"]))
    assert got == {"2327": "listed", "2412": "listed", "6594": "delisted"}
    # 0050 ETF, 9105 DR, 6919 innovation board, 1204 delisted before start,
    # 9188 DR, and the old 2412 delisting row (code reused) are all excluded.


def test_api_error_message_is_surfaced(tmp_path, monkeypatch):
    monkeypatch.setattr(D.requests, "get", lambda *a, **k: FakeResp(
        {"status": 402, "msg": "Requests reach the upper limit."}, status=402))
    c = D.FinMindClient(tmp_path, sleep=0, verbose=False, wait_on_quota=False)
    with pytest.raises(D.FinMindError, match="upper limit"):
        c.get("TaiwanStockPrice", data_id="2330", start_date="2024-01-01", end_date="2024-02-01")
    assert not any(tmp_path.rglob("*.csv"))  # failures are not cached


def test_quota_402_waits_then_resumes(tmp_path, monkeypatch):
    replies = iter([({"status": 402, "msg": "upper limit"}, 402),
                    ({"status": 200, "msg": "success", "data": [{"date": "2024-01-02", "stock_id": "2330"}]}, 200)])
    monkeypatch.setattr(D.requests, "get", lambda *a, **k: FakeResp(*next(replies)))
    slept = []
    monkeypatch.setattr(D.time, "sleep", slept.append)
    c = D.FinMindClient(tmp_path, sleep=0, verbose=False)
    df = c.get("TaiwanStockPrice", data_id="2330", start_date="2024-01-01", end_date="2024-01-05")
    assert len(df) == 1 and slept[0] == D.FinMindClient.QUOTA_WAIT_S


def test_other_4xx_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def bad(*a, **k):
        calls.append(1)
        return FakeResp({"status": 400, "msg": "Your level is free."}, status=400)

    monkeypatch.setattr(D.requests, "get", bad)
    c = D.FinMindClient(tmp_path, sleep=0, verbose=False)
    with pytest.raises(D.FinMindError):
        c.get("TaiwanStockDividendResult", start_date="2024-01-01")
    assert len(calls) == 1
