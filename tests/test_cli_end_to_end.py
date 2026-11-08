"""Run the real-data CLI path end to end against a fake FinMind API whose
prices come from the synthetic market. Catches wiring bugs between loader,
universe, industry labels, holdout and report that unit tests cannot see."""

import numpy as np
import pandas as pd

import twfactor.data as D
from twfactor.cli import main


class FakeResp:
    def __init__(self, payload):
        self.payload, self.status_code = payload, 200

    def json(self):
        return self.payload


def _finmind_rows(panel):
    rows = []
    for sid in panel.ret.columns:
        c = panel.close[sid].dropna()
        v = panel.value[sid].reindex(c.index)
        spread = c.diff().fillna(0.0)
        rows.append(pd.DataFrame({
            "date": c.index.strftime("%Y-%m-%d"), "stock_id": sid.replace("S", "9"),
            "Trading_Volume": 1000, "Trading_money": v.to_numpy(), "open": c.to_numpy(),
            "max": c.to_numpy(), "min": c.to_numpy(), "close": c.to_numpy(),
            "spread": spread.to_numpy(), "Trading_turnover": 10}))
    return pd.concat(rows, ignore_index=True)


def test_finmind_command_end_to_end(tmp_path, monkeypatch):
    panel = D.make_synthetic(n_stocks=80, n_days=900, seed=2, start="2016-01-04")
    prices = _finmind_rows(panel)
    ids = sorted(prices["stock_id"].unique())
    # 4-digit codes not starting with 0 and not 91xx (those are filtered as DRs)
    ids_map = {sid: f"{1000 + i}" for i, sid in enumerate(ids)}
    prices["stock_id"] = prices["stock_id"].map(ids_map)
    ids = sorted(ids_map.values())
    delisted = ids[-5:]                                  # pretend the last five delisted
    info = pd.DataFrame({"stock_id": ids[:-5], "stock_name": "x", "type": "twse",
                         "industry_category": [f"IND{i % 6}" for i in range(len(ids) - 5)],
                         "date": "2024-12-04"})
    dl = pd.DataFrame({"date": "2019-06-01", "stock_id": delisted, "stock_name": "gone"})
    empty = pd.DataFrame(columns=["date", "stock_id"])
    rng = np.random.default_rng(0)
    per = prices[["date", "stock_id"]].copy()
    per["PBR"] = rng.uniform(0.5, 5, len(per)).round(2)
    per["PER"] = rng.uniform(5, 40, len(per)).round(2)
    per["dividend_yield"] = rng.uniform(0, 7, len(per)).round(2)
    months = pd.period_range("2015-01", "2019-12", freq="M")
    rev = pd.DataFrame([(sid, m.year, m.month, float(rng.uniform(1e8, 2e8)))
                        for sid in ids for m in months],
                       columns=["stock_id", "revenue_year", "revenue_month", "revenue"])
    rev["date"] = [str((pd.Period(year=y, month=m, freq="M") + 1).start_time.date())
                   for y, m in zip(rev["revenue_year"], rev["revenue_month"])]

    def get(url, params, headers, timeout):
        ds, sid = params["dataset"], params.get("data_id")
        t = {"TaiwanStockPrice": prices, "TaiwanStockInfo": info, "TaiwanStockDelisting": dl,
             "TaiwanStockDividendResult": empty, "TaiwanStockParValueChange": empty,
             "TaiwanStockCapitalReductionReferencePrice": empty,
             "TaiwanStockPER": per, "TaiwanStockMonthRevenue": rev}[ds]
        if sid is not None:
            t = t[t["stock_id"] == sid]
        return FakeResp({"status": 200, "msg": "success", "data": t.to_dict("records")})

    monkeypatch.setattr(D.requests, "get", get)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "rep"
    main(["finmind", "--start", "2016-01-01", "--end", "2019-12-31", "--sample", "70",
          "--no-ml", "--industry-neutral", "--holdout-start", "2019-01-01",
          "--min-avg-value", "0", "--out", str(out)])

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "industry-neutralised" in report
    assert "## 7. Holdout" in report
    assert "Survivorship:" in report
    for f in ("book_to_price", "earnings_yield", "dividend_yield", "revenue_growth", "revenue_high"):
        assert f"| {f} |" in report, f
    uni = pd.read_csv(out / "universe.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
    assert set(uni.loc[uni["status"] == "delisted", "stock_id"]) == set(delisted)
