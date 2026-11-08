"""Return construction on real FinMind rows (see tests/fixtures/README.md).

These are the cases where a naive implementation silently produces garbage:
on ex-dividend / capital-reduction / par-value-change days FinMind reports
spread = 0, so both pct_change and close-minus-spread are wrong.
"""

from pathlib import Path

import pandas as pd
import pytest

from twfactor.data import build_returns, events_from_finmind

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def built():
    rd = lambda f: pd.read_csv(FIX / f, dtype={"stock_id": str})
    ev = events_from_finmind(rd("dividend_result.csv"), rd("capital_reduction.csv"),
                             rd("par_value_change.csv"))
    return build_returns(rd("prices.csv"), ev)


def _r(ret, date, sid):
    return ret.loc[pd.Timestamp(date), sid]


@pytest.mark.parametrize("date,sid,expected,naive", [
    ("2024-07-04", "2412", 118.5 / 120.74 - 1, 118.5 / 125.5 - 1),  # cash dividend
    ("2025-08-25", "2327", 143 / 136.5 - 1, 143 / 546 - 1),         # par value change 4:1
    ("2017-08-18", "2327", 177 / 168.13 - 1, 177 / 120.5 - 1),      # cash capital reduction
])
def test_event_day_uses_reference_price(built, date, sid, expected, naive):
    ret, *_ = built
    assert _r(ret, date, sid) == pytest.approx(expected, abs=1e-9)
    assert abs(naive - expected) > 0.03  # the naive answer really is badly wrong


def test_spread_alone_would_give_zero(built):
    """Documents the trap: FinMind's spread is 0 on these days."""
    prices = pd.read_csv(FIX / "prices.csv", dtype={"stock_id": str})
    rows = prices.set_index(["date", "stock_id"]).loc[
        [("2024-07-04", "2412"), ("2025-08-25", "2327"), ("2017-08-18", "2327")], "spread"]
    assert (rows == 0).all()


def test_normal_days_are_plain_close_to_close(built):
    ret, close, *_ = built
    assert _r(ret, "2024-07-05", "2412") == pytest.approx(120 / 118.5 - 1)
    assert _r(ret, "2025-08-26", "2327") == pytest.approx(138.5 / 143 - 1)


def test_unexplained_gap_is_nan_not_guessed():
    prices = pd.read_csv(FIX / "prices.csv", dtype={"stock_id": str})
    ret, _, _, audit = build_returns(prices, events=None)  # no corporate-action data at all
    assert pd.isna(_r(ret, "2024-07-04", "2412"))
    assert pd.isna(_r(ret, "2025-08-25", "2327"))
    assert set(audit["kind"]) == {"unresolved"}
    assert len(audit) == 3


def test_audit_lists_each_correction(built):
    *_, audit = built
    got = {(str(d.date()), s, k) for d, s, k in audit[["date", "stock_id", "kind"]].itertuples(index=False)}
    assert got == {("2024-07-04", "2412", "dividend"),
                   ("2025-08-25", "2327", "par_value"),
                   ("2017-08-18", "2327", "capital_reduction")}


def test_conflicting_events_on_same_day_are_dropped():
    div = pd.DataFrame({"date": ["2024-07-04"], "stock_id": ["2412"], "before_price": [125.5], "reference_price": [120.74]})
    par = pd.DataFrame({"date": ["2024-07-04"], "stock_id": ["2412"], "before_close": [125.5], "after_ref_close": [60.0]})
    assert events_from_finmind(div, None, par).empty


def test_duplicate_rows_do_not_break_previous_close():
    prices = pd.read_csv(FIX / "prices.csv", dtype={"stock_id": str})
    dup = pd.concat([prices, prices[prices["date"] == "2024-07-05"]], ignore_index=True)
    ret, *_ = build_returns(dup, None)
    assert _r(ret, "2024-07-05", "2412") == pytest.approx(120 / 118.5 - 1)
