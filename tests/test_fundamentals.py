"""Fundamentals must be used on the date they were public, not the date they describe."""

import numpy as np
import pandas as pd
import pytest

from twfactor.fundamentals import (align_daily_valuation, align_monthly_revenue,
                                   monthly_revenue_features, revenue_available_date)


def _rev(sid="2330", start="2018-01", n=24, base=100.0, growth=0.01, seed=0):
    rng = np.random.default_rng(seed)
    ps = pd.period_range(start, periods=n, freq="M")
    rev = base * (1 + growth) ** np.arange(n) * rng.uniform(0.95, 1.05, n)
    return pd.DataFrame({
        "date": [str((p + 1).start_time.date()) for p in ps],   # FinMind convention: next month's 1st
        "stock_id": sid, "country": "Taiwan", "revenue": rev,
        "revenue_month": ps.month, "revenue_year": ps.year,
    })


def test_availability_is_the_11th_of_next_month():
    d = revenue_available_date(pd.Series([2019, 2019]), pd.Series([3, 12]))
    assert list(d) == [pd.Timestamp("2019-04-11"), pd.Timestamp("2020-01-11")]


def test_march_revenue_is_invisible_until_april_11():
    rev = _rev(start="2017-10", n=19)              # Oct 2017 .. Apr 2019
    idx = pd.bdate_range("2019-04-01", "2019-04-30")
    out = align_monthly_revenue(rev, idx, ["2330"])["rev_yoy_3m"]["2330"]
    feats = monthly_revenue_features(rev).set_index("period")
    feb, mar = feats.loc[pd.Period("2019-02", "M"), "rev_yoy_3m"], feats.loc[pd.Period("2019-03", "M"), "rev_yoy_3m"]
    assert np.isfinite(feb) and np.isfinite(mar) and feb != mar
    # FinMind dates March revenue 2019-04-01, but it is not usable then:
    assert out.loc["2019-04-01"] == pytest.approx(feb)
    assert out.loc["2019-04-10"] == pytest.approx(feb)
    assert out.loc["2019-04-11"] == pytest.approx(mar)


def test_yoy_is_log_ratio_of_three_month_sums():
    rev = _rev(n=15, growth=0.0, seed=1)
    f = monthly_revenue_features(rev).set_index("period")
    r = rev.set_index(pd.PeriodIndex.from_fields(year=rev["revenue_year"], month=rev["revenue_month"], freq="M"))["revenue"]
    p = pd.Period("2019-03", "M")
    expected = np.log(r[p - 2:p].sum() / r[p - 14:p - 12].sum())
    assert f.loc[p, "rev_yoy_3m"] == pytest.approx(expected)
    assert np.isnan(f.loc[pd.Period("2018-12", "M"), "rev_yoy_3m"])   # needs 12 months of history


def test_gap_in_months_does_not_fake_a_yoy():
    rev = _rev(n=20)
    rev = rev[~((rev["revenue_year"] == 2018) & (rev["revenue_month"] == 3))]   # March 2018 missing
    f = monthly_revenue_features(rev).set_index("period")
    assert np.isnan(f.loc[pd.Period("2019-03", "M"), "rev_yoy_3m"])


def test_stale_revenue_is_dropped():
    rev = _rev(start="2018-01", n=15)              # last month Mar 2019 -> public 2019-04-11
    idx = pd.bdate_range("2019-04-11", "2019-08-30")
    out = align_monthly_revenue(rev, idx, ["2330"], max_age_days=75)["rev_yoy_3m"]["2330"]
    assert out.loc["2019-06-24"] == out.loc["2019-04-11"]   # 74 days old: still used
    assert np.isnan(out.loc["2019-06-26"])                  # 76 days old: dropped


def test_revenue_published_after_cutoff_cannot_change_values_before_it():
    rev = _rev(n=30)
    idx = pd.bdate_range("2019-01-01", "2020-06-30")
    cutoff = pd.Timestamp("2019-09-15")
    avail = revenue_available_date(rev["revenue_year"], rev["revenue_month"])
    bad = rev.copy()
    bad.loc[avail > cutoff, "revenue"] *= np.random.default_rng(3).uniform(0.2, 5, (avail > cutoff).sum())
    a = align_monthly_revenue(rev, idx, ["2330"])
    b = align_monthly_revenue(bad, idx, ["2330"])
    for k in a:
        pd.testing.assert_frame_equal(a[k].loc[:cutoff], b[k].loc[:cutoff])
        assert not a[k].loc[cutoff:].equals(b[k].loc[cutoff:])


def test_valuation_forward_fill_is_limited():
    idx = pd.bdate_range("2024-01-01", periods=12)
    per = pd.DataFrame({"date": [str(idx[0].date())], "stock_id": ["2330"],
                        "PER": [20.0], "PBR": [5.0], "dividend_yield": [1.5]})
    out = align_daily_valuation(per, idx, ["2330"], max_fill_days=5)["PBR"]["2330"]
    assert out.iloc[:6].eq(5.0).all()
    assert out.iloc[6:].isna().all()
