import pytest

from twfactor.analysis import rebalance_dates
from twfactor.data import make_synthetic
from twfactor.universe import investable


@pytest.fixture(scope="session")
def panel():
    return make_synthetic(n_stocks=200, n_days=1100, seed=1)


@pytest.fixture(scope="session")
def mask(panel):
    return investable(panel)


@pytest.fixture(scope="session")
def dates(panel, mask):
    d = rebalance_dates(panel.ret.index)
    return d[(mask.loc[d].sum(axis=1) >= 50).to_numpy()]
