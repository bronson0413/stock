"""End-to-end study: factors -> diagnostics -> backtests -> markdown report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: F401  (used via matplotlib.dates in _small_multiples)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A
from .backtest import Costs, run_backtest
from .factors import composite, compute_all, neutralize, standardize
from .metrics import deflated_sharpe, per_period_sharpe, perf_stats
from .ml import walk_forward_signal
from .universe import investable

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6", "#6b6b66"]
QUANT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _small_multiples(series: dict, kind: str, title: str, path, baseline: float = 0.0,
                     ncol: int = 4) -> None:
    """One small panel per series, single color, shared y-axis.

    Used instead of one chart with a line per strategy: with ten-plus
    strategies a categorical palette runs out and colors would repeat.
    """
    names = list(series)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.3 * nrow), sharey=True, squeeze=False)
    for ax, n in zip(axes.ravel(), names):
        v = series[n]
        if kind == "line":
            ax.plot(v.index, v.values, color=SERIES[0], linewidth=1.4)
            ax.tick_params(axis="x", labelrotation=0)
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        else:
            ax.bar(range(len(v)), v.values, color=SERIES[0], width=0.65)
            ax.set_xticks(range(len(v)), list(v.index))
        ax.axhline(baseline, color=MUTED, linewidth=0.8)
        ax.set_title(n, fontsize=8, color=INK, loc="left")
        _style(ax)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c in df.columns:
            v = row[c]
            f = fmt.get(c, "{:.3f}")
            cells.append(f.format(v) if isinstance(v, (int, float, np.floating)) and pd.notna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _window_perf(results: dict, start, end, trial_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Performance table for results restricted to [start, end)."""
    sl = lambda x: x.loc[(x.index >= start) & (x.index < end)]
    excess = {n: sl(r.excess) for n, r in results.items()}
    trials = [per_period_sharpe(excess[n]) for n in trial_names]
    perf = {}
    for n, r in results.items():
        d = sl(r.daily)
        st, ex = perf_stats(d["net"]), perf_stats(excess[n])
        years = len(d) / 252
        gross_ex = perf_stats(d["gross"] - d["benchmark"])
        perf[n] = {
            "net_cagr": st["cagr"], "net_sharpe": st["sharpe"], "max_dd": st["max_drawdown"],
            "gross_excess_ann": gross_ex["cagr"],
            "excess_ann": ex["cagr"], "excess_sharpe": ex["sharpe"],
            "ann_turnover": d["turnover"].sum() / years,
            "ann_cost": d["cost"].sum() / years,
            "DSR": deflated_sharpe(excess[n], trials) if n in trial_names else float("nan"),
            "blocked_trades": d["blocked"].sum(),
            "delisted_exits": d["delisted_exits"].sum(),
        }
    df = pd.DataFrame(perf).T
    df.index.name = "strategy"
    return df, excess


def run_study(panel, out_dir: str | Path, universe_kwargs: dict | None = None,
              horizon: int = 21, top_frac: float = 0.2, include_ml: bool = True,
              costs: Costs = Costs(), title: str = "Factor study",
              holdout_start: str | None = None, notes: list[str] | None = None,
              industry_neutral: bool = False) -> dict:
    """Run every factor through diagnostics and the backtest and write report.md.

    holdout_start: if given, every table and figure is computed on the
    research window only, and the period from holdout_start on is reported
    once, separately, at the end. IC dates whose forward-return window would
    reach into the holdout are excluded from the research statistics.
    """
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    mask = investable(panel, **(universe_kwargs or {}))
    raw = compute_all(panel)
    z = {k: standardize(v, mask) for k, v in raw.items()}
    idx = panel.ret.index
    dates = A.rebalance_dates(idx)
    dates = dates[(mask.loc[dates].sum(axis=1) >= 50).to_numpy()]
    if len(dates) < 24:
        raise RuntimeError(f"Only {len(dates)} usable rebalance dates; need >= 24")

    end_all = idx[-1] + pd.Timedelta(days=1)
    if holdout_start is not None:
        h0 = pd.Timestamp(holdout_start)
        cut = idx.searchsorted(h0)
        # forward window of an IC date d ends at position(d) + 1 + horizon
        res_dates = pd.DatetimeIndex([d for d in dates if idx.get_loc(d) + 1 + horizon < cut])
        hold_dates = dates[dates >= h0]
        if len(hold_dates) < 6:
            raise RuntimeError("Holdout has fewer than 6 rebalances; move holdout_start earlier")
    else:
        h0, res_dates, hold_dates = end_all, dates, pd.DatetimeIndex([])

    notes = list(notes or [])
    industry = None
    if industry_neutral:
        if panel.industry is None:
            raise ValueError("industry_neutral=True needs panel.industry")
        industry = panel.industry
        z = {k: neutralize(v, industry, dates) for k, v in z.items()}
        n_ind = industry.reindex(panel.ret.columns).nunique()
        notes.append(f"Factors are industry-neutralised ({n_ind} industry labels, today's classification)")

    signals = dict(z)
    signals["composite_equal"] = composite(z)
    if include_ml:
        signals["ml_walk_forward"] = walk_forward_signal(panel, raw, mask, dates, horizon=horizon,
                                                         industry=industry)
    if panel.true_signal is not None:
        signals["TRUE_SIGNAL (synthetic oracle)"] = panel.true_signal

    fwd = A.forward_returns(panel.ret, horizon)
    ic_rows, quant, decay, results, ic_series = {}, {}, {}, {}, {}
    for name, sig in signals.items():
        ic = A.rank_ic(sig, fwd, mask, res_dates)
        ic_series[name] = ic
        ic_rows[name] = A.ic_summary(ic)
        # Coverage: how many investable stocks actually have this signal on a
        # rebalance date. Fundamentals start at different dates per stock.
        ic_rows[name]["median_names"] = float(
            (sig.reindex(res_dates).notna() & mask.loc[res_dates]).sum(axis=1).median())
        q = A.quantile_returns(sig, fwd, mask, res_dates)
        quant[name] = q.mean() * (252 / horizon)
        # Mean-based check. Rank IC follows medians, and the median of a
        # compounded return falls with volatility even when the mean does
        # not, so rank IC alone can "find" a low-volatility effect that is
        # not in the returns. The top-minus-bottom quintile mean spread can't.
        if len(q.columns) >= 2:
            spread = q[q.columns.max()] - q[q.columns.min()]
            ic_rows[name]["q5_q1_ann"] = spread.mean() * (252 / horizon)
            ic_rows[name]["q5_q1_t_nw"] = A.newey_west_tstat(spread, max(2, horizon // 21 + 1))
        if name in z:
            decay[name] = A.ic_decay(sig, panel.ret, mask, res_dates,
                                     cutoff=h0 if len(hold_dates) else None)["ic_mean"]
        results[name] = run_backtest(panel, sig, mask, dates, top_frac=top_frac, costs=costs)

    # Common evaluation window: start when every strategy (incl. ML) is live.
    start = max(ic_series[n].index.min() for n in results if len(ic_series[n]))
    trial_names = [n for n in results if not n.startswith("TRUE_SIGNAL")]
    perf_df, excess = _window_perf(results, start, h0, trial_names)
    bench_d = results[trial_names[0]].daily["benchmark"]
    bench = perf_stats(bench_d.loc[(bench_d.index >= start) & (bench_d.index < h0)])

    # Year-by-year excess return: is it one lucky year?
    yearly = pd.DataFrame({n: (1 + e).groupby(e.index.year).prod() - 1 for n, e in excess.items()}).T

    # ---- figures ----
    _small_multiples({n: (1 + excess[n]).cumprod() for n in trial_names}, "line",
                     "Cumulative excess return vs equal-weight universe, net of costs (1.0 = benchmark)",
                     out / "figures" / "excess_equity.png", baseline=1.0)

    names_q = list(quant)
    ncol = 4
    nrow = int(np.ceil(len(names_q) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow), sharey=True)
    for ax, n in zip(np.ravel(axes), names_q):
        v = quant[n]
        ax.bar([str(i) for i in v.index], v.values, color=QUANT[: len(v)], width=0.7)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(n, fontsize=8, color=INK, loc="left")
        _style(ax)
    for ax in np.ravel(axes)[len(names_q):]:
        ax.axis("off")
    fig.suptitle(f"Annualised {horizon}d forward return by quintile (1 = lowest score)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "figures" / "quintiles.png", dpi=150)
    plt.close(fig)

    dec = pd.DataFrame(decay)
    _small_multiples({n: dec[n].set_axis([f"{h}d" for h in dec.index]) for n in dec.columns}, "bar",
                     "Mean rank IC by forward horizon", out / "figures" / "ic_decay.png", baseline=0.0)

    # ---- holdout (looked at once, after research is frozen) ----
    hold_df = None
    if len(hold_dates):
        hold_df, _ = _window_perf(results, h0, end_all, trial_names)
        hold_ic = {n: A.rank_ic(sig, fwd, mask, hold_dates).mean() for n, sig in signals.items()}
        hold_df.insert(0, "ic_mean", pd.Series(hold_ic))
        hold_df = hold_df.drop(columns=["DSR"])

    # ---- markdown ----
    ic_df = pd.DataFrame(ic_rows).T
    ic_df.index.name = "signal"
    yearly.index.name = "strategy"
    pct = "{:.1%}"
    perf_fmt = {"net_cagr": pct, "max_dd": pct, "gross_excess_ann": pct, "excess_ann": pct,
                "ann_turnover": "{:.1f}x",
                "ann_cost": pct, "blocked_trades": "{:.0f}", "delisted_exits": "{:.0f}"}
    research_end = (h0 - pd.Timedelta(days=1)).date() if hold_df is not None else idx[-1].date()
    md = [
        f"# {title}",
        "",
        f"- Research window: {start.date()} to {research_end} "
        f"({len(res_dates)} monthly IC dates; metrics use the window where all strategies are live)",
    ]
    if hold_df is not None:
        md.append(f"- **Holdout: {h0.date()} to {idx[-1].date()}** ({len(hold_dates)} rebalances) -- "
                  "excluded from every table except section 7")
    md += [
        f"- Universe: point-in-time investable filter, median {int(mask.loc[dates].sum(axis=1).median())} stocks per rebalance",
        f"- Portfolio: long-only, top {top_frac:.0%} by signal, equal weight, monthly; signal at close d, trade at close d+1",
        f"- Costs: buy {costs.buy_rate:.4%}, sell {costs.sell_rate:.4%} (commission x{costs.commission_discount}, 0.3% tax, slippage {costs.slippage:.2%}/side)",
        f"- Benchmark (equal-weight universe): CAGR {bench['cagr']:.1%}, Sharpe {bench['sharpe']:.2f}",
        f"- DSR computed over {len(trial_names)} trials (every strategy in the table except the oracle). "
        "It only counts what is in this run: every earlier variant you tried is also a trial.",
    ]
    md += [f"- {n}" for n in notes]
    md += [
        "",
        f"## 1. Predictive power (rank IC, {horizon}-day forward, monthly samples)",
        "",
        _md_table(ic_df, {"hit_rate": pct, "n_periods": "{:.0f}", "q5_q1_ann": pct,
                          "median_names": "{:.0f}"}),
        "",
        "`q5_q1_*` = mean forward return of the top minus the bottom quintile. When rank IC is "
        "significant but this is not, the 'effect' is likely volatility drag on medians, not returns. "
        "`median_names` = investable stocks with a value for the signal on a typical rebalance date.",
        "",
        "## 2. Backtest (net of costs)",
        "",
        _md_table(perf_df, perf_fmt),
        "",
        "`gross_excess_ann` is before costs and `excess_ann` after; the gap is what trading costs take.",
        "",
        "![excess](figures/excess_equity.png)",
        "",
        "## 3. Excess return by calendar year",
        "",
        _md_table(yearly, {c: pct for c in yearly.columns}),
        "",
        "## 4. Quintile monotonicity",
        "",
        "![quintiles](figures/quintiles.png)",
        "",
        "## 5. IC decay",
        "",
        _md_table(dec.rename_axis("horizon_days")),
        "",
        "![decay](figures/ic_decay.png)",
        "",
        "## 6. Factor correlation (average cross-sectional rank correlation)",
        "",
        _md_table(A.factor_correlation(z, mask, res_dates)),
        "",
    ]
    if hold_df is not None:
        md += [
            "## 7. Holdout (out-of-sample, not used for any decision above)",
            "",
            _md_table(hold_df, perf_fmt),
            "",
            "Read this once. If you change a factor or parameter after seeing it, the holdout",
            "is no longer out-of-sample.",
            "",
        ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return {"ic": ic_df, "perf": perf_df, "holdout": hold_df, "yearly": yearly,
            "decay": dec, "results": results}
