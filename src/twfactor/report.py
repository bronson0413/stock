"""End-to-end study: factors -> diagnostics -> backtests -> markdown report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A
from .backtest import Costs, run_backtest
from .factors import compute_all, standardize
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


def run_study(panel, out_dir: str | Path, universe_kwargs: dict | None = None,
              horizon: int = 21, top_frac: float = 0.2, include_ml: bool = True,
              costs: Costs = Costs(), title: str = "Factor study") -> dict:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    mask = investable(panel, **(universe_kwargs or {}))
    raw = compute_all(panel)
    z = {k: standardize(v, mask) for k, v in raw.items()}
    dates = A.rebalance_dates(panel.ret.index)
    dates = dates[(mask.loc[dates].sum(axis=1) >= 50).to_numpy()]
    if len(dates) < 24:
        raise RuntimeError(f"Only {len(dates)} usable rebalance dates; need >= 24")

    signals = dict(z)
    signals["composite_equal"] = sum(z.values()) / len(z)
    if include_ml:
        signals["ml_walk_forward"] = walk_forward_signal(panel, raw, mask, dates, horizon=horizon)
    if panel.true_signal is not None:
        signals["TRUE_SIGNAL (synthetic oracle)"] = panel.true_signal

    fwd = A.forward_returns(panel.ret, horizon)
    ic_rows, quant, decay, results, ic_series = {}, {}, {}, {}, {}
    for name, sig in signals.items():
        ic = A.rank_ic(sig, fwd, mask, dates)
        ic_series[name] = ic
        ic_rows[name] = A.ic_summary(ic)
        q = A.quantile_returns(sig, fwd, mask, dates)
        quant[name] = q.mean() * (252 / horizon)
        if name in z:
            decay[name] = A.ic_decay(sig, panel.ret, mask, dates)["ic_mean"]
        results[name] = run_backtest(panel, sig, mask, dates, top_frac=top_frac, costs=costs)

    # Common evaluation window: start when every strategy (incl. ML) is live.
    starts = [ic_series[n].index.min() for n in results if len(ic_series[n])]
    start = max(starts)
    excess = {n: r.excess.loc[start:] for n, r in results.items()}
    trial_names = [n for n in results if not n.startswith("TRUE_SIGNAL")]
    trials = [per_period_sharpe(excess[n]) for n in trial_names]

    perf = {}
    for n, r in results.items():
        st = perf_stats(r.daily["net"].loc[start:])
        ex = perf_stats(excess[n])
        d = r.daily.loc[start:]
        perf[n] = {
            "net_cagr": st["cagr"], "net_sharpe": st["sharpe"], "max_dd": st["max_drawdown"],
            "excess_ann": ex["cagr"], "excess_sharpe": ex["sharpe"],
            "ann_turnover": d["turnover"].sum() / (len(d) / 252),
            "ann_cost": d["cost"].sum() / (len(d) / 252),
            "DSR": deflated_sharpe(excess[n], trials) if n in trial_names else float("nan"),
            "blocked_trades": r.blocked,
        }
    bench = perf_stats(results[trial_names[0]].daily["benchmark"].loc[start:])

    # Year-by-year excess return: is it one lucky year?
    yearly = pd.DataFrame({n: (1 + e).groupby(e.index.year).prod() - 1 for n, e in excess.items()}).T

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, n in enumerate(trial_names):
        (1 + excess[n]).cumprod().plot(ax=ax, color=SERIES[i % len(SERIES)], linewidth=1.6, label=n)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_title("Cumulative excess return vs equal-weight universe (net of costs)", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("")
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures" / "excess_equity.png", dpi=150)
    plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, n in enumerate(dec.columns):
        ax.plot(range(len(dec)), dec[n], marker="o", markersize=4, linewidth=1.6, color=SERIES[i], label=n)
    ax.set_xticks(range(len(dec)), [f"{h}d" for h in dec.index])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Mean rank IC by forward horizon", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ic_decay.png", dpi=150)
    plt.close(fig)

    # ---- markdown ----
    ic_df = pd.DataFrame(ic_rows).T
    ic_df.index.name = "signal"
    perf_df = pd.DataFrame(perf).T
    perf_df.index.name = "strategy"
    yearly.index.name = "strategy"
    pct = "{:.1%}"
    md = [
        f"# {title}",
        "",
        f"- Period evaluated: {start.date()} to {panel.ret.index[-1].date()} "
        f"({len(dates)} monthly rebalances in total; metrics use the window where all strategies are live)",
        f"- Universe: point-in-time investable filter, median {int(mask.loc[dates].sum(axis=1).median())} stocks per rebalance",
        f"- Portfolio: long-only, top {top_frac:.0%} by signal, equal weight, monthly; signal at close d, trade at close d+1",
        f"- Costs: buy {costs.buy_rate:.4%}, sell {costs.sell_rate:.4%} (commission x{costs.commission_discount}, 0.3% tax, slippage {costs.slippage:.2%}/side)",
        f"- Benchmark (equal-weight universe): CAGR {bench['cagr']:.1%}, Sharpe {bench['sharpe']:.2f}",
        f"- DSR computed over {len(trials)} trials (every strategy in the table except the oracle)",
        "",
        "## 1. Predictive power (rank IC, 21-day forward, monthly samples)",
        "",
        _md_table(ic_df, {"hit_rate": pct, "n_periods": "{:.0f}"}),
        "",
        "## 2. Backtest (net of costs)",
        "",
        _md_table(perf_df, {"net_cagr": pct, "max_dd": pct, "excess_ann": pct,
                            "ann_turnover": "{:.1f}x", "ann_cost": pct, "blocked_trades": "{:.0f}"}),
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
        _md_table(A.factor_correlation(z, mask, dates)),
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return {"ic": ic_df, "perf": perf_df, "yearly": yearly, "decay": dec, "results": results}
