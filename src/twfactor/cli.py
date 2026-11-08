"""Command line entry point.

    twfactor synthetic --out reports/synthetic
    twfactor finmind --start 2016-01-01 --end 2025-12-31 --sample 300 --out reports/tw
    twfactor audit   --start 2016-01-01 --end 2025-12-31 --stocks 2330 2412 2327
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .backtest import Costs
from .data import FinMindClient, FinMindError, load_finmind, make_synthetic, stock_universe
from .report import run_study


def _pick_stocks(uni, stocks, sample: int | None, seed: int) -> list[str]:
    if stocks:
        return list(stocks)
    ids = uni["stock_id"].tolist()
    if sample and sample < len(ids):
        # Random, not the first N codes: low codes are old-economy names
        # (cement, food, textiles), so "first N" is a sector bet.
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=sample, replace=False).tolist())
        print(f"Random sample of {sample} (seed {seed})")
    return ids


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="twfactor")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="reports/out")
    common.add_argument("--top-frac", type=float, default=0.2)
    common.add_argument("--commission-discount", type=float, default=1.0,
                        help="e.g. 0.6 for a 6-fold broker discount (default: no discount)")
    common.add_argument("--no-ml", action="store_true")
    common.add_argument("--industry-neutral", action="store_true",
                        help="rank stocks within their industry instead of across the market")
    common.add_argument("--holdout-start", default=None,
                        help="reserve data from this date as an untouched out-of-sample period")

    s = sub.add_parser("synthetic", parents=[common], help="run on simulated data with a planted signal")
    s.add_argument("--stocks", type=int, default=300)
    s.add_argument("--days", type=int, default=2400)
    s.add_argument("--start", default="2016-01-04")
    s.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("finmind", parents=[common], help="run on real TWSE/TPEx data")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--stocks", nargs="*", help="explicit stock ids (default: full universe incl. delisted)")
    f.add_argument("--sample", type=int, default=None, help="random sample size, for a quicker first run")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--min-avg-value", type=float, default=2e7)
    f.add_argument("--cache", default="data/cache")

    a = sub.add_parser("audit", help="show every corporate-action correction for some stocks")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--stocks", nargs="+", required=True)
    a.add_argument("--cache", default="data/cache")

    args = p.parse_args(argv)
    try:
        _run(args)
    except FinMindError as e:
        raise SystemExit(f"FinMind request failed: {e}")


def _run(args) -> None:
    if args.cmd == "audit":
        _, audit = load_finmind(args.stocks, args.start, args.end, FinMindClient(args.cache))
        if audit.empty:
            print("No corporate-action days in range.")
        else:
            print(audit.to_string(float_format=lambda x: f"{x:.4f}"))
            print("\nCompare `ret` with `naive_pct_change`: the difference is what a naive "
                  "loader would get wrong. Check a few dates against public 除權息 records.")
        return

    costs = Costs(commission_discount=args.commission_discount)
    out = Path(args.out)
    if args.cmd == "synthetic":
        panel = make_synthetic(args.stocks, args.days, seed=args.seed, start=args.start)
        res = run_study(panel, out, top_frac=args.top_frac, include_ml=not args.no_ml,
                        costs=costs, title="Synthetic market (planted signal) -- engine validation",
                        holdout_start=args.holdout_start, industry_neutral=args.industry_neutral)
    else:
        client = FinMindClient(args.cache)
        out.mkdir(parents=True, exist_ok=True)
        uni = stock_universe(client, args.start)
        uni.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")  # BOM so Excel shows 中文
        n_del_all = int((uni["status"] == "delisted").sum())
        print(f"Universe: {len(uni)} common stocks ({n_del_all} delisted since {args.start} -- "
              "kept to avoid survivorship bias)")
        ids = _pick_stocks(uni, args.stocks, args.sample, args.seed)
        industry = uni.set_index("stock_id")["industry_category"]
        print(f"Loading {len(ids)} stocks (cached under {args.cache}; re-run to resume after a quota stop)...")
        panel, audit = load_finmind(ids, args.start, args.end, client, industry=industry)
        audit.to_csv(out / "corporate_actions_audit.csv", index=False, encoding="utf-8-sig")
        n_unres = int((audit["kind"] == "unresolved").sum()) if not audit.empty else 0
        n_fix = len(audit) - n_unres
        print(f"Corporate-action days corrected: {n_fix}; unresolved (set to NaN): {n_unres}")
        loaded = list(panel.ret.columns)
        n_del = int(uni[uni["stock_id"].isin(loaded)]["status"].eq("delisted").sum())
        notes = [
            f"Data: FinMind daily prices for {len(loaded)} stocks"
            + (f" (random sample of {args.sample}, seed {args.seed})" if args.sample and not args.stocks else ""),
            f"Survivorship: {n_del} of the loaded stocks were delisted during the period and are included",
            f"Corporate actions: {n_fix} ex-dividend / capital-reduction / par-value days priced off the "
            f"exchange reference price; {n_unres} unexplained gaps left as missing "
            "(see corporate_actions_audit.csv)",
        ]
        res = run_study(panel, out, universe_kwargs={"min_avg_value": args.min_avg_value},
                        top_frac=args.top_frac, include_ml=not args.no_ml, costs=costs,
                        title=f"Taiwan equities {args.start} to {args.end}",
                        holdout_start=args.holdout_start, notes=notes,
                        industry_neutral=args.industry_neutral)
    print(res["perf"].round(3).to_string())
    if res.get("holdout") is not None:
        print("\nHoldout:")
        print(res["holdout"].round(3).to_string())
    print(f"\nReport written to {out}/report.md")


if __name__ == "__main__":
    main()
