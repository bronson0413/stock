"""Command line entry point.

    twfactor synthetic --out reports/synthetic
    twfactor finmind --start 2015-01-01 --end 2025-12-31 --out reports/tw
    twfactor audit  --start 2015-01-01 --end 2025-12-31 --stocks 2330 2317
"""

from __future__ import annotations

import argparse

from .backtest import Costs
from .data import audit_ex_dividend_gaps, list_stocks, load_finmind, make_synthetic
from .report import run_study


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="twfactor")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="reports/out")
    common.add_argument("--top-frac", type=float, default=0.2)
    common.add_argument("--commission-discount", type=float, default=1.0,
                        help="e.g. 0.6 for a 6-fold broker discount (default: no discount)")
    common.add_argument("--no-ml", action="store_true")

    s = sub.add_parser("synthetic", parents=[common], help="run on simulated data with a planted signal")
    s.add_argument("--stocks", type=int, default=300)
    s.add_argument("--days", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("finmind", parents=[common], help="run on real TWSE/TPEx data")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--stocks", nargs="*", help="stock ids; default = all listed common stocks")
    f.add_argument("--max-stocks", type=int, default=None, help="cap for a quick first run")
    f.add_argument("--min-avg-value", type=float, default=2e7)

    a = sub.add_parser("audit", help="list days where pct_change and reference-price return disagree")
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--stocks", nargs="+", required=True)

    args = p.parse_args(argv)

    if args.cmd == "audit":
        panel = load_finmind(args.stocks, args.start, args.end)
        print(audit_ex_dividend_gaps(panel.close, panel.ret).to_string())
        return

    costs = Costs(commission_discount=args.commission_discount)
    if args.cmd == "synthetic":
        panel = make_synthetic(args.stocks, args.days, seed=args.seed)
        res = run_study(panel, args.out, top_frac=args.top_frac, include_ml=not args.no_ml,
                        costs=costs, title="Synthetic market (planted signal) -- engine validation")
    else:
        ids = args.stocks or list_stocks()
        if args.max_stocks:
            ids = ids[: args.max_stocks]
        print(f"Loading {len(ids)} stocks from FinMind (cached under data/cache)...")
        panel = load_finmind(ids, args.start, args.end)
        res = run_study(panel, args.out, universe_kwargs={"min_avg_value": args.min_avg_value},
                        top_frac=args.top_frac, include_ml=not args.no_ml, costs=costs,
                        title=f"Taiwan equities {args.start} to {args.end}")
    print(res["perf"].round(3).to_string())
    print(f"\nReport written to {args.out}/report.md")


if __name__ == "__main__":
    main()
