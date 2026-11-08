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
from .data import FinMindClient, load_finmind, make_synthetic, stock_universe
from .report import run_study


def _pick_stocks(client: FinMindClient, start: str, stocks, sample: int | None, seed: int, out: Path):
    if stocks:
        return list(stocks)
    uni = stock_universe(client, start)
    out.mkdir(parents=True, exist_ok=True)
    uni.to_csv(out / "universe.csv", index=False)
    n_del = int((uni["status"] == "delisted").sum())
    print(f"Universe: {len(uni)} common stocks ({n_del} delisted since {start} -- kept to avoid survivorship bias)")
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

    s = sub.add_parser("synthetic", parents=[common], help="run on simulated data with a planted signal")
    s.add_argument("--stocks", type=int, default=300)
    s.add_argument("--days", type=int, default=1500)
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
        panel = make_synthetic(args.stocks, args.days, seed=args.seed)
        res = run_study(panel, out, top_frac=args.top_frac, include_ml=not args.no_ml,
                        costs=costs, title="Synthetic market (planted signal) -- engine validation")
    else:
        client = FinMindClient(args.cache)
        ids = _pick_stocks(client, args.start, args.stocks, args.sample, args.seed, out)
        print(f"Loading {len(ids)} stocks (cached under {args.cache}; re-run to resume after a quota stop)...")
        panel, audit = load_finmind(ids, args.start, args.end, client)
        out.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out / "corporate_actions_audit.csv", index=False)
        n_unres = int((audit["kind"] == "unresolved").sum()) if not audit.empty else 0
        print(f"Corporate-action days corrected: {len(audit) - n_unres}; unresolved (set to NaN): {n_unres}")
        res = run_study(panel, out, universe_kwargs={"min_avg_value": args.min_avg_value},
                        top_frac=args.top_frac, include_ml=not args.no_ml, costs=costs,
                        title=f"Taiwan equities {args.start} to {args.end}")
    print(res["perf"].round(3).to_string())
    print(f"\nReport written to {out}/report.md")


if __name__ == "__main__":
    main()
