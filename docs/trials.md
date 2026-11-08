# Trial log

Every variant run on real data counts toward multiple testing, even the ones that were
thrown away. The Deflated Sharpe Ratio in a report only knows about the strategies in that
one run, so the real number of trials lives here.

| # | Date | What changed | Research-window result (excess Sharpe / DSR) | Kept? |
|---|---|---|---|---|
| 1 | | Baseline: 5 price/volume factors, composite, ML; `--sample 300 --holdout-start 2023-01-01` | | |
