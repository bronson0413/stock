Real FinMind / TWSE rows captured 2026-11-08, used to test corporate-action handling.

| Stock | Date | Event | Correct return |
|---|---|---|---|
| 2412 中華電 | 2024-07-04 | cash dividend 4.758 (ref 120.74) | 118.5 / 120.74 - 1 = -1.855% |
| 2327 國巨 | 2025-08-25 | par value change, 4-for-1 (ref 136.5) | 143 / 136.5 - 1 = +4.762% |
| 2327 國巨 | 2017-08-18 | cash capital reduction (ref 168.13) | 177 / 168.13 - 1 = +5.276% |

On all three days FinMind reports `spread = 0` (TWSE shows `X0.00`, "不比價"), so
`close / (close - spread) - 1` gives 0 and `close.pct_change()` gives
-5.6%, -73.8% and +46.9%. Both are wrong; the reference price is required.
