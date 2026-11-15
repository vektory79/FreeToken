# Route affinity A/B summary

| arm | affinity | tok/s | p50 ms/step | hit mean, % | battery pass | arith pass | prose texts changed vs base |
|---|---|---|---|---|---|---|---|
| aff000 | 0.0 | 22.54 | 43.9 | 35.7 | 5 | 5 | base |
| aff005 | 0.05 | 26.34 | 37.4 | 48.94 | 5 | 5 | 4 |
| aff020 | 0.2 | 38.55 | 25.6 | 73.61 | 5 | 5 | 4 |

## Battery, per arm

| prompt | aff000 | aff005 | aff020 |
|---|---|---|---|
| arith_mult | PASS | PASS | PASS |
| arith_pow | PASS | PASS | PASS |
| arith_mod | PASS | PASS | PASS |
| arith_eq | PASS | PASS | PASS |
| arith_bin | FAIL | FAIL | FAIL |
| arith_prime | PASS | PASS | PASS |
| prose_cache | same | CHANGED | CHANGED |
| prose_japan | same | CHANGED | CHANGED |
| prose_smooth | same | CHANGED | CHANGED |
| prose_window | same | CHANGED | CHANGED |

## Hit-rate lines (per arm)

- aff000: n=19 mean=35.7 values=[29.4, 42.5, 37.5, 35.1, 36.9, 37.6, 32.8, 33.5, 36.2, 34.9, 41.9, 36.7, 39.6, 41.6, 33.3, 31.9, 32.3, 30.0, 34.6]
- aff005: n=19 mean=48.94 values=[42.4, 49.0, 52.7, 45.2, 47.1, 55.6, 49.4, 46.4, 49.3, 49.3, 52.2, 49.8, 47.2, 47.0, 50.5, 46.7, 47.8, 52.2, 50.0]
- aff020: n=19 mean=73.61 values=[61.3, 73.6, 76.8, 75.9, 71.3, 70.1, 76.5, 78.1, 75.7, 73.0, 72.5, 76.6, 73.2, 73.4, 72.9, 75.9, 73.8, 73.7, 74.2]
