# Changelog

## 0.1.0 — 2026-10-02

- First version: Qwen3.6-35B-A3B on one DGX Spark (TP1) with TensorFold v0.6.2 (installed from upstream, unmodified),
  drafting with the checkpoint's own MTP layer, behind one `run.sh`, configured through `cluster.env`, set up with
  `./setup.sh` (dgx-spark-recipe-kit v0.5.1 under `kit/`).
- Memory profiles: `MEM_GB=auto` (default) picks `64` below 80 GiB of memory: `PARALLEL=4`, `CONTEXT=65536`,
  TensorFold startup estimate 34.7 GiB.
- Benched on our Spark with `CONCURRENT=1,8 bench/bench.sh`: default 160.5 / 133.3 tok/s single-stream code / prose,
  558.5 / 471.7 tok/s total at 8 users (greedy code / chat), cold prefill 7,112 tok/s at ~7.3K tokens; 64 GB profile
  (simulated) 159.9 / 132.8 and 414.6 / 344.7 tok/s, peak 38.1 GiB with four streams filled to ~63K tokens. Smoke
  test passed and 54 / 54 concurrent replies equal to the same request alone in every run. `verified: true`.
- `scripts/smoke-test.sh` runs the kit's smoke test with `SMOKE_NO_THINK=1` (kit 0.5.1): the arithmetic, fact and
  code checks go with thinking off, since Qwen3.6 thinks past their token budgets with it on.
