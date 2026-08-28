# Benchmarks

> **First measurement landed 2026-08-28.** The repo now has a validated live
> TP=3 boot and one concurrency sweep against
> `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` on the 3x Spark stack. Treat these
> as baseline numbers for the current stable config, not the final ceiling:
> `MAX_NUM_SEQS=8`, `GPU_MEM_UTIL=0.80`, `RAY_memory_usage_threshold=0.99`,
> `MTP_TOKENS=4`, `MAX_MODEL_LEN=262144`, `MOE_BACKEND=marlin`.

## Measured Baseline

Raw result:
`results/tp3-3x-dgx-spark-glm53-flash-20260828T005308Z.json`

Benchmark command:

```bash
scripts/benchmark_tp3.py \
  --host <head-ip>:8888 \
  --concurrency 1,2,4,8,16 \
  --runs 3 \
  --max-tokens 256 \
  --json results/tp3-3x-dgx-spark-glm53-flash-20260828T005308Z.json
```

Validation passed before and after the sweep (`9 passed, 0 failed`, including
the vision path). `/metrics` showed `num_requests_running=0` and
`num_requests_waiting=0` before and after; all benchmark requests completed.

| concurrency | median aggregate tok/s | median per-stream tok/s | runs aggregate tok/s | note |
|---:|---:|---:|---|---|
| 1 | 36.6 | 36.6 | 36.6, 37.0, 30.3 | single-stream baseline |
| 2 | 46.3 | 25.5 | 46.3, 50.3, 45.1 | stable |
| 4 | 75.5 | 20.5 | 47.3, 76.5, 75.5 | wide spread; first run was slow |
| 8 | 87.4 | 11.8 | 87.4, 85.8, 117.9 | wide spread; one fast outlier |
| 16 | 115.1 | 10.4 | 115.1, 114.8, 126.0 | stable enough; runs in waves under `MAX_NUM_SEQS=8` |

Interpretation: the current production candidate is about **36-37 tok/s
single-stream** and about **115 tok/s aggregate at 16 concurrent requests**.
Because `MAX_NUM_SEQS` is still 8, concurrency 16 is not a pure simultaneous
decode shape; it measures two scheduler waves and is useful as an operator
capacity number. The next tuning run should raise `MAX_NUM_SEQS` and repeat
the sweep after disk cleanup.

## What to expect (derived, not measured)

Reference points: MiaAI's 2-node recipe reports 23–30 tok/s single-stream
and ~72 tok/s aggregate at ×8 (their README); the sibling DSv4 TP=3 cluster
measured its aggregate ceiling at whatever `--max-num-seqs` was set to,
three settings in a row (238 → 431 → 618 tok/s for 6 → 16 → 32).

At TP=3, per rank: ~60 GiB of weights instead of ~91, so

* **single-stream**: bandwidth-bound decode reads ⅓ of the weights per rank
  instead of ½ — expect roughly ×1.3–1.5 over the dual kit (≈ 30–45 tok/s),
  minus interconnect overhead per layer. MTP-4 acceptance carries over.
* **aggregate**: the KV pool grows from the 2-node recipe, but the stable TP=3
  budget is currently `GPU_MEM_UTIL=0.80` with
  `RAY_memory_usage_threshold=0.99`; higher budgets can trip Ray's
  node-memory monitor on GB10 UMA during init. The concurrency ceiling may
  still move past the stock `--max-num-seqs 8`. If the DSv4 lesson transfers
  — and it is the single
  most transferable lesson that repo produced — the first thing that caps
  this cluster will be that knob, not the hardware.
* **context**: KV per token is tiny on this architecture (11 MLA layers ×
  512 B fp8 ≈ 5.6 KB/token; the 34 KDA layers hold constant-size state), so
  the 1M-native window (`MAX_MODEL_LEN=1048576`) should fit comfortably at
  TP=3. Whether *prefill time* makes 1M pleasant is a measurement, not a
  derivation.

## Methodology (locked in now, so the numbers mean something later)

1. **Head node only.** Workers serve nothing; summing per-node double-counts.
2. **Quiet endpoint.** `curl -s :8888/metrics | grep num_requests_running`
   must be 0 before every run — overlapping traffic measures contention.
3. **Thinking OFF** (`benchmark_tp3.py` default). A reasoning model's
   thinking prelude is unbounded and prompt-sensitive; with it on, two runs
   of the "same" benchmark measure different workloads.
4. **Three runs per level, report the median, flag spread > 25%.**
5. Sweep `--concurrency 1,2,4,8,16,32` at `--max-num-seqs` 8, then raise
   `MAX_NUM_SEQS` and re-sweep — the interesting number is where saturation
   stops tracking the knob.
6. `scripts/validate_tp3.sh` must pass before and after every sweep — a
   benchmark of a mis-sharded model is a benchmark of noise.
7. Raw JSON into `results/`, named `tp3-3x-dgx-spark-*.json`, one file per
   configuration.

## The three ways to ruin a measurement (inherited from the DSv4 repo)

* benchmarking against other traffic (see 2 — check, don't assume)
* comparing runs with different speculative settings (MTP on/off changes
  both tok/s and its meaning: accepted draft tokens are cheaper)
* trusting a single run — cold FA2 JIT, a first-touch page migration on UMA,
  or a background template refresh can eat one run invisibly


## Two raw artifacts, on purpose

`results/` holds two sweeps of the same config (`MAX_NUM_SEQS=8`):

* `...T005308Z.json` — the **idle-endpoint baseline** (the numbers quoted
  above; no other traffic on the cluster). Predates the script's run-metadata
  fields; the run parameters are recorded here instead: cc 1,2,4,8,16 × 3
  runs, 256 max tokens, thinking off.
* `...T023328Z.json` — the same sweep **under live agent traffic** (two
  agents idling on the endpoint), full run metadata embedded. cc16 median
  118 tok/s — consistent with the baseline, which is itself a useful result:
  light background load does not collapse aggregate throughput.
