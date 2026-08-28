#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Concurrency sweep for a TP=3 GLM-5.3-Flash deployment.

Measures aggregate throughput at several concurrency levels and reports both
the aggregate (what the cluster delivers) and the per-stream rate (what one
caller experiences).

Three things worth knowing before you trust the output:

  * Measure at the HEAD NODE ONLY. Under tensor parallelism the head's
    endpoint serves the whole cluster -- the workers listen on nothing and
    emit no completions of their own. Summing per-node would double-count.

  * Make sure nothing else is using the endpoint. A single in-flight request
    from somewhere else is enough to skew a run badly. Check first:
        curl -s http://HOST:8888/metrics | grep num_requests_running
    A run that overlaps other traffic measures contention, not capacity.

  * Every request here sends chat_template_kwargs {"enable_thinking": false}.
    GLM-5.3 is a reasoning model: with thinking ON, the measured tokens
    include an unbounded, prompt-sensitive thinking prelude, and two runs of
    the "same" benchmark can measure very different workloads. Benchmark the
    decode path, not the model's mood.

Usage:
    benchmark_tp3.py [--host HOST:PORT] [--model NAME] [--max-tokens N]
                     [--concurrency 1,2,4,8] [--runs N] [--thinking]
                     [--json OUT]
"""
import argparse
import datetime as dt
import json
import statistics
import threading
import time
import urllib.request


def fetch_model_id(base):
    """Ask the server what it is serving, so the caller need not hardcode it."""
    with urllib.request.urlopen(f"{base}/v1/models", timeout=20) as r:
        return json.load(r)["data"][0]["id"]


def one_request(base, model, prompt, max_tokens, thinking, out, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        usage = d.get("usage", {})
        out.append({
            "i": idx,
            "latency_s": time.time() - t0,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
        })
    except Exception as exc:                     # noqa: BLE001 - report, don't crash the sweep
        out.append({"i": idx, "latency_s": -1, "completion_tokens": 0,
                    "prompt_tokens": 0, "error": str(exc)})


def run_level(base, model, prompt, max_tokens, thinking, n):
    """Fire n requests at once and time the whole batch."""
    results = []
    threads = [threading.Thread(target=one_request,
                                args=(base, model, prompt, max_tokens,
                                      thinking, results, i))
               for i in range(n)]
    w0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - w0

    ok = [r for r in results if r["latency_s"] > 0]
    if not ok:
        return {"concurrency": n, "ok": 0, "of": n, "failed": True}

    total_out = sum(r["completion_tokens"] for r in ok)
    mean_latency = statistics.mean(r["latency_s"] for r in ok)
    return {
        "concurrency": n,
        "ok": len(ok),
        "of": n,
        "wall_s": round(wall, 2),
        "output_tokens": total_out,
        # Aggregate: what the cluster delivers in total.
        "aggregate_tok_s": round(total_out / wall, 1),
        # Per-stream: what a single caller sees. Falls as concurrency rises.
        "per_stream_tok_s": round(total_out / len(ok) / mean_latency, 1),
        "mean_latency_s": round(mean_latency, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8888",
                    help="head node HOST:PORT (workers serve nothing)")
    ap.add_argument("--model", default=None,
                    help="model id; discovered from /v1/models if omitted")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,2,4,8",
                    help="comma-separated concurrency levels; sweep past "
                         "--max-num-seqs to find the config ceiling")
    ap.add_argument("--runs", type=int, default=3,
                    help="repeats per level; >1 so an outlier is visible")
    ap.add_argument("--thinking", action="store_true",
                    help="leave the reasoning channel ON (default: off, so "
                         "every run measures the same workload)")
    ap.add_argument("--prompt",
                    default="Write a Python function that merges two sorted "
                            "lists. Explain briefly.")
    ap.add_argument("--json", default=None, help="write raw results here")
    args = ap.parse_args()

    base = f"http://{args.host}"
    model = args.model or fetch_model_id(base)
    levels = [int(x) for x in args.concurrency.split(",")]

    started_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()

    print(f"target   : {base}", flush=True)
    print(f"model    : {model}", flush=True)
    print(f"tokens   : {args.max_tokens} per completion, temperature 0", flush=True)
    print(f"thinking : {'ON' if args.thinking else 'off'}", flush=True)
    print(f"runs     : {args.runs} per level\n", flush=True)

    all_results = []
    for n in levels:
        per_level = []
        for _ in range(args.runs):
            r = run_level(base, model, args.prompt, args.max_tokens,
                          args.thinking, n)
            all_results.append(r)
            if r.get("failed"):
                print(f"cc={n:<3} ALL REQUESTS FAILED", flush=True)
                continue
            per_level.append(r["aggregate_tok_s"])
            print(f"cc={n:<3} wall={r['wall_s']:6.2f}s  ok={r['ok']}/{r['of']}  "
                  f"aggregate={r['aggregate_tok_s']:6.1f} tok/s  "
                  f"per-stream={r['per_stream_tok_s']:5.1f} tok/s", flush=True)
        if len(per_level) > 1:
            spread = (max(per_level) - min(per_level)) / statistics.median(per_level)
            # A wide spread usually means something else was using the endpoint.
            flag = "   <-- wide spread, check for other traffic" if spread > 0.25 else ""
            print(f"      median={statistics.median(per_level):6.1f} tok/s  "
                  f"spread={spread * 100:.0f}%{flag}", flush=True)
        print(flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"target": base, "model": model,
                       "max_tokens": args.max_tokens,
                       "thinking": args.thinking,
                       "prompt": args.prompt,
                       "concurrency": levels,
                       "runs": args.runs,
                       "started_at": started_at,
                       "finished_at": dt.datetime.now(dt.UTC).replace(
                           microsecond=0).isoformat(),
                       "results": all_results},
                      fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
