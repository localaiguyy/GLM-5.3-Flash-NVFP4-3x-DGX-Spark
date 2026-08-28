# GLM-5.3-Flash-NVFP4 on 3× DGX Spark (TP=3)

Running **GLM-5.3-Flash** (NVFP4, multimodal image+video, MTP speculative
decoding, 1M-native context) across **three** NVIDIA DGX Sparks with vLLM
tensor parallelism over Ray.

The interesting part is that this **should not work out of the box**. The
model has 64 MLA attention heads and 64 KDA linear-attention heads, and 64
is not divisible by 3. Stock vLLM refuses to start — in four separate
places — and the parts that would *not* refuse would silently drop
checkpoint columns instead. This repo is the set of patches that make TP=3
correct.

> **Inspired by [MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark]**. That repo
> is the excellent 2-node recipe that made this possible, and it is where
> the SM121 kernel layer, the serving profile, the NCCL pins and the UMA
> memory discipline come from. **This is independent work, not a fork** —
> the 2-node recipe works because 64 heads divide 2 cleanly, so none of the
> padding machinery here exists there. Credit for the foundation goes to
> MiaAI-Lab; any mistakes in the TP=3 work are mine.
>
> This is also the second repo in a series:
> [DeepSeek-V4-Flash-DSpark-3x-DGX-Spark] converted MiaAI's DSv4 2-node
> recipe to TP=3. The two models fail TP=3 for **different structural
> reasons**, and comparing the two patches is half the point — see
> [Why this patch looks nothing like the DSv4 one](docs/PATCH-SITES.md#why-this-patch-looks-nothing-like-the-dsv4-one-and-why-that-matters).

---

## Status: release candidate, live validated

Honesty first. What is **verified** right now, against the real
`glm53-flash-arm64-cu130` image tree, the real checkpoint index, and one live
3x Spark boot:

* every patch site **applies cleanly** to the actual image's vLLM tree
  (`0.1.dev20051+g487ecf187`), is idempotent, and **reverts
  byte-identically**;
* the padding arithmetic asserts the real geometry
  (`patches/tp3/glm53_tp_pad.py --test`);
* the shipped loader bodies pass **38 numerical checks** on real tensors
  (`patches/tp3/test_pad_loaders.py`), including two end-to-end proofs —
  a padded TP=3 column+row projection pipeline and a padded TP=3 SwiGLU
  expert each reproduce their TP=1 reference exactly, **with parameters
  pre-filled with NaN** so an un-zeroed pad lane cannot hide;
* the checkpoint analysis (head counts, tensor shapes, the absence of
  attention sinks, expert divisibility) was measured from the checkpoint's
  own safetensors headers, not read off a model card.

The first live baseline also passed `scripts/validate_tp3.sh` before and after
benchmarking (`9 passed, 0 failed`, including the vision path) and measured
about **36-37 tok/s single-stream** and **115 tok/s aggregate at concurrency
16** under the current stable config (`MAX_NUM_SEQS=8`, `GPU_MEM_UTIL=0.80`,
`RAY_memory_usage_threshold=0.99`, marlin MoE). See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and the raw JSON under `results/`.

What is still pending before calling this production-hardened: disk cleanup on
the head node, a longer soak, and a second sweep after raising `MAX_NUM_SEQS`.

---

## The problem in one table

TP sharding divides tensors along head- and channel-dimensions. Whether a
dimension divides decides everything:

| Surface | Size | % 3 | Consequence at TP=3 |
|---|---|---|---|
| MLA heads (11 sparse-attention layers + MTP) | 64 | ❌ | `ValueError` at argument parsing |
| KDA linear-attention heads (34 layers) | 64 | ❌ | assert + `divide()` in the KDA module |
| MoE intermediate (routed + shared experts) | 2048 | ❌ | loader floor-division **drops 2 columns per expert** |
| Vocab | 154880 | ❌ | embedding shard dies *after* NCCL join — reads as a network fault |
| Vision tower heads | 16 | ❌ | ViT shard error — solved by config, not patch |
| Routed expert count | 288 | ✅ | nothing — 288 = 3 × 96 |
| Dense MLP (layers 0–2) | 12288 | ✅ | nothing |
| Indexer heads / router / hyper-connections | 32 / — / — | — | replicated already; nothing |

## The fix: pad the heads, zero the slabs

Pad heads **64 → 66** (22/rank), MoE intermediate **2048 → 2112** (704/rank,
64-aligned), vocab **154880 → 154944** (lcm(64, 3)); the vision tower runs
**data-parallel** (`--mm-encoder-tp-mode data`, replicated per rank) instead
of being sharded at all. Every padded lane is **zero**, and zero is exactly
nothing through this model end to end:

```
pad MLA head : zero q_b/kv_b rows -> zero head output -> zero o_proj cols -> +0 in all-reduce
pad KDA head : zero q/k/v rows    -> delta-rule state never leaves zero  -> zero o_proj cols
pad MoE col  : silu(0) * 0 = 0 through SwiGLU (and its clamped variant)
pad vocab row: stock masked lookup + logits sliced back to the org vocab
```

Unlike the DSv4 model there are **no attention sinks** here (verified across
all 113,074 checkpoint tensors), so there is no `-inf` subtlety — but a new
trap replaces it: the modelopt-NVFP4 path allocates expert weights with
`torch.empty`, and **marlin repacks the full padded tensor after loading**,
so every pad tail is explicitly zero-filled by the patched loaders. Trusting
the allocator here would ship garbage into the kernels while looking
perfectly healthy.

The patch is **14 anchor-matched sites** + one installed arithmetic module.
Eleven came from static analysis of the image tree; the last three were
earned the honest way, one failed boot at a time: a *static* state-shape
classmethod the platform calls before any layer exists (site 12), and the
GB10 FlashInfer-autotune rank-killer MiaAI documents but does not bake the
skips for (sites 13–14). The fourth killer needed no patch at all — Ray's
node-memory monitor SIGKILLs a vLLM actor on GB10 UMA at its default 95%
threshold, silently; `RAY_memory_usage_threshold=0.99` (set by the
launcher) is the fix. Each site is explained, with what needed *no* patch
and how that was established, in
[`docs/PATCH-SITES.md`](docs/PATCH-SITES.md).

## Which checkpoint does this work with?

**Any GLM-5.3-Flash NVFP4 checkpoint with the stock geometry** — the patch
keys on shapes, not weights. Verified name-identical (all 113,074 tensors)
between [`LibertAIDAI/GLM-5.3-Flash-NVFP4`] (the 2-node recipe's pin) and
[`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`]:

```
num_attention_heads 64      kda heads 64 x dim 128     qk/v head_dim 256 (NoPE)
q_lora 1536 / kv_lora 512   n_routed_experts 288       moe_intermediate 2048
vocab 154880                45 layers + 1 MTP          120 shards, 181.3 GiB
```

Two derivative-checkpoint traps are handled explicitly (both hit in
practice): a **stale chat template** snapshotted from the base repo before
its multimodal fix (the launcher refuses a template without `emit_image`;
point `CHAT_TEMPLATE_URL` at the base model), and a **shorter
`quantization_config.ignore` list** that can make vLLM treat BF16 fused
modules as quantized (preflight warns; overlay the reference list). Details:
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Quick start

You need three DGX Sparks that can already run the 2-node recipe, plus a
fabric between them (developed against 100G RoCE; the launcher is
network-agnostic — set the per-node CX7 pins).

Version lock, because `mia/glm53-flash-spark:mm-ray-v1` is a mutable local
tag: everything here was built and validated against base image
`vllm/vllm-openai:glm53-flash-arm64-cu130` at digest
`sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`
(vLLM `0.1.dev20051+g487ecf187`). The patcher's anchors fail loudly on any
other tree — that is by design; re-anchor deliberately, never force.

```bash
# 1. Build MiaAI's images (kernel + mm-ray) from their repo, unchanged
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark ../GLM-5.3-Flash-NVFP4-Dual-DGX-Spark

# 2. Copy the example config and point it at your three nodes
cp config/env.glm53.tp3.example my.env && $EDITOR my.env

# 3. Start it — downloads/syncs weights, ships the patch to every node,
#    applies it inside every container BEFORE vllm imports, launches Ray+vLLM
ENV_FILE=my.env ./start.sh

# 4. Prove it is correct, not just alive
scripts/validate_tp3.sh <head-ip>:8888
```

Every file that has to change, in order:
**[`docs/CHANGES-REQUIRED.md`](docs/CHANGES-REQUIRED.md)**

## Monitoring a long-running cluster

`scripts/glm53-decode-watchdog` keeps a real completion in the loop: a 200
on `/v1/models` is not a decode test (the head answers it from local state
after every worker is gone), so the probe issues an actual completion —
thinking disabled, so the reasoning channel cannot eat the probe budget. Two
consecutive failures = confirmed wedge → it captures py-spy stacks from
**every** rank plus the GPU spin signature *before* alerting, and leaves the
cluster running, because the live process holds the only copy of that state.

Installing it as a service (the copy-paste version — adjust paths/users):

```ini
# /etc/systemd/system/glm53-watchdog.service
[Unit]
Description=GLM-5.3 TP=3 decode watchdog
[Service]
Type=oneshot
User=youruser                    # must own STATE_DIR and hold the sudoers grants below
Environment=GLM53_MODEL=dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4
Environment=GLM53_ENDPOINT=http://localhost:8888/v1/chat/completions
Environment="GLM53_RANKS=<head-fabric-ip> <worker1-fabric-ip> <worker2-fabric-ip>"
Environment=GLM53_SELF_IP=<head-fabric-ip>
Environment="GLM53_CONTAINERS=glm53-flash-head glm53-flash-worker1 glm53-flash-worker2"
Environment=GLM53_STATE_DIR=/var/lib/glm53-watchdog
Environment=GLM53_ALERT_CMD=/usr/local/bin/your-notifier   # optional
ExecStart=/opt/glm53-tp3/scripts/glm53-decode-watchdog

# /etc/systemd/system/glm53-watchdog.timer
[Unit]
Description=Run the GLM-5.3 decode watchdog every 5 minutes
[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
```

The contract, stated explicitly because silent capture failure is the whole
enemy here: `GLM53_STATE_DIR` must be **owned by the unit user** (the script
refuses to start otherwise); the unit user needs passwordless sudo for
`docker exec`, `docker logs` and `nvidia-smi` on every rank (workers reached
over ssh as `GLM53_SSH_USER`); and `GLM53_RANKS` addresses must resolve
**from the head node**, not from your laptop. Then:

```bash
sudo mkdir -p /var/lib/glm53-watchdog && sudo chown youruser /var/lib/glm53-watchdog
sudo systemctl enable --now glm53-watchdog.timer
```

## How correctness is established

Failure modes in this work are silent: a wrong head shard, a non-zero pad
column or a dropped expert column all produce grammatical, confident, wrong
output with nothing in the logs. So correctness is established on four axes:

1. **Arithmetic** — `glm53_tp_pad.py --test`: the real geometry, per-rank
   real/pad splits, identity at TP=1/2/4, the kill-switch.
2. **The shipped loader code, numerically** — `test_pad_loaders.py` exec()s
   the actual replacement bodies from the patcher against real tensors:
   every checkpoint element lands on exactly one rank exactly once; every
   pad lane is zero *even when the parameter starts as NaN* (the
   `torch.empty` trap, simulated); the stock path is byte-equivalent when
   nothing is padded; and the two end-to-end pipelines match TP=1.
3. **Behavioural** — `validate_tp3.sh` runs probes that degrade first under
   a bad shard: exact recall, arithmetic, **sequential state** (34 of 45
   layers are recurrent KDA — a corrupted head shows up in running-state
   tasks before anything else), a needle across ~1.5k tokens, a
   degeneration check, and a solid-color image through the vision path.
4. **The paranoia pass** — the NaN-canary procedure for proving pad slabs
   on a live build, in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Repository layout

```
patches/tp3/apply_tp3_patch.py   the whole patch, as one idempotent script
patches/tp3/glm53_tp_pad.py      the padding arithmetic (importable, tested)
patches/tp3/test_pad_loaders.py  38 numerical checks on the SHIPPED bodies
start.sh                         3-node launcher (derivative of MiaAI's, MIT)
scripts/validate_tp3.sh          correctness probes — run this, not just curl
scripts/glm53-decode-watchdog    liveness probe; captures per-rank state
scripts/benchmark_tp3.py         the concurrency sweep (thinking off)
config/env.glm53.tp3.example     the full 3-node env block
docs/CHANGES-REQUIRED.md         every change needed, in order
docs/PATCH-SITES.md              all sites and why each exists — and the
                                 DSv4 comparison
docs/BENCHMARKS.md               methodology now, numbers after the window
docs/TROUBLESHOOTING.md          the failures and the checkpoint traps
results/                         raw measurements as JSON
```

## Credits

* **[MiaAI-Lab][MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark]** — the 2×
  DGX Spark recipe this work is built on top of and inspired by.
* **[Z.ai](https://huggingface.co/zai-org)** — GLM-5.3-Flash.
* **[LibertAIDAI][`LibertAIDAI/GLM-5.3-Flash-NVFP4`]** — the NVFP4
  quantization, and **[dealignai][`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`]**
  — the geometry-identical derivative.
* **[vLLM](https://github.com/vllm-project/vllm)** / **NVIDIA** — the
  serving engine and the glm5next model tree in the
  `glm53-flash-arm64-cu130` image.

## License

MIT — see [LICENSE](LICENSE).

[MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
[DeepSeek-V4-Flash-DSpark-3x-DGX-Spark]: https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark
[`LibertAIDAI/GLM-5.3-Flash-NVFP4`]: https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4
[`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`]: https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4
