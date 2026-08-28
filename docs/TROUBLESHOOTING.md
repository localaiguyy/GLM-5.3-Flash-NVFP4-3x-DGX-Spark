# Troubleshooting

The failures you are likely to meet, in the order you are likely to meet
them. The general rule for all of them: **read the first-dying rank's log**
— the other ranks report the consequence, not the cause.

---

## Boot refuses with "Total number of attention heads (64) must be divisible..."

The patch is not applied in the container that printed this (or
`VLLM_GLM53_TP_PAD=0` is set). Check inside the container, not on the host:

```bash
docker exec glm53-flash-head python3 /opt/glm53-tp3/apply_tp3_patch.py --check
```

If `/opt/glm53-tp3` does not exist, the mount is missing — the launcher
mounts `patches/tp3/` there on every node and rsyncs it to the workers.

## The patcher reports MISS or STALE

`MISS`: the vLLM tree in your image does not contain the anchor text — a
different image build. Do not force it; diff the target file against the
anchors in `apply_tp3_patch.py` and re-anchor deliberately. The patch was
written against `vllm/vllm-openai:glm53-flash-arm64-cu130`
(vLLM `0.1.dev20051+g487ecf187`) as wrapped by `glm53-flash-sm121:v8`.

`STALE`: a *previous version* of a site is applied in this container's
writable layer. Destroy the containers (`./start.sh stop`) and start fresh —
`docker restart` keeps the old edits forever.

## One rank dies at a weight-shape assert; the others report NCCL/Ray peer loss

The signature of a patch-version mismatch between nodes (one worker got an
old `tp3/` sync) — or of running a checkpoint whose geometry differs from
GLM-5.3-Flash. The launcher refuses to boot when the patch rsync fails;
if you bypassed it, don't. Verify per-node:

```bash
for h in head worker1 worker2; do
  docker exec glm53-flash-$h python3 /opt/glm53-tp3/apply_tp3_patch.py --check
done
```

## ncclCommInitRank busy-waits forever (~100% CPU, ~0.7 GiB GPU, no /health)

NCCL cannot use IP aliases — the CX7 interface + RDMA device pins are wrong
for one of the nodes. Set `HEAD_CX7_IF/IB` and `WORKER<i>_CX7_IF/IB` to what
`ip -br link` / `ibv_devices` show **on each node**, and check
`NCCL_IB_GID_INDEX` against `show_gids` (RoCEv2 entries). This is inherited
from the 2-node recipe; the third node just adds one more place to get it
wrong. `NCCL_DEBUG=INFO ./start.sh` prints which device each rank bound.

## Engine init finishes, then a rank dies during warmup/init

UMA discipline is stricter at TP=3 because GB10 GPU allocations are backed by
unified system memory and Ray's node-memory monitor counts that pressure. Keep
`GPU_MEM_UTIL=0.80`, `RAY_memory_usage_threshold=0.99`, `--skip-mm-profiling`
on, and the 4 GiB Ray object store. The silent-failure signature is a live
worker container with Ray logs saying `Node memory usage above threshold` /
`Ray killed 1 worker(s)`, while kernel dmesg has no OOM. That means Ray killed
the vLLM actor, not Linux. Also check `docker ps` on every node; no other model
may run on any of the three.

## Multimodal requests fail with a head/shape error in the vision tower

The launcher must pass `--mm-encoder-tp-mode data` (default here). Without
it the ViT's 16 heads are sharded across 3 ranks and die. If your vLLM build
logs "model does not support mm-encoder-tp-mode data", it predates the
GLM-4V encoder-DP support — patch level mismatch, see MISS above.

## The model answers "I can't see images" to every image

Not a TP problem — a **chat-template** problem, and it is a known trap with
derivative checkpoints: a quantizer snapshots its base repo at some commit,
and if that snapshot predates the template fix, the stale file is what they
ship — *and re-downloading from the derivative repo fetches it again*. The
launcher's refresh refuses a template without `emit_image`; point
`CHAT_TEMPLATE_URL` at the **base** model:

```bash
CHAT_TEMPLATE_URL=https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/chat_template.jinja ./start.sh
```

Quick check on any template: `grep -c emit_image chat_template.jinja`
(0 = stale).

For an already-refreshed production node that must restart without internet,
set `SKIP_TEMPLATE_REFRESH=1`; the launcher still verifies that the cached
template contains `emit_image` before booting.

## BF16 modules fail to load as NVFP4 (KeyError/shape error naming weight_scale)

Derivative checkpoints sometimes ship a **shorter**
`quantization_config.ignore` list than the reference LibertAIDAI checkpoint
— missing the vLLM-side *fused* module names (`fused_qkvbfg_a_proj`,
`qkv_proj`, `fused_fg_b_proj`, the conv1ds, `o_norm`). vLLM matches ignore
globs against its own module names, so a missing fused glob can make the
loader treat a BF16 module as quantized. The launcher preflight warns about
this. Fix: overlay the reference list into your local snapshot's
`config.json` (`quantization_config.ignore` only — nothing else differs; the
tensor sets of the two checkpoints are name-identical).

## Decode stops but /v1/models still answers

The wedge. A 200 on `/v1/models` is not a decode test — the head answers it
from local state after every worker is gone. `scripts/glm53-decode-watchdog`
probes with a real completion, requires two consecutive failures, then
captures py-spy stacks from **every** rank plus the GPU spin signature
(high util + idle power + 0% mem util) *before* alerting, and leaves the
cluster running — the live process holds the only copy of that state.

When reading the stacks: under a jammed device queue every host thread
freezes at whatever kernel launch it issues *next*, so the kernels named in
py-spy are the jam's **victims**, not its cause. Cross-rank stacks in
several *different* kernels is a livelock's expected presentation, not
evidence against it. Compare stacks across ranks; never convict a kernel on
one rank's dump.

## Proving the pad slabs on your own build (paranoia mode, recommended once)

The pad rows are supposed to be zero because the loaders zero-fill them. To
*prove* it on a live build rather than trust it, use the NaN-canary: before
weight loading, fill a padded parameter with NaN (e.g. a one-off edit in the
loader, or `param.data.fill_(float("nan"))` at allocation), boot, and dump
the pad region:

```python
# inside the head container, after load, e.g. via a debugger or a one-off print
p = model.model.layers[3].self_attn.q_b_proj.weight   # rank 2 holds the pad
print(p[-2 * 256 :].abs().max())    # last 2 pad heads x qk_head_dim -> must be 0.0
```

An un-zeroed tail cannot hide from that — NaN survives everything except an
explicit overwrite. The loader unit tests (`test_pad_loaders.py`) run this
exact scenario on synthetic tensors; the on-cluster version is the same idea
with the real 181 GiB.

## After changing anything: validate, don't vibe-check

```bash
scripts/validate_tp3.sh <head-ip>:8888
```

Fluency survives a broken shard; precise recall, arithmetic, sequential
state and the vision path do not. All sections must pass; a multimodal skip is
acceptable only when the serving profile intentionally disables image input.
