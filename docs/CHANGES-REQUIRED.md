# Every change required to run GLM-5.3-Flash-NVFP4 at TP=3

This is the complete list. If you have a working 2-node deployment of
MiaAI-Lab's recipe, these are the differences — nothing else was needed, and
nothing here is optional.

There are four groups of changes:

1. [The vLLM patch](#1-the-vllm-patch) — applied inside every container
2. [The image](#2-the-image-unchanged) — **unchanged**; build it from MiaAI's repo
3. [The launcher](#3-the-launcher) — the stock script assumes exactly one worker
4. [Operational rules](#4-operational-rules-you-will-regret-skipping)

---

## 1. The vLLM patch

All sites live in one idempotent script: `patches/tp3/apply_tp3_patch.py`
(plus `glm53_tp_pad.py`, the arithmetic module it installs into the tree).
It runs **inside each container at startup**, edits the installed vLLM tree,
and exits non-zero if anything is wrong — so a bad patch refuses the boot
instead of serving quietly broken output.

```bash
python3 apply_tp3_patch.py            # apply
python3 apply_tp3_patch.py --check    # report, change nothing
python3 apply_tp3_patch.py --revert   # restore from .tp3bak
```

**Every site is a no-op at TP=1/2/4** (64 heads, 2048 intermediate and
154880 vocab all divide those), so applying this to a 2-node deployment
changes nothing. That property is asserted in the pad module's tests, not
assumed.

| Site | File | What | Why |
|---|---|---|---|
| 1 | `config/model.py` | allow non-divisible heads for `glm5_next` | **The boot blocker** — fires during argument parsing (SpeculativeConfig hits it for the MTP draft), before any model code runs |
| 2 | `glm5next/nvidia/model.py` | model assert on the *padded* count | second gate |
| 3 | `glm5next/nvidia/attention.py` | **MLA heads 64 → 66** | q_b/kv_b/o_proj, the MLA wrapper and the kv_b absorption all derive from it |
| 4 | `glm5next/nvidia/kda.py` | **KDA heads 64 → 66** | projections, conv1d channels, dt_bias/A_log lanes and the recurrent-state shape all derive from it |
| 5 | `vocab_parallel_embedding.py` | vocab pad to `lcm(64, tp)` → 154944 | 154880 % 3 ≠ 0 dies *after* NCCL join — reads as a network fault |
| 6a–c | `glm5next/nvidia/model.py` | MoE intermediate 2048 → 2112 | one padded value for routed **and** shared experts (704/rank, 64-aligned) |
| 7a–b | `fused_moe/routed_experts.py` | shard math for a non-divisible checkpoint | the stock floor (2048//3=682) drops 2 columns per expert and misaligns ranks 1–2 |
| 7c | `fused_moe/routed_experts.py` | **zero pad tails** | modelopt allocates `torch.empty`; marlin repacks the FULL padded tensor |
| 8–10 | `parameter.py` | pad-aware column / merged / row loaders | dim-by-dim clamp + zero-fill; byte-equivalent stock path when nothing is padded |
| 11 | `weight_utils.py` | pad-aware `sharded_weight_loader` | KDA per-head 1-D params (dt_bias, A_log) |
| 12 | `glm5next/nvidia/model.py` | pad heads in the STATIC `get_mamba_state_shape_from_config` | the platform calls it BEFORE any layer exists; must agree exactly with the padded layers |
| 13 | `kernel_warmup.py` | skip sparse-MLA decode autotune | GB10 rank killer (MiaAI-documented, unbaked in the image) |
| 14 | `kernel_warmup.py` | early-return `flashinfer_autotune` | same killer's second half; heuristic kernels — slower, alive |

Full rationale per site, and what needed *no* patch: `docs/PATCH-SITES.md`.

Before trusting any of it on your own tree:

```bash
python3 patches/tp3/glm53_tp_pad.py --test      # the arithmetic
python3 patches/tp3/test_pad_loaders.py         # the shipped loader bodies, on real tensors
VLLM_ROOT=/path/to/vllm python3 patches/tp3/apply_tp3_patch.py --check
```

## 2. The image (unchanged)

The kernel layer (`glm53-flash-sm121:v8` — FlashInfer 0.6.18 pin, SM90
sparse-MLA + FA2 gated onto GB10, PDL off, the indexer fixes) and the
serving layer (`mm-ray-v1`) are MiaAI's and are used exactly as they ship.
**Nothing in the kernel layer is TP-dependent.** Build both from their repo
(`files/build.sh`); this repo's launcher will call it if the tag is missing
(clone their repo next to this one, or set `MIA_REPO_DIR`).

## 3. The launcher

MiaAI's `start.sh` assumes **exactly one worker** — the worker's ssh target,
IP, home, CX7 pins and NCCL dir are all singletons, referenced at ~20 call
sites. Rather than sed-transforming an 810-line script (the approach the
DSv4 repo could take, because its stock launcher was a small compose+env
affair), this repo ships a **derivative launcher** (MIT, attributed) that
drives N workers:

* one `WORKER<i>_*` block per worker (`config/env.glm53.tp3.example`)
* preflight, image ship (`docker save | ssh docker load`), weight rsync,
  chat-template sync, container launch, stop/status/logs — all loop over
  workers
* the head waits for `CLUSTER_SIZE = TP` Ray nodes (already generic upstream)
* **the patch is mounted read-only into every container and applied before
  anything imports vllm** — head inner script and worker inner script both
  run `apply_tp3_patch.py || exit 1`
* the patch dir is rsynced to every worker with **no `|| true`**: a failed
  sync leaves a worker running an *old* patch, and the only symptom is a
  shape assert on one rank — or silently wrong output. A safeguard that
  cannot report its own death is not a safeguard.
* `--mm-encoder-tp-mode data` is passed by default: the vision tower has 16
  heads, 16 % 3 ≠ 0, and replicating it (~0.9 GiB/rank) beats patching a ViT
* `CHAT_TEMPLATE_URL` is overridable, and the refresh **refuses** a template
  without `emit_image` — derivative checkpoints (abliterations) snapshot
  their base repo and can ship a stale, multimodal-blind template. Point it
  at the base model's template for those.
* a preflight warns when a derivative checkpoint's
  `quantization_config.ignore` lacks the fused-module globs the reference
  checkpoint carries (see TROUBLESHOOTING)

Diff it against MiaAI's to audit: the serving profile, NCCL pins, UMA
discipline and inner-script structure are theirs, kept verbatim wherever a
loop over workers did not force a change.

## 4. Operational rules you will regret skipping

### Destroy the containers when you change the patch

`docker rm -f`, not restart. The patch is applied *inside* the container, so
its edits live in that container's writable layer and survive every
stop/start. Change the patch, restart, and the patcher sees its own previous
marker and reports `applied` — permanently, because the stock text it would
replace is already gone. (The launcher's `stop`/`restart` already
`docker rm -f` on every node; keep it that way.) The marker hash catches a
*changed* patch (`STALE`, hard failure) — it cannot catch a stale *container*
whose marker matches an old file you since edited locally. When in doubt:
`./start.sh stop && ./start.sh`.

### Verify the patch inside the container, not on the host

A host-side checksum proves only what the host has. The container imports
from its own layer:

```bash
docker exec glm53-flash-head python3 /opt/glm53-tp3/apply_tp3_patch.py --check
```

### Read the log of the rank that died first

When one rank exits, the other two report the *consequence* (a Ray actor
death, an NCCL timeout naming a peer IP), not the cause. The launcher
collects every rank's log on failure — read the one whose timestamps end
first. It looks like a network fault and usually is not one.

### Scope every log grep with a time bound

The launcher recreates containers, but Ray session logs inside a reused
cache volume persist. `--since 30m` on `docker logs`, and compare against
`date -u`, or a previous attempt's error reads as current.

### "It starts" is not "it is correct"

A wrong head shard, a non-zero pad slab or a dropped expert column produces
fluent, confident, wrong output with nothing in the logs. Run
`scripts/validate_tp3.sh <head>:8888` after every boot — factual recall,
arithmetic, sequential state (the KDA layers), a needle across ~1.5k tokens,
a degeneration check, and a real image through the vision path. All sections
must pass before the deployment is trusted.
