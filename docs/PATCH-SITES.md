# The patch sites

Target: the vLLM tree inside `vllm/vllm-openai:glm53-flash-arm64-cu130`
(vLLM `0.1.dev20051+g487ecf187`, an NVIDIA one-off build), i.e. the tree that
MiaAI-Lab's `glm53-flash-sm121:v8` / `mm-ray-v1` images wrap. Paths are
relative to `/usr/local/lib/python3.12/dist-packages/vllm/`.

All sites live in `patches/tp3/apply_tp3_patch.py`, which matches on exact
source text rather than line numbers, so it survives minor upstream drift and
tells you loudly when it cannot find an anchor. The patcher also installs
`glm53_tp_pad.py` (the padding arithmetic, importable and unit-tested) into
`models/glm5next/`.

**Every site is a no-op at TP=1, TP=2 and TP=4**, because 64 heads, 2048
intermediate and 154880 vocab all divide those. The pad helpers return their
input unchanged when tp divides, and every pad-aware loader takes a
byte-equivalent stock path when nothing is padded.

---

## Why this patch looks nothing like the DSv4 one (and why that matters)

The sibling repo ([DeepSeek-V4-Flash-DSpark-3x-DGX-Spark]) fights a
*grouped* attention: DSv4's o_proj is a per-group batched matmul whose
`r = heads_per_group × head_dim` is a hard contract with the checkpoint, plus
per-head attention sinks whose pad lanes must be `-inf`, never 0. The whole
group-padding mechanism there exists because of those two facts.

GLM-5.3-Flash has **neither**:

* Its 11 `deepseek_sparse_attention` layers (+ the MTP layer) are plain MLA —
  `q_b_proj [16384, 1536]`, `kv_b_proj [32768, 512]`, `o_proj [4096, 16384]`
  = 64 heads × 256, no groups anywhere.
* There are **no sink tensors** — verified across all 113,074 names in the
  checkpoint index. Every padded lane in this model is plain **zero**.

So the constraint really is the textbook one — *TP must divide the head
count* — and the fix is the textbook fix: pad the head count, zero the pad
slabs. What GLM adds instead is **breadth**: a second head system (34 KDA
linear-attention layers), a vision tower, an MTP layer, and a
modelopt-NVFP4 expert path whose allocator makes zero-filling mandatory.

| | DSv4-Flash | GLM-5.3-Flash |
|---|---|---|
| Hard part | o_groups=8 BMM contract, `-inf` sinks | breadth: MLA + KDA + MoE + vocab + ViT |
| Pad value | 0 for weights, **-inf** for sinks | **0 everywhere** |
| Head pad | 64→72 via groups 8→9 | 64→66 directly (72 optional) |
| MoE intermediate | 2048→2112 | 2048→2112 (same site, same number) |
| Vocab | 129280→129408 | 154880→154944 (same mechanism) |
| Vision | n/a (text-only) | replicate via `--mm-encoder-tp-mode data`, no patch |

---

## Group 1 — the four gates (sites 1–4)

Stock vLLM refuses 64 % 3 in four independent places; all four must open.

### 1. `config/model.py` — `verify_with_parallel_config`

**The boot blocker.** Runs during *argument parsing* — `SpeculativeConfig`
triggers it for the MTP draft model before any model class exists, exactly
like DSv4 site 1. The patch allows the non-divisible case **only** for
`model_type` starting with `glm5_next` and only while `VLLM_GLM53_TP_PAD` is
enabled; every other arch is rejected with the stock message.

### 2. `models/glm5next/nvidia/model.py` — model-level assert

`assert config.num_attention_heads % world_size == 0` — re-pointed at the
padded head count, so it still guards (a padded count that *still* fails to
divide is a real bug and should die here).

### 3. `models/glm5next/nvidia/attention.py` — MLA head pad 64 → 66

The core MLA change, and deliberately minimal: pad `num_heads` at the top of
`Glm5NextMLAAttention.__init__` and let everything downstream derive from it
— `q_b_proj`/`kv_b_proj` output sizes, `o_proj` input size, the MLA wrapper's
`num_local_heads`, and the kv_b absorption split all follow automatically.

Why zero pads are exactly nothing here: a pad head's `q_b`/`kv_b` rows are
zero → its attention output is zero → its `o_proj` columns are zero, so the
all-reduce adds 0. MLA's KV cache stores the *headless* compressed latent
(`kv_lora_rank=512` per token), so pad heads do not even exist in the cache.

The **indexer is untouched**: `wq_b` is `ReplicatedLinear` and
`wk_weights_proj` is constructed with `disable_tp=True` ("no tensor
parallel, just replicated" — their comment). Its 32 heads never shard.
NVIDIA's own tree already zero-pads indexer *heads* to satisfy DeepGEMM's
{32, 64} head requirement — precedent for this exact technique, in-tree.

### 4. `models/glm5next/nvidia/kda.py` — KDA head pad 64 → 66

Same scheme for the 34 linear-attention layers. `num_heads` is padded before
the assert/`divide()` calls, and everything derives: the fused
`in_proj_qkvbfg_a` output sizes (q/k/v 8192→8448, b 64→66), `f_b`/`g_b`,
the three conv1d channel counts, `dt_bias`/`A_log` lane counts, `o_proj`
input, and — importantly — the **recurrent-state shape**
(`kda_state_shape(tp_size, num_heads, ...)` receives the padded count, so
the state cache and the projections stay consistent).

Why zero pads are exactly nothing here too: the delta-rule update is
`S += k^T (v − S k) β`; with k = v = 0 the state never leaves zero (fresh
sequences start from an explicitly zeroed state), and the pad heads' o_proj
columns are zero regardless. `dt_bias`/`A_log` pad lanes are zero-filled —
`softplus(0)` and `exp(0)` are finite, so no NaN can be manufactured.

The merged in_proj's **replicated** entries (f_a, g_a — shard ids 4 and 5,
loaded with tp_rank forced to 0) are never padded and never clamped: their
full-width slice always exists, so they take the stock path by construction.

---

## Group 2 — the two non-head dimensions (sites 5–6c)

### 5. `model_executor/layers/vocab_parallel_embedding.py` — vocab to lcm

`154880 % 3 != 0`. The image's `pad_vocab_size` uses a plain 64, so the
embedding shard dies **after** all three ranks have joined NCCL — which
reads as a distributed problem rather than a padding one (DSv4 site 5, same
mechanism, new number). `lcm(64, 3) = 192` → **154944**, which divides 3 and
stays 64-aligned. The embedding class already handles org < padded (masked
lookup; the stock loader zero-fills tail rows) and the logits processor
slices back to the org vocab, so only the granularity changes.

### 6a–6c. `models/glm5next/nvidia/model.py` — MoE intermediate 2048 → 2112

One padded value, computed once in `Glm5NextMoE.__init__`, feeds **both** the
routed experts (`FusedMoEFactory`) and the shared-expert MLP — they must
agree or the two paths shard differently. 2112 → 704/rank, and 704 is
16- and 64-aligned, so NVFP4 group-16 packing and marlin tiles stay legal.
Zero pad columns are nothing through SwiGLU (`silu(0) · 0 = 0`) and through
its clamped variant (`swiglu_limit=10` clamps 0 to 0). The dense MLP layers
(0–2, intermediate 12288) divide 3 and are untouched. The MTP layer builds
its MoE through this same class, so it is covered without its own site.

The 288 routed experts divide 3 — the expert-count assert that needed
scoping in DSv4 (site 7 there) never fires here.

---

## Group 3 — the loaders (sites 7a–11)

Sites 1–6 change *shapes*. These make a padded parameter actually
**loadable**: without them vLLM allocates the padded parameter, hands the
loader an unpadded checkpoint tensor, and hits a shape assert — or worse,
loads quietly wrong.

### 7a–7b. `fused_moe/routed_experts.py` — `_load_w13` / `_load_w2`

This build's expert loaders are already "pad-aware" — but for a different
padding. They compute

```python
loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
```

which is correct when the *parameter* is tile-padded while the *checkpoint*
width still divides tp (their MXFP4 case). At 2048 @ tp=3 the floor is 682:
it **drops checkpoint columns** (3 × 682 = 2046 ≠ 2048) and misaligns every
rank past 0 relative to the padded parameter layout. When the checkpoint
width does not divide tp, the offsets must come from the **padded param
shard width** instead (704). The patch adds exactly that branch — the
divisible case keeps the stock floor byte-for-byte — and zeroes a rank's
slab before the early "nothing to load" return.

### 7c. `fused_moe/routed_experts.py` — zero the pad tail

⛔ **The allocation trap, promoted from a DSv4 footnote to a hard
requirement.** The DSv4 deployment's MoE backend allocated expert tensors
with `torch.zeros`, so its narrow-and-skip loaders left pad tails at zero
*by accident of the allocator*. This checkpoint is **modelopt** NVFP4, and
the modelopt path allocates with `torch.empty` — 15+ occurrences, zero
`torch.zeros` (verified in this image). Worse, **marlin repacks the full
padded tensor after loading**, so allocation garbage would be repacked
straight into the live kernel input.

`_narrow_expert_data_for_padding` is the one funnel every expert tensor
(weights *and* group scales) passes through, so the patch zeroes the tail
there, before narrowing it away. Zero NVFP4 bytes decode to 0.0 and zero
e4m3 scale bytes are +0.0 — `0 × anything = 0` end to end.

### 8–10. `model_executor/parameter.py` — the three generic loaders

The direct port of DSv4 sites 11–13 — same file lineage, same three
functions, same remedy:

| # | Loader | Feeds (here) |
|---|---|---|
| 8 | `load_column_parallel_weight` | q_b/kv_b_proj, f_b/g_b, conv1d rows |
| 9 | `load_merged_column_weight` | KDA in_proj_qkvbfg_a, shared-expert gate_up |
| 10 | `load_row_parallel_weight` | every o_proj, shared-expert down_proj |

Each takes the byte-equivalent stock path when the full slice exists and
matches, and otherwise: zero-fill (torch.empty allocations again), then a
**dim-by-dim clamped copy** — `min(src, dst)` on *every* axis, not just the
sharded one. That dim-by-dim rule is inherited from a real DSv4 bug: padding
widens both the sharded dim (a rank's tail past the checkpoint end) *and*
non-sharded dims of sibling tensors, and a shard-dim-only clamp never
engages on the second kind. A `source exceeds destination` assert keeps a
genuine shape bug loud instead of silently clamp-loading it.

**The fourth loader is deliberately unpatched.** `load_qkv_weight` (keyed on
`shard_id`, not `tp_rank`) has the same structure and a different index;
this model does not exercise it, and an unexercised edit to generic loader
code is its own risk — the same call the DSv4 repo made, for the same
reason.

### 11. `model_executor/model_loader/weight_utils.py` — `sharded_weight_loader`

New relative to DSv4 (that model had no per-head 1-D parameters). KDA's
`dt_bias` [8192] and `A_log` [64] (reshaped 4-D) load through this axis
loader; at a padded head count the last rank's slice runs past the
checkpoint end. Clamp + zero-fill; zero is safe per site 4's argument.

---

## What needed NO patch (and how we know)

* **Indexer** — replicated (`ReplicatedLinear` / `disable_tp=True`), 32
  heads never shard. Read directly from the model source.
* **Router/gate + e_score_correction_bias** — replicated `GateLinear`.
* **Hyper-connection params** (`hc_attn_*`, `hc_ffn_*`, `mhc`) — per-layer
  stream-mixing vectors, replicated.
* **MTP glue** (`eh_proj` is a plain `nn.Linear` [4096, 8192], enorm/hnorm/
  shared_head.norm are RMSNorms) — replicated; the MTP block itself is a
  full `Glm5NextDecoderLayer`, so sites 3/4/6 cover it automatically.
* **Vision tower** — 16 heads, and 16 % 3 ≠ 0, but the model class inherits
  `supports_encoder_tp_data = True` from the GLM-4V family, so
  `--mm-encoder-tp-mode data` runs the whole tower **replicated** per rank
  (~0.9 GiB BF16). A config flag beats a ViT patch. The launcher passes it
  by default.
* **`fused_qkv_a_proj`** (MLA's merged q_a/kv_a) — projects to the lora
  ranks (1536 / 512), which are not head-dimensioned; nothing to pad.
* **KDA `o_norm`** — RMSNorm over head_dim (128), per-dim not per-head.
* **`fused_moe/config.py`'s divisibility assert** — left in place on
  purpose: site 6 pads *before* the divide, so the assert passes; if someone
  defeats site 6, it fails loudly, which is the correct behaviour.

---

## Group 4 — the sites the live boots found (12–14)

Static analysis produced sites 1–11 and a clean first boot *attempt*. Three
more gates only revealed themselves at runtime — kept separate here because
their lesson is different: **instance-level patches cannot cover
config-level static paths, and "the process died silently" has more than
one cause.**

### 12. `models/glm5next/nvidia/model.py` — the STATIC state-shape classmethod

The platform's hybrid block-size alignment calls
`Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)`
**before any layer exists**, and that classmethod feeds the *raw config*
head count straight into `kda_state_shape` → `divide(64, 3)` →
`AssertionError` on a worker rank at engine init. Sites 3/4 pad the layer
instances; they cannot reach a classmethod that derives shapes from the
config independently. The fix pads with the same helper — and it **must**
be the same helper: this classmethod's return sizes the recurrent-state
cache, and the padded layer's projections must agree with it exactly.

★ Generalizable: grep a model for `@classmethod` + `_from_config`
shape/dtype calculators before trusting an instance-level pad.

### 13–14. `model_executor/warmup/kernel_warmup.py` — the GB10 autotune killer

A worker rank died **natively** — no Python traceback, no stderr, no
kernel OOM, container still alive — minutes into init. MiaAI's own SM121
patch file documents the culprit for GB10 (*"fused_moe gemm1/gemm2
autotune kills rank 0 on GB10"*), but their skips live in the **unbaked
sm120 bundle**, so the serving image does not carry them. Site 13 skips
`flashinfer_sparse_mla_decode_autotune_warmup`; site 14 early-returns
`flashinfer_autotune` entirely. FlashInfer falls back to heuristics —
slower kernels, but alive.

### The fifth killer needs no patch: Ray's node-memory monitor

With all 14 sites applied, one rank *still* died silently at the same
phase. The real killer: on GB10, GPU allocations are backed by **unified
system memory**, so Ray's node-memory monitor counts the model's ~100 GiB
as node memory pressure and **SIGKILLs the vLLM actor at its default ~95%
threshold** — leaving exactly the same no-traceback corpse as a native
crash. The tell (once you know to look): the *worker's own Ray logs* say
`Node memory usage above threshold` / `Ray killed 1 worker(s)`, while
kernel dmesg shows no OOM. The launcher sets
`RAY_memory_usage_threshold=0.99` in every container and defaults
`GPU_MEM_UTIL=0.80`; see TROUBLESHOOTING for the signature.

★ Debugging lesson, stated plainly: a silent rank death had **three
different causes** in one bring-up (a Python assert relayed poorly, a
native library exit, and a supervisor SIGKILL). The diagnostic that
separated them was reading the **dying worker's own logs inside its
still-running container** (`/tmp/ray/session_latest/logs/worker-*.err`,
`raylet.out`) — never the head's log, which only ever says
"died unexpectedly."

## Choosing 66 vs 72

Default is **66** (22 heads/rank) — the minimal pad. The serving path here
is FlashInfer sparse-MLA SM90 + FA2 (per MiaAI's kernel layer), and neither
backend hard-codes a head-count table the way DSv4's SM120 decode dispatch
did. If a kernel ever objects to 22 local heads, set
`VLLM_GLM53_TP_PAD_ALIGN8=1` for 72 (24/rank, %8) — at ~12% extra attention
compute, which is roughly 2% of decode. Measure before keeping it.

---

## Verification built into the patcher

* **Per-site markers, versioned by a hash of the replacement body.** A
  changed patch body produces a different marker, so a *stale* patch is
  detected instead of being mistaken for a current one.
* **`MISS`, `STALE` and `AMBIGUOUS` are hard failures** (exit 1), never
  warnings. The launcher runs the patcher with `|| exit 1`, so a patch that
  cannot apply refuses the boot rather than serving quietly broken output.
* **`.tp3bak` backups** and `--revert` for every edited file — revert was
  verified byte-identical against a pristine copy of the image tree.
* **`--check`** reports every site's state without changing anything.

And outside the patcher:

* `glm53_tp_pad.py --test` asserts the real geometry (66/2112/154944, the
  per-rank real/pad splits, identity at TP=1/2/4, the kill-switch).
* `test_pad_loaders.py` exec()s the **shipped replacement bodies** against
  real tensors: every checkpoint element lands on exactly one rank exactly
  once, every pad lane is zero *even when the parameter starts as NaN*
  (the torch.empty trap, simulated), the stock path is taken when nothing
  is padded, and a padded TP=3 column+row pipeline and a padded TP=3
  SwiGLU expert both reproduce their TP=1 references numerically.

[DeepSeek-V4-Flash-DSpark-3x-DGX-Spark]: https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark
