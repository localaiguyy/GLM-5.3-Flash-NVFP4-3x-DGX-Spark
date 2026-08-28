#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Apply the GLM-5.3-Flash TP=3 padding patch to an installed vLLM tree.

Target: the ``vllm/vllm-openai:glm53-flash-arm64-cu130`` tree as shipped in
MiaAI-Lab's ``glm53-flash-sm121:v8`` / ``mia/glm53-flash-spark:mm-ray-v1``
images (vLLM ``0.1.dev20051+g487ecf187``). Anchors match on exact source
text, not line numbers, so minor drift is survivable and real drift is LOUD.

Design rules (inherited from the DSv4 TP=3 patcher, kept on purpose):

  * idempotent — a marker comment versioned by a hash of the replacement
    body is left at every site. Re-running skips applied sites; a marker
    with a DIFFERENT hash is a STALE patch and a hard failure, because the
    stock text it would replace is already gone.
  * MISS and STALE are hard failures (exit 1), never warnings. The launcher
    runs this with ``|| exit 1`` so a patch that cannot apply refuses the
    boot instead of serving quietly broken output.
  * ``.tp3bak`` backups and ``--revert`` for every edited file.
  * every site is a NO-OP at TP=1/2/4 — the pad helpers return their input
    unchanged when tp already divides, and the pad-aware loaders take the
    byte-equivalent stock path when nothing is padded.

Usage:
    python3 apply_tp3_patch.py            # apply (default)
    python3 apply_tp3_patch.py --check    # report, change nothing
    python3 apply_tp3_patch.py --revert   # restore every .tp3bak

Env:
    VLLM_ROOT          override the vllm tree root
                       (default /usr/local/lib/python3.12/dist-packages/vllm)
    VLLM_GLM53_TP_PAD  runtime kill-switch read by the PATCHED code
                       (default 1; 0 restores stock refusal behaviour)

See docs/PATCH-SITES.md for what every site does and why it exists.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))
HERE = Path(__file__).resolve().parent
LOG = "[glm53-tp3]"

MARK = "GLM53-TP3"


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# The sites. Each is (relpath, label, anchor, replacement, marker_indent).
# The marker comment is inserted as the first line of the replacement at
# marker_indent spaces; the hash covers the replacement body WITHOUT the
# marker so cosmetic marker changes cannot silently re-apply a site.
# ---------------------------------------------------------------------------

SITES: list[tuple[str, str, str, str, int]] = []


def site(relpath: str, label: str, anchor: str, replacement: str, indent: int) -> None:
    SITES.append((relpath, label, anchor, replacement, indent))


# --- 1. config/model.py — the boot blocker -------------------------------
# verify_with_parallel_config runs during ARGUMENT PARSING (SpeculativeConfig
# triggers it for the MTP draft model before any model class exists). Without
# this you never reach the model code, exactly like DSv4 site 1.

site(
    "config/model.py",
    "config: allow non-divisible heads for glm5_next",
    """\
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )
""",
    """\
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            # glm5_next pads its head count inside the model code (see
            # vllm/models/glm5next/glm53_tp_pad.py), so a non-divisible TP
            # size is legal for that arch while the pad is enabled. This
            # check runs during argument parsing — SpeculativeConfig hits it
            # for the MTP draft too — so it must be relaxed HERE, not in the
            # model. Every other arch is rejected exactly as stock.
            import os as _g53_os

            _g53_ok = _g53_os.environ.get("VLLM_GLM53_TP_PAD", "1") not in (
                "",
                "0",
                "false",
                "False",
            ) and str(
                getattr(getattr(self, "hf_config", None), "model_type", "")
            ).startswith("glm5_next")
            if not _g53_ok:
                raise ValueError(
                    f"Total number of attention heads ({total_num_attention_heads})"
                    " must be divisible by tensor parallel size "
                    f"({tensor_parallel_size})."
                )
""",
    8,
)

# --- 2. glm5next model-level assert ---------------------------------------

site(
    "models/glm5next/nvidia/model.py",
    "model: assert on the PADDED head count",
    """\
        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )
""",
    """\
        world_size = get_tensor_model_parallel_world_size()
        from vllm.models.glm5next.glm53_tp_pad import maybe_pad_heads as _g53_ph

        assert _g53_ph(config.num_attention_heads, world_size) % world_size == 0, (
            "num_attention_heads must be divisible by world_size "
            "(even after the GLM53 TP pad)"
        )
""",
    8,
)

# --- 3. MLA head pad ------------------------------------------------------
# Everything downstream (q_b_proj/kv_b_proj/o_proj sizes, the MLA wrapper's
# num_local_heads, the kv_b absorption split) derives from self.num_heads, so
# padding it HERE keeps the whole layer self-consistent. The pad heads carry
# zero q_b/kv_b rows and zero o_proj columns (pad-aware loaders), so:
# zero q/k -> uniform softmax over zero v -> zero head output -> annihilated
# by the zero o_proj columns. No sinks exist in this model (verified across
# all 113,074 checkpoint tensors), so there is no -inf subtlety like DSv4's.

site(
    "models/glm5next/nvidia/attention.py",
    "MLA: pad heads 64 -> 66 (or 72 with ALIGN8)",
    """\
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
""",
    """\
        tp_size = get_tensor_model_parallel_world_size()
        from vllm.models.glm5next.glm53_tp_pad import (
            describe_pad as _g53_describe,
            maybe_pad_heads as _g53_ph,
        )

        self.num_heads_real = num_heads
        num_heads = _g53_ph(num_heads, tp_size)
        if num_heads != self.num_heads_real:
            print(_g53_describe("mla-heads", self.num_heads_real, num_heads, tp_size))
        self.num_heads = num_heads
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
""",
    8,
)

# --- 4. KDA head pad ------------------------------------------------------
# Same scheme for the 34 linear-attention layers: projections, conv1d
# channels, dt_bias/A_log lanes and the recurrent-state shape all derive
# from self.num_heads. A pad head's q/k/v rows are zero, so its delta-rule
# state never leaves zero (fresh sequences start from an explicit zero
# state), and its o_proj columns are zero regardless. dt_bias/A_log pad
# lanes are zero-filled: softplus(0) and exp(0) are finite.

site(
    "models/glm5next/nvidia/kda.py",
    "KDA: pad heads 64 -> 66 (or 72 with ALIGN8)",
    """\
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.conv_size = conv_size
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
""",
    """\
        self.head_dim = head_dim
        from vllm.models.glm5next.glm53_tp_pad import (
            describe_pad as _g53_describe,
            maybe_pad_heads as _g53_ph,
        )

        self.num_heads_real = num_heads
        num_heads = _g53_ph(num_heads, self.tp_size)
        if num_heads != self.num_heads_real:
            print(
                _g53_describe("kda-heads", self.num_heads_real, num_heads, self.tp_size)
            )
        self.num_heads = num_heads
        self.conv_size = conv_size
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
""",
    8,
)

# --- 5. vocab pad to lcm(64, tp) ------------------------------------------
# 154880 % 3 != 0. With the plain 64 padding the embedding shard dies AFTER
# all ranks have joined NCCL — which reads as a distributed fault rather
# than a padding one (DSv4 site 5, same mechanism, new number: 154944).
# The embedding class already handles org_vocab < padded vocab (masked
# lookup; the stock loader zero-fills the tail rows), and the logits
# processor slices back to the org vocab, so ONLY the padding granularity
# needs to change.

site(
    "model_executor/layers/vocab_parallel_embedding.py",
    "vocab: pad to lcm(padding, tp)",
    '''\
def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to
''',
    '''\
def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value (and to a multiple of tp)."""
    try:
        from vllm.models.glm5next.glm53_tp_pad import tp_pad_enabled as _g53e

        if _g53e():
            import math as _g53_math

            from vllm.distributed import get_tensor_model_parallel_world_size

            pad_to = _g53_math.lcm(pad_to, get_tensor_model_parallel_world_size())
    except Exception:
        # Outside a distributed context (or with the pad module absent) the
        # stock granularity applies; any resulting non-divisibility still
        # fails loudly downstream.
        pass
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to
''',
    0,
)

# --- 6. MoE intermediate pad (routed + shared) ----------------------------
# 2048 % 3 != 0 -> 2112 (704/rank; 64-aligned keeps NVFP4 group-16 packing
# and marlin tiles legal). One padded value feeds BOTH the FusedMoEFactory
# (routed experts) and the shared-expert MLP, so the two stay consistent.

site(
    "models/glm5next/nvidia/model.py",
    "MoE: compute the padded intermediate once",
    """\
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
""",
    """\
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        from vllm.models.glm5next.glm53_tp_pad import (
            maybe_pad_multiple as _g53_pad_multiple,
        )

        self._g53_moe_intermediate = _g53_pad_multiple(
            config.moe_intermediate_size, self.tp_size
        )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
""",
    8,
)

site(
    "models/glm5next/nvidia/model.py",
    "MoE: shared expert uses the padded intermediate",
    """\
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
""",
    """\
            intermediate_size = self._g53_moe_intermediate * config.n_shared_experts
""",
    12,
)

site(
    "models/glm5next/nvidia/model.py",
    "MoE: routed experts use the padded intermediate",
    """\
            intermediate_size=config.moe_intermediate_size,
""",
    """\
            intermediate_size=self._g53_moe_intermediate,
""",
    12,
)

# --- 7. routed-experts loaders: non-divisible shard math + zero tails -----
# The stock loaders compute loaded_per_rank = checkpoint_width // tp_size.
# That floor is correct for the case they were written for (a param padded
# for kernel tiles while the CHECKPOINT width still divides tp), but at
# 2048 @ tp=3 it is 682: it drops checkpoint columns (3 x 682 = 2046 != 2048)
# AND misaligns every rank past 0 relative to the padded parameter layout.
# When the checkpoint width does not divide tp, the offsets must come from
# the PADDED param shard width instead. Both branches clamp; both zero the
# never-written tail, because the modelopt-NVFP4 path allocates expert
# weights with torch.empty (verified: 15+ torch.empty, zero torch.zeros)
# and marlin repacks the FULL padded tensor after loading.

site(
    "model_executor/layers/fused_moe/routed_experts.py",
    "experts: w13 shard math for a non-divisible checkpoint",
    """\
            # When the parameter has been padded (e.g. MXFP4 rounding up
            # intermediate_size_per_partition), shard_size is the padded
            # size.  Compute the offset into the checkpoint weight using
            # the *unpadded* per-rank size so that every TP rank lands at
            # the correct slice.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
""",
    """\
            # When the parameter has been padded (e.g. MXFP4 rounding up
            # intermediate_size_per_partition), shard_size is the padded
            # size.  Compute the offset into the checkpoint weight using
            # the *unpadded* per-rank size so that every TP rank lands at
            # the correct slice.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            if loaded_weight.shape[shard_dim] % tp_size != 0:
                # The checkpoint width itself does not divide tp (2048 @
                # tp=3): the parameter was padded GLOBALLY, so every rank's
                # slice starts at rank * <padded shard width> in checkpoint
                # units. shard_size is that width, in the same (possibly
                # packed) units as this checkpoint tensor.
                loaded_per_rank = shard_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # No checkpoint data for this rank's slab: ZERO it instead
                # of leaving allocation garbage, then skip the copy.
                if shard_id == "w1":
                    expert_data.narrow(shard_dim, 0, shard_size).zero_()
                else:
                    expert_data.narrow(shard_dim, shard_size, shard_size).zero_()
                return
""",
    12,
)

site(
    "model_executor/layers/fused_moe/routed_experts.py",
    "experts: w2 shard math for a non-divisible checkpoint",
    """\
            # Same padding fix as _load_w13: use unpadded per-rank size.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
""",
    """\
            # Same padding fix as _load_w13: use unpadded per-rank size.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            if loaded_weight.shape[shard_dim] % tp_size != 0:
                # Non-divisible checkpoint width (see _load_w13): offset by
                # the padded param width. For w2 the param IS the shard.
                loaded_per_rank = expert_data.shape[shard_dim]
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # No checkpoint data for this rank's slab: zero it all.
                expert_data.zero_()
                return
""",
    12,
)

site(
    "model_executor/layers/fused_moe/routed_experts.py",
    "experts: zero the pad tail before narrowing it away",
    """\
        dims = (hidden_dim,) if shard_dim is None else (hidden_dim, shard_dim)
        if loaded_weight.ndim > 0:
            for dim in dims:
                if (
                    0 <= dim < expert_data.ndim
                    and dim < loaded_weight.ndim
                    and expert_data.shape[dim] > loaded_weight.shape[dim]
                ):
                    expert_data = expert_data.narrow(dim, 0, loaded_weight.shape[dim])
        return expert_data
""",
    """\
        dims = (hidden_dim,) if shard_dim is None else (hidden_dim, shard_dim)
        if loaded_weight.ndim > 0:
            for dim in dims:
                if (
                    0 <= dim < expert_data.ndim
                    and dim < loaded_weight.ndim
                    and expert_data.shape[dim] > loaded_weight.shape[dim]
                ):
                    # Zero the tail the copy will never write. The modelopt
                    # path allocates with torch.empty, and marlin repacks the
                    # FULL padded tensor after loading — garbage here would
                    # repack straight into the live kernel input. Zero-valued
                    # NVFP4 bytes decode to 0.0 and zero e4m3 scale bytes are
                    # +0.0, so 0 * anything = 0 through the whole pipeline.
                    expert_data.narrow(
                        dim,
                        loaded_weight.shape[dim],
                        expert_data.shape[dim] - loaded_weight.shape[dim],
                    ).zero_()
                    expert_data = expert_data.narrow(dim, 0, loaded_weight.shape[dim])
        return expert_data
""",
    8,
)

# --- 8-10. generic parameter loaders (DSv4 sites 11-13, ported) -----------
# Sites 1-7 change SHAPES. These make a padded parameter actually LOADABLE:
# without them vLLM allocates the padded parameter, hands the loader an
# unpadded checkpoint tensor, and dies on `assert shapes equal` (or worse,
# doesn't). All three: byte-equivalent stock path when nothing is padded;
# otherwise zero-fill + a dim-by-dim clamped copy. The clamp is dim-by-dim
# on purpose — padding widens BOTH sharded dims (a rank's tail past the
# checkpoint end) and non-sharded dims (a sibling tensor sized from the
# padded head count), and handling only the obvious one was a real DSv4 bug.
# The fourth loader (load_qkv_weight, keyed on shard_id not tp_rank) is
# DELIBERATELY not patched: this model never exercises it, and an
# unexercised edit to generic loader code is its own risk.

site(
    "model_executor/parameter.py",
    "param: pad-aware load_column_parallel_weight",
    """\
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)
""",
    """\
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        start = self.tp_rank * shard_size
        avail = loaded_weight.shape[self.output_dim] - start
        if avail >= shard_size:
            candidate = loaded_weight.narrow(self.output_dim, start, shard_size)
            if candidate.shape == self.data.shape:
                self.data.copy_(candidate)
                return
        # Padded parameter: zero-fill (allocations may be torch.empty), then
        # copy the checkpoint-backed region with a dim-by-dim clamp.
        self.data.zero_()
        take = min(shard_size, max(avail, 0))
        if take > 0:
            src = loaded_weight.narrow(self.output_dim, start, take)
            dst = self.data
            assert all(
                d >= s for d, s in zip(dst.shape, src.shape)
            ), f"GLM53 pad loader: source {tuple(src.shape)} exceeds dest {tuple(dst.shape)}"
            for i in range(dst.dim()):
                b = min(dst.shape[i], src.shape[i])
                dst = dst.narrow(i, 0, b)
                src = src.narrow(i, 0, b)
            dst.copy_(src)
""",
    4,
)

site(
    "model_executor/parameter.py",
    "param: pad-aware load_merged_column_weight",
    """\
        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
""",
    """\
        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        start = self.tp_rank * shard_size
        avail = loaded_weight.shape[self.output_dim] - start
        if avail >= shard_size:
            candidate = loaded_weight.narrow(self.output_dim, start, shard_size)
            if candidate.shape == param_data.shape:
                param_data.copy_(candidate)
                return
        # Padded merged shard (KDA in_proj_qkvbfg_a q/k/v/b entries, the
        # shared-expert gate_up, ...): zero-fill this slab, clamped copy.
        # Replicated entries (KDA f_a/g_a) never reach here — their forced
        # tp_rank=0 slice always exists in full.
        param_data.zero_()
        take = min(shard_size, max(avail, 0))
        if take > 0:
            src = loaded_weight.narrow(self.output_dim, start, take)
            dst = param_data
            assert all(
                d >= s for d, s in zip(dst.shape, src.shape)
            ), f"GLM53 pad loader: source {tuple(src.shape)} exceeds dest {tuple(dst.shape)}"
            for i in range(dst.dim()):
                b = min(dst.shape[i], src.shape[i])
                dst = dst.narrow(i, 0, b)
                src = src.narrow(i, 0, b)
            dst.copy_(src)
""",
    4,
)

site(
    "model_executor/parameter.py",
    "param: pad-aware load_row_parallel_weight",
    """\
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)
""",
    """\
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.input_dim]
        start = self.tp_rank * shard_size
        avail = loaded_weight.shape[self.input_dim] - start
        if avail >= shard_size:
            candidate = loaded_weight.narrow(self.input_dim, start, shard_size)
            if len(candidate.shape) == 0:
                candidate = candidate.reshape(1)
            if candidate.shape == self.data.shape:
                self.data.copy_(candidate)
                return
        # Padded row shard (o_proj input columns past the checkpoint end on
        # the last rank): zero-fill, clamped copy. The zero columns are what
        # annihilate the pad heads' (zero) outputs in the all-reduce.
        self.data.zero_()
        take = min(shard_size, max(avail, 0))
        if take > 0:
            src = loaded_weight.narrow(self.input_dim, start, take)
            if len(src.shape) == 0:
                src = src.reshape(1)
            dst = self.data
            assert all(
                d >= s for d, s in zip(dst.shape, src.shape)
            ), f"GLM53 pad loader: source {tuple(src.shape)} exceeds dest {tuple(dst.shape)}"
            for i in range(dst.dim()):
                b = min(dst.shape[i], src.shape[i])
                dst = dst.narrow(i, 0, b)
                src = src.narrow(i, 0, b)
            dst.copy_(src)
""",
    4,
)

# --- 11b. the STATIC state-shape classmethod ------------------------------
# The platform's hybrid block-size alignment calls
# Glm5NextForCausalLM.get_mamba_state_shape_from_config() BEFORE any layer
# exists, and that classmethod feeds the RAW config head count straight into
# kda_state_shape -> divide(64, 3) -> AssertionError on every worker rank.
# (Found the honest way: it killed the first TP=3 boot at engine init.)
# It must return exactly the shape the padded layer instance computes, or the
# state cache and the projections disagree — so pad with the same helper.

site(
    "models/glm5next/nvidia/model.py",
    "model: pad heads in get_mamba_state_shape_from_config",
    """\
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )
""",
    """\
        from vllm.models.glm5next.glm53_tp_pad import maybe_pad_heads as _g53_ph

        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            _g53_ph(hf_config.linear_num_heads, tp_size),
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )
""",
    8,
)

# --- 13-14. FlashInfer autotune kills a rank on GB10 ----------------------
# Observed live at TP=3: a worker rank dies NATIVELY (no traceback, container
# survives) at the exact second "[Autotuner]: Autotuning process ends" is
# logged, taking engine init down with "RayWorkerProc rank=[N] died
# unexpectedly". This is the SAME killer MiaAI-Lab documents in their SM121
# patch file ("fused_moe gemm1/gemm2 autotune kills rank 0 on GB10") — their
# skips live in the unbaked sm120 bundle, so the serving image does not carry
# them. These two sites are their remedies, applied persistently. Not
# TP-related, but required to boot this image on GB10 at all under shapes
# that reach the autotuner.

site(
    "model_executor/warmup/kernel_warmup.py",
    "warmup: skip sparse-MLA decode autotune (GB10 rank killer)",
    """\
    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)
""",
    """\
    # Skip FlashInfer sparse-MLA decode autotune — kills a rank on GB10
    # (MiaAI-Lab's GLM53_SKIP_FI_SPARSE_WARMUP, applied persistently).
    deepseek_v4_sparse_mla_attention_warmup(worker)
""",
    4,
)

site(
    "model_executor/warmup/kernel_warmup.py",
    "warmup: skip FlashInfer autotune entirely (GB10 rank killer)",
    """\
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group
""",
    """\
    # fused_moe gemm1/gemm2 autotune kills a rank on GB10 (MiaAI-Lab's
    # GLM53_SKIP_FI_AUTOTUNE, applied persistently). FlashInfer falls back
    # to heuristics — slower kernels, but alive.
    logger.info_once("Skipping FlashInfer autotune on SM121 (GB10)")
    return
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group
""",
    4,
)

# --- 11. 1-D per-head params (dt_bias, A_log) -----------------------------
# The KDA layer loads dt_bias [8192] and A_log [64] through
# sharded_weight_loader, which narrows by rank * param_shard — at a padded
# head count the last rank's slice runs past the checkpoint end. Zero is the
# correct pad for both (softplus(0) and exp(0) are finite, and the pad
# heads' outputs are annihilated by their zero o_proj columns regardless).

site(
    "model_executor/model_loader/weight_utils.py",
    "weights: pad-aware sharded_weight_loader",
    '''\
def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader
''',
    '''\
def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        avail = loaded_weight.shape[shard_axis] - start_idx
        if avail >= shard_size:
            return default_weight_loader(
                param, loaded_weight.narrow(shard_axis, start_idx, shard_size)
            )
        # Padded shard (per-head 1-D params at a padded head count):
        # zero-fill, then copy the clamped checkpoint-backed region.
        param.data.zero_()
        take = min(shard_size, max(avail, 0))
        if take > 0:
            src = loaded_weight.narrow(shard_axis, start_idx, take)
            dst = param.data
            for i in range(dst.dim()):
                if i < src.dim():
                    b = min(dst.shape[i], src.shape[i])
                    dst = dst.narrow(i, 0, b)
                    src = src.narrow(i, 0, b)
            dst.copy_(src)

    return loader
''',
    0,
)


# ---------------------------------------------------------------------------
# Applier machinery
# ---------------------------------------------------------------------------


def marker_for(label: str, replacement: str, indent: int) -> str:
    return " " * indent + f"# [{MARK} {_hash(replacement)}] {label}\n"


def marker_re_prefix(label: str, indent: int) -> str:
    # everything up to the hash — used to detect a STALE (different-hash) marker
    return " " * indent + f"# [{MARK} "


def apply_site(
    path: Path, label: str, anchor: str, replacement: str, indent: int, check: bool
) -> str:
    text = path.read_text()
    mark = marker_for(label, replacement, indent)
    if mark in text:
        return "applied"
    # a marker for this LABEL with a different hash = stale patch
    for line in text.splitlines(keepends=True):
        if line.startswith(marker_re_prefix(label, indent)) and line.rstrip().endswith(
            label
        ):
            return "STALE"
    n = text.count(anchor)
    if n == 0:
        return "MISS"
    if n > 1:
        return f"AMBIGUOUS({n})"
    if check:
        return "pending"
    bak = path.with_suffix(path.suffix + ".tp3bak")
    if not bak.exists():
        bak.write_text(text)
    path.write_text(text.replace(anchor, mark + replacement, 1))
    return "patched"


def install_pad_module(check: bool) -> str:
    src = HERE / "glm53_tp_pad.py"
    if not src.is_file():
        return "MISS(glm53_tp_pad.py not next to the applier)"
    dest = ROOT / "models/glm5next/glm53_tp_pad.py"
    body = src.read_text()
    if dest.exists() and dest.read_text() == body:
        return "applied"
    if check:
        return "pending"
    dest.write_text(body)
    return "patched"


def revert() -> int:
    restored = 0
    dest = ROOT / "models/glm5next/glm53_tp_pad.py"
    if dest.exists():
        dest.unlink()
        print(f"{LOG} removed {dest}")
        restored += 1
    for rel in sorted({s[0] for s in SITES}):
        p = ROOT / rel
        bak = p.with_suffix(p.suffix + ".tp3bak")
        if bak.exists():
            p.write_text(bak.read_text())
            bak.unlink()
            print(f"{LOG} restored {rel}")
            restored += 1
    print(f"{LOG} revert done ({restored} file(s))")
    return 0


def main(argv: list[str]) -> int:
    check = "--check" in argv
    if "--revert" in argv:
        return revert()
    if not ROOT.is_dir():
        print(f"{LOG} FATAL: vllm tree not found at {ROOT} (set VLLM_ROOT)")
        return 1

    failures = 0
    status = install_pad_module(check)
    print(f"{LOG} {status:>10}  install glm53_tp_pad.py -> models/glm5next/")
    if status.startswith("MISS"):
        failures += 1

    for rel, label, anchor, replacement, indent in SITES:
        p = ROOT / rel
        if not p.is_file():
            print(f"{LOG}       MISS  {label} ({rel} does not exist)")
            failures += 1
            continue
        status = apply_site(p, label, anchor, replacement, indent, check)
        print(f"{LOG} {status:>10}  {label}  [{rel}]")
        if status in ("MISS", "STALE") or status.startswith("AMBIGUOUS"):
            failures += 1

    if failures:
        print(
            f"{LOG} FATAL: {failures} site(s) failed — the tree does not match "
            f"what this patch was written against. REFUSING; do not serve."
        )
        return 1
    print(f"{LOG} all sites OK ({'check only' if check else 'applied'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
