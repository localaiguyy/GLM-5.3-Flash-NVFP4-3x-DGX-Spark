# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash TP padding for non-divisible tensor-parallel sizes (TP=3 on 3 Sparks).

WHY THIS EXISTS
---------------
GLM-5.3-Flash has 64 MLA attention heads (the 11 deepseek_sparse_attention
layers + the MTP layer) and 64 KDA linear-attention heads (the other 34
layers). 64 % 3 != 0, so vLLM refuses TP=3 three separate times:

    config/model.py            verify_with_parallel_config   raise ValueError
    models/glm5next/.../model.py   num_attention_heads assert
    models/glm5next/.../kda.py     num_heads assert + divide()

and two more dimensions fail after the heads are fixed:

    moe_intermediate_size = 2048    (2048 % 3 != 0; routed AND shared experts)
    vocab_size            = 154880  (154880 % 3 != 0)

This module supplies the padding arithmetic so a 3-node tensor-parallel
deployment is legal AND numerically identical to TP=1. It is pure Python on
purpose — no torch import — so the arithmetic is unit-testable anywhere:

    python3 glm53_tp_pad.py --test

HOW THIS DIFFERS FROM THE DSv4 TP=3 PATCH (the sibling repo)
------------------------------------------------------------
DeepSeek-V4-Flash's o_proj is a per-group batched matmul with a hard
`r = heads_per_group * head_dim` contract baked into the checkpoint, plus
per-head attention sinks whose pad lanes must be -inf. GLM-5.3-Flash has
NEITHER: its MLA layers use a plain per-head o_proj (no groups), and there
are no sink tensors anywhere in the checkpoint (verified over all 113,074
tensor names). So the whole group-padding machinery from the DSv4 repo does
not exist here, and every padded lane is plain ZERO:

  * a pad head's q_b/kv_b rows are zero  -> its attention output is zero
  * its o_proj columns are zero          -> it contributes nothing to hidden
  * a pad KDA head's q/k/v rows are zero -> its delta-rule state never
    updates from zero, and its o_proj columns are zero anyway
  * pad MoE columns are zero             -> silu(0) * 0 = 0 through SwiGLU
    (and through SwiGLU-clamp: the clamp of 0 is 0)

The one value that must NOT be assumed zero is what `torch.empty` leaves in
a pad tail: the modelopt-NVFP4 quant path allocates expert weights with
torch.empty (verified in this image's modelopt.py — 15+ sites, zero
torch.zeros), so the loader patches ZERO-FILL every pad tail explicitly.
0 * garbage = garbage; 0 * 0 = 0.

THE NUMBERS (GLM-5.3-Flash-NVFP4 at TP=3, measured from the checkpoint)
-----------------------------------------------------------------------
  MLA heads      64 -> 66   (22/rank; ranks 0-1 all-real, rank 2: 20 real + 2 pad)
  KDA heads      64 -> 66   (same split; projection 8192 -> 8448, 2816/rank)
  MoE interm.  2048 -> 2112 (704/rank; rank 2: 640 real + 64 pad; 16- and
                             64-aligned so NVFP4 group-16 packing and marlin
                             tiles stay legal: 704 % 64 == 0)
  vocab      154880 -> 154944 (51648/rank; lcm(64, 3) = 192 alignment)

Set VLLM_GLM53_TP_PAD_ALIGN8=1 to round the per-rank head count up to a
multiple of 8 instead (64 -> 72, 24/rank) if an attention kernel objects to
22 local heads. That costs ~12% extra attention compute for nothing when 22
works — measure before keeping it.

Enabled by env VLLM_GLM53_TP_PAD (default "1").
"""

from __future__ import annotations

import math
import os

__all__ = [
    "tp_pad_enabled",
    "align8_enabled",
    "maybe_pad_heads",
    "maybe_pad_multiple",
    "rank_real_width",
    "describe_pad",
]


def tp_pad_enabled() -> bool:
    return os.environ.get("VLLM_GLM53_TP_PAD", "1") not in ("", "0", "false", "False")


def align8_enabled() -> bool:
    return os.environ.get("VLLM_GLM53_TP_PAD_ALIGN8", "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def maybe_pad_heads(num_heads: int, tp_size: int) -> int:
    """Smallest TP-legal head count >= num_heads.

    Identity when tp already divides num_heads (TP=1/2/4/8...), so every
    call site is a no-op on the shapes the stock code already supports.

    With VLLM_GLM53_TP_PAD_ALIGN8=1 the per-rank count is additionally
    rounded to a multiple of 8 (64 -> 72 at TP=3 instead of 66).
    """
    if tp_size <= 1 or num_heads % tp_size == 0:
        return num_heads
    if not tp_pad_enabled():
        return num_heads  # let the stock assert fire and say why
    local = -(-num_heads // tp_size)  # ceil
    if align8_enabled():
        local = ((local + 7) // 8) * 8
    return local * tp_size


def maybe_pad_multiple(value: int, tp_size: int, align: int = 64) -> int:
    """Smallest v >= value with v % lcm(tp_size, align) == 0.

    Used for the MoE intermediate size (align=64 keeps the per-rank width
    NVFP4-group-16 and marlin-tile legal) and, with the same arithmetic, the
    vocab (lcm(64, 3) = 192 -> 154880 -> 154944).

    Identity when value already divides cleanly.
    """
    if tp_size <= 1:
        return value
    step = math.lcm(tp_size, align)
    if value % step == 0:
        return value
    if not tp_pad_enabled():
        return value
    return -(-value // step) * step


def rank_real_width(total_real: int, padded_per_rank: int, tp_rank: int) -> int:
    """How many REAL (checkpoint-backed) units rank `tp_rank` holds.

    Under the global-pad scheme every rank owns `padded_per_rank` units and
    rank r's slice starts at r * padded_per_rank in checkpoint coordinates;
    the tail past `total_real` is pad. This is the clamp every patched
    loader applies:  min(padded_per_rank, total_real - r * padded_per_rank).
    """
    start = tp_rank * padded_per_rank
    return max(0, min(padded_per_rank, total_real - start))


def describe_pad(kind: str, real: int, padded: int, tp_size: int) -> str:
    """One-line INFO string so a padded boot is visible in the log."""
    local = padded // tp_size
    last_real = rank_real_width(real, local, tp_size - 1)
    return (
        f"GLM53 TP pad [{kind}]: {real} -> {padded} (tp={tp_size}, "
        f"{local}/rank; last rank holds {last_real} real + "
        f"{local - last_real} zero-pad)"
    )


# ---------------------------------------------------------------------------
# Self-tests: the real GLM-5.3-Flash-NVFP4 geometry, asserted.
# ---------------------------------------------------------------------------


def _run_tests() -> None:
    # identity on divisible shapes — every site must be a no-op at TP=1/2/4
    for tp in (1, 2, 4):
        assert maybe_pad_heads(64, tp) == 64, tp
        assert maybe_pad_multiple(2048, tp) == 2048, tp
    assert maybe_pad_multiple(154880, 2, align=64) == 154880
    # 12288 (dense MLP) divides 3 — never padded
    assert maybe_pad_multiple(12288, 3) == 12288

    # TP=3: the numbers this repo exists for
    os.environ.pop("VLLM_GLM53_TP_PAD_ALIGN8", None)
    assert maybe_pad_heads(64, 3) == 66
    assert 66 // 3 == 22
    os.environ["VLLM_GLM53_TP_PAD_ALIGN8"] = "1"
    assert maybe_pad_heads(64, 3) == 72
    os.environ.pop("VLLM_GLM53_TP_PAD_ALIGN8")

    assert maybe_pad_multiple(2048, 3) == 2112
    assert 2112 // 3 == 704 and 704 % 64 == 0 and 704 % 16 == 0
    assert maybe_pad_multiple(154880, 3) == 154944
    assert 154944 // 3 == 51648 and 154944 % 64 == 0

    # per-rank real widths under the global-pad scheme
    #   MLA/KDA heads: 22, 22, 20 real (+0, +0, +2 pad)
    assert [rank_real_width(64, 22, r) for r in range(3)] == [22, 22, 20]
    #   MoE intermediate: 704, 704, 640 real (+0, +0, +64 pad)
    assert [rank_real_width(2048, 704, r) for r in range(3)] == [704, 704, 640]
    #   in packed-U8 units (2 nvfp4/byte): 352, 352, 320
    assert [rank_real_width(1024, 352, r) for r in range(3)] == [352, 352, 320]
    #   per-16-group scale columns: 44, 44, 40
    assert [rank_real_width(128, 44, r) for r in range(3)] == [44, 44, 40]
    #   KDA projection rows (64 heads x 128): 2816, 2816, 2560
    assert [rank_real_width(8192, 2816, r) for r in range(3)] == [2816, 2816, 2560]
    #   vocab rows: 51648, 51648, 51584
    assert [rank_real_width(154880, 51648, r) for r in range(3)] == [
        51648,
        51648,
        51584,
    ]
    #   align8 variant: heads 24/rank -> 24, 24, 16 real
    assert [rank_real_width(64, 24, r) for r in range(3)] == [24, 24, 16]

    # kill-switch: with the pad disabled everything is identity
    os.environ["VLLM_GLM53_TP_PAD"] = "0"
    try:
        assert maybe_pad_heads(64, 3) == 64
        assert maybe_pad_multiple(2048, 3) == 2048
    finally:
        os.environ.pop("VLLM_GLM53_TP_PAD")

    print("glm53_tp_pad: all self-tests passed")
    print(" ", describe_pad("mla-heads", 64, 66, 3))
    print(" ", describe_pad("kda-heads", 64, 66, 3))
    print(" ", describe_pad("moe-intermediate", 2048, 2112, 3))
    print(" ", describe_pad("vocab", 154880, 154944, 3))


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _run_tests()
    else:
        print(__doc__)
