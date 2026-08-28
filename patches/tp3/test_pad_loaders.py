#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Numerical tests for the TP=3 pad-aware loaders — against the SHIPPED code.

The three generic parameter loaders and sharded_weight_loader are tested by
exec()ing the exact replacement bodies out of apply_tp3_patch.py's SITES
table, so what is tested is what ships, not a copy that can drift. The
routed-expert loader blocks (which patch the middle of a method) are driven
by a faithful harness around the exec()d replacement block.

What is proven, on real torch tensors at TP=3:

  * every checkpoint element lands on exactly one rank, exactly once
    (concatenating the ranks' real regions reconstructs the checkpoint)
  * every pad lane is exactly zero — including when the parameter was
    allocated with torch.empty full of NaNs (the modelopt trap)
  * the stock path is taken, and is byte-identical, when nothing is padded
  * end-to-end numerics: a padded TP=3 column+row projection pipeline
    reproduces the TP=1 reference to float32 exactness
  * the stock floor-division expert math WOULD have dropped checkpoint
    columns; the patched math loses none (the regression this repo fixes)

Run:  python3 test_pad_loaders.py
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pull the SITES table (and the pad arithmetic) from the shipped files.
# ---------------------------------------------------------------------------


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patcher = _load("apply_tp3_patch", HERE / "apply_tp3_patch.py")
pad = _load("glm53_tp_pad", HERE / "glm53_tp_pad.py")

REPL = {label: (repl, indent) for _, label, _, repl, indent in patcher.SITES}


def exec_method(label: str, func_name: str):
    """exec() a full `def ...` replacement and return the function object."""
    body, _indent = REPL[label]
    ns: dict = {"torch": torch}
    exec(textwrap.dedent(body), ns)
    return ns[func_name]


class FakeParam:
    """Just enough of BasevLLMParameter for the exec()d loaders."""

    def __init__(self, data: torch.Tensor, output_dim=0, input_dim=0, tp_rank=0):
        self.data = data
        self._output_dim = output_dim
        self._input_dim = input_dim
        self.tp_rank = tp_rank

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def input_dim(self):
        return self._input_dim


PASS = 0


def ok(msg: str):
    global PASS
    PASS += 1
    print(f"  PASS {msg}")


def check(cond: bool, msg: str):
    if not cond:
        print(f"  FAIL {msg}")
        sys.exit(1)
    ok(msg)


TP = 3

# ---------------------------------------------------------------------------
# 1. load_column_parallel_weight — q_b_proj analog
#    64 heads x head_dim 4 -> rows 256; padded 66 heads -> 264, 88/rank.
# ---------------------------------------------------------------------------
print("== column parallel (q_b/kv_b analog) ==")
col = exec_method("param: pad-aware load_column_parallel_weight",
                  "load_column_parallel_weight")
ckpt = torch.randn(256, 8)
local_rows = pad.maybe_pad_heads(64, TP) // TP * 4          # 88
shards = []
for r in range(TP):
    p = FakeParam(torch.full((local_rows, 8), float("nan")), output_dim=0, tp_rank=r)
    col(p, ckpt)
    real = pad.rank_real_width(256, local_rows, r)
    check(torch.equal(p.data[:real], ckpt[r * local_rows : r * local_rows + real]),
          f"rank{r}: real rows exact ({real})")
    check(torch.all(p.data[real:] == 0), f"rank{r}: pad tail zero (was NaN)")
    shards.append(p.data)
recon = torch.cat([s[: pad.rank_real_width(256, local_rows, r)]
                   for r, s in enumerate(shards)])
check(torch.equal(recon, ckpt), "checkpoint reconstructs exactly once from ranks")

# stock path control: TP=2, divisible — result identical to a plain narrow
p = FakeParam(torch.full((128, 8), float("nan")), output_dim=0, tp_rank=1)
col(p, ckpt)
check(torch.equal(p.data, ckpt[128:256]), "divisible shard = stock narrow (TP=2)")

# ---------------------------------------------------------------------------
# 2. load_row_parallel_weight — o_proj analog + END-TO-END NUMERICS
# ---------------------------------------------------------------------------
print("== row parallel (o_proj analog) + end-to-end ==")
row = exec_method("param: pad-aware load_row_parallel_weight",
                  "load_row_parallel_weight")
W = torch.randn(8, 256)                       # o_proj [hidden, heads*v_dim]
padded_cols = pad.maybe_pad_heads(64, TP) // TP * 4 * TP    # 264
local_cols = padded_cols // TP
tokens = 5
h_real = torch.randn(tokens, 256)             # real per-head outputs
y_ref = h_real @ W.t()                        # TP=1 reference
y_sum = torch.zeros(tokens, 8)
for r in range(TP):
    p = FakeParam(torch.full((8, local_cols), float("nan")), input_dim=1, tp_rank=r)
    row(p, W)
    real = pad.rank_real_width(256, local_cols, r)
    check(torch.all(p.data[:, real:] == 0), f"rank{r}: pad cols zero")
    # this rank's activation slice: real head lanes + ZERO pad lanes
    h_r = torch.zeros(tokens, local_cols)
    h_r[:, :real] = h_real[:, r * local_cols : r * local_cols + real]
    y_sum += h_r @ p.data.t()                 # the all-reduce
check(torch.allclose(y_sum, y_ref, atol=1e-5),
      "TP=3 padded row pipeline == TP=1 reference (all-reduce sum)")

# ---------------------------------------------------------------------------
# 3. load_merged_column_weight — KDA in_proj / shared-expert gate_up analog
# ---------------------------------------------------------------------------
print("== merged column (KDA in_proj / gate_up analog) ==")
_merged_block, _ = REPL["param: pad-aware load_merged_column_weight"]


def merged(p, loaded_weight, *, shard_offset, shard_size):
    """Drive the exec()d mid-method replacement block for the merged loader."""
    loc = dict(self=p, loaded_weight=loaded_weight,
               shard_offset=shard_offset, shard_size=shard_size)
    blk = textwrap.dedent(_merged_block).replace("return\n", "raise StopIteration\n")
    try:
        exec(blk, {"torch": torch}, loc)
    except StopIteration:
        pass
# merged param: [q 88 | b 22] per rank (padded from 256-row q + 64-row b)
q_ckpt = torch.randn(256, 8)
b_ckpt = torch.randn(64, 8)
for r in range(TP):
    data = torch.full((88 + 22, 8), float("nan"))
    p = FakeParam(data, output_dim=0, tp_rank=r)
    merged(p, q_ckpt, shard_offset=0, shard_size=88)
    merged(p, b_ckpt, shard_offset=88, shard_size=22)
    qr = pad.rank_real_width(256, 88, r)
    br = pad.rank_real_width(64, 22, r)
    check(torch.equal(data[:qr], q_ckpt[r * 88 : r * 88 + qr])
          and torch.all(data[qr:88] == 0),
          f"rank{r}: q shard exact + zero tail ({qr} real)")
    check(torch.equal(data[88 : 88 + br], b_ckpt[r * 22 : r * 22 + br])
          and torch.all(data[88 + br : 110] == 0),
          f"rank{r}: b shard exact + zero tail ({br} real)")
# replicated-entry control (KDA f_a/g_a force tp_rank=0, full width)
data = torch.full((16, 8), float("nan"))
p = FakeParam(data, output_dim=0, tp_rank=0)
merged(p, torch.randn(16, 8), shard_offset=0, shard_size=16)
check(not torch.isnan(data).any(), "replicated entry: stock full copy")

# ---------------------------------------------------------------------------
# 4. sharded_weight_loader — dt_bias / A_log analog
# ---------------------------------------------------------------------------
print("== sharded_weight_loader (dt_bias / A_log analog) ==")
body, _ = REPL["weights: pad-aware sharded_weight_loader"]
_rank_holder = {"r": 0}
ns = {
    "torch": torch,
    "LoaderFunction": object,
    "get_tensor_model_parallel_rank": lambda: _rank_holder["r"],
    "default_weight_loader": lambda p, w: p.data.copy_(w),
}
exec(textwrap.dedent(body), ns)
swl = ns["sharded_weight_loader"]
dt_ckpt = torch.randn(256)                    # dt_bias analog (8192 -> 256)
for r in range(TP):
    _rank_holder["r"] = r
    p = FakeParam(torch.full((88,), float("nan")))
    swl(0)(p, dt_ckpt)
    real = pad.rank_real_width(256, 88, r)
    check(torch.equal(p.data[:real], dt_ckpt[r * 88 : r * 88 + real])
          and torch.all(p.data[real:] == 0),
          f"rank{r}: 1-D shard exact + zero tail")
# A_log analog: 4-D [1,1,H,1] sharded on axis 2
a_ckpt = torch.randn(1, 1, 64, 1)
for r in range(TP):
    _rank_holder["r"] = r
    p = FakeParam(torch.full((1, 1, 22, 1), float("nan")))
    swl(2)(p, a_ckpt)
    real = pad.rank_real_width(64, 22, r)
    check(torch.equal(p.data[0, 0, :real, 0], a_ckpt[0, 0, r * 22 : r * 22 + real, 0])
          and torch.all(p.data[0, 0, real:, 0] == 0),
          f"rank{r}: A_log 4-D shard exact + zero tail")

# ---------------------------------------------------------------------------
# 5. routed-expert loader blocks — non-divisible checkpoint width
#    Harness mirrors _load_w13's surrounding method; the patched decision
#    block itself is exec()d from the SITES table.
# ---------------------------------------------------------------------------
print("== routed experts (w13/w2 blocks + zero-tail) ==")


class _MPC:
    tp_size = TP


class _MC:
    moe_parallel_config = _MPC()


class _Loader:
    moe_config = _MC()


w13_block, _ = REPL["experts: w13 shard math for a non-divisible checkpoint"]
narrow_block, _ = REPL["experts: zero the pad tail before narrowing it away"]


def run_w13(expert_data, loaded_weight, shard_id, tp_rank, shard_dim=0):
    """Faithful mirror of the patched _load_w13 control flow."""
    self = _Loader()
    shard_size = expert_data.shape[shard_dim] // 2      # is_act_and_mul
    loc = dict(self=self, shard_size=shard_size, shard_dim=shard_dim,
               shard_id=shard_id, tp_rank=tp_rank,
               expert_data=expert_data, loaded_weight=loaded_weight)
    blk = textwrap.dedent(w13_block)
    # the block ends with an early `return` path; emulate via exception
    blk = blk.replace("return\n", "raise StopIteration\n")
    try:
        exec(blk, {"torch": torch}, loc)
    except StopIteration:
        return
    # stock tail that follows the patched block in _load_w13
    narrow_size = min(loc["loaded_per_rank"], loc["available"])
    loaded_weight = loc["loaded_weight"].narrow(
        shard_dim, loc["start_offset"], narrow_size)
    if shard_id == "w1":
        ed = expert_data.narrow(shard_dim, 0, shard_size)
    else:
        ed = expert_data.narrow(shard_dim, shard_size, shard_size)
    # _narrow_expert_data_for_padding, patched (zero tail then narrow)
    loc2 = dict(expert_data=ed, loaded_weight=loaded_weight,
                hidden_dim=1, shard_dim=shard_dim)
    blk2 = textwrap.dedent(narrow_block).replace("return expert_data\n", "")
    exec(blk2, {"torch": torch}, loc2)
    ed = loc2["expert_data"]
    for dim in (1, shard_dim):
        if ed.shape[dim] > loaded_weight.shape[dim]:
            ed = ed.narrow(dim, 0, loaded_weight.shape[dim])
    ed.copy_(loaded_weight)


# intermediate 32 @ tp=3 -> padded 36, 12/rank (real 12/12/8): the stock
# floor math would shard by 32//3=10 and DROP the last 2 columns entirely.
inter, hidden = 32, 6
padded = 36
per_rank = padded // TP
gate_ckpt = torch.randn(inter, hidden)
seen = []
for r in range(TP):
    ed = torch.full((2 * per_rank, hidden), float("nan"))   # gate+up slab
    run_w13(ed, gate_ckpt, "w1", r)
    real = pad.rank_real_width(inter, per_rank, r)
    check(torch.equal(ed[:real], gate_ckpt[r * per_rank : r * per_rank + real]),
          f"rank{r}: w13 gate rows exact ({real} real)")
    check(torch.all(ed[real:per_rank] == 0), f"rank{r}: w13 pad tail zero (was NaN)")
    seen.append(ed[:real])
recon = torch.cat(seen)
check(torch.equal(recon, gate_ckpt),
      "w13: every checkpoint row loaded exactly once (stock floor math drops 2)")

# w2: shard on the input dim (packed-units analog)
w2_block, _ = REPL["experts: w2 shard math for a non-divisible checkpoint"]


def run_w2(expert_data, loaded_weight, tp_rank, shard_dim=1):
    self = _Loader()
    loc = dict(self=self, shard_dim=shard_dim, tp_rank=tp_rank,
               expert_data=expert_data, loaded_weight=loaded_weight)
    blk = textwrap.dedent(w2_block).replace("return\n", "raise StopIteration\n")
    try:
        exec(blk, {"torch": torch}, loc)
    except StopIteration:
        return
    # stock tail that follows the patched block in _load_w2
    narrow_size = min(loc["loaded_per_rank"], loc["available"])
    loaded_weight = loc["loaded_weight"].narrow(
        shard_dim, loc["start_offset"], narrow_size)
    ed = expert_data
    loc2 = dict(expert_data=ed, loaded_weight=loaded_weight,
                hidden_dim=0, shard_dim=shard_dim)
    blk2 = textwrap.dedent(narrow_block).replace("return expert_data\n", "")
    exec(blk2, {"torch": torch}, loc2)
    ed = loc2["expert_data"]
    for dim in (0, shard_dim):
        if ed.shape[dim] > loaded_weight.shape[dim]:
            ed = ed.narrow(dim, 0, loaded_weight.shape[dim])
    ed.copy_(loaded_weight)


down_ckpt = torch.randn(hidden, inter)
seen = []
for r in range(TP):
    ed = torch.full((hidden, per_rank), float("nan"))
    run_w2(ed, down_ckpt, r)
    real = pad.rank_real_width(inter, per_rank, r)
    check(torch.equal(ed[:, :real], down_ckpt[:, r * per_rank : r * per_rank + real])
          and torch.all(ed[:, real:] == 0),
          f"rank{r}: w2 cols exact + zero tail ({real} real)")
    seen.append(ed[:, :real])
check(torch.equal(torch.cat(seen, dim=1), down_ckpt),
      "w2: every checkpoint column loaded exactly once")

# divisible control: checkpoint 30 @ tp=3 — patched math must equal stock
ed = torch.full((2 * 10, hidden), float("nan"))
run_w13(ed, torch.randn(30, hidden)[:30], "w1", 1)
check(not torch.isnan(ed[:10]).any(), "divisible checkpoint: stock floor path")

# ---------------------------------------------------------------------------
# 6. end-to-end MoE numerics: padded TP=3 SwiGLU expert == TP=1 reference
# ---------------------------------------------------------------------------
print("== end-to-end expert numerics ==")
gate_full = torch.randn(inter, hidden)
up_full = torch.randn(inter, hidden)
down_full = torch.randn(hidden, inter)
x = torch.randn(4, hidden)
act = lambda g, u: torch.nn.functional.silu(g) * u          # noqa: E731
y_ref = act(x @ gate_full.t(), x @ up_full.t()) @ down_full.t()
y_sum = torch.zeros(4, hidden)
for r in range(TP):
    slab = torch.full((2 * per_rank, hidden), float("nan"))
    run_w13(slab, gate_full, "w1", r)
    run_w13(slab, up_full, "w3", r)
    w2 = torch.full((hidden, per_rank), float("nan"))
    run_w2(w2, down_full, r)
    g, u = slab[:per_rank], slab[per_rank:]
    y_sum += act(x @ g.t(), x @ u.t()) @ w2.t()
check(torch.allclose(y_sum, y_ref, atol=1e-4),
      "TP=3 padded SwiGLU expert == TP=1 reference (through zero pad lanes)")

print(f"\nall {PASS} loader checks passed")
