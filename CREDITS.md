# Credits

## The recipe this is built on

**[MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark)**

This work exists because that repo already solved the hard parts: getting
GLM-5.3-Flash-NVFP4 serving on DGX Spark hardware at all — the SM121 kernel layer
(SM90 sparse-MLA + FA2 gated onto GB10, PDL off, the indexer top-k fixes), the
FlashInfer/NCCL/cutlass pins, the Ray tensor-parallel launcher, the multimodal
serving profile, and the UMA memory discipline. Going from two nodes to three only
required the head/vocab/MoE padding in this repo — everything underneath it came
from there.

This is **independent work inspired by that repo, not a fork**. None of the padding
machinery here exists upstream, because it does not need to: 64 attention heads
divide 2 cleanly, so the 2-node case never hits the problem this repo solves. The
3-node launcher is a derivative of their MIT-licensed `start.sh` with multi-worker
support added; the kernel layer (`files/`) is used as they ship it. Any errors in
the TP=3 work are mine, not theirs.

This repo follows the same shape as
[DeepSeek-V4-Flash-DSpark-3x-DGX-Spark](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark),
the earlier TP=3 conversion of a MiaAI-Lab 2-node recipe. The padding *mechanism*
differs (that model pads an attention *group* axis under a batched-matmul contract;
this one pads plain head counts) — see `docs/PATCH-SITES.md` for why the difference
matters.

## Upstream projects

* **[Z.ai / zai-org](https://huggingface.co/zai-org)** — GLM-5.3-Flash.
* **[LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)** — the
  NVFP4 quantization this recipe was built around, and
  **[dealignai](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4)** —
  the geometry-identical derivative this deployment serves.
* **[vLLM](https://github.com/vllm-project/vllm)** — the serving engine (the
  `glm53-flash-arm64-cu130` image these patches are applied to).
* **NVIDIA** — DGX Spark / GB10, and the glm5next model tree in that image.
