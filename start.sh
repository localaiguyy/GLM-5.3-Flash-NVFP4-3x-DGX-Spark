#!/usr/bin/env bash
# ============================================================================
# start.sh — GLM-5.3-Flash-NVFP4 on 3× DGX Spark (TP=3)
# ============================================================================
#
# Derivative of MiaAI-Lab's 2-node launcher
# (https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark, MIT).
# Their start.sh assumes exactly one worker; this one drives N workers
# (default 2 → TP=3) and applies the TP=3 padding patch inside every
# container before anything imports vllm. Everything else — the SM121
# kernel image, the serving profile, the NCCL pins, the UMA discipline —
# is their work. See CREDITS.md.
#
#   head    : this machine (spark1) — Ray head + vLLM API :8888
#   worker1 : second Spark — Ray worker, 1× GB10
#   worker2 : third Spark  — Ray worker, 1× GB10
#   layout  : tensor-parallel-size 3 (one GPU per node, Ray executor)
#
# What we do:
#   1. preflight  — docker/ssh/disk on every node + TP-geometry sanity
#   2. image      — mia/glm53-flash-spark:mm-ray-v1 (build if missing, ship to all)
#   3. download   — weights into local HF cache if missing (~181 GiB)
#   4. sync       — rsync cache + the tp3 patch to every worker (each rank
#                   loads from local disk; a failed patch sync REFUSES boot)
#   5. launch     — Ray workers on spark2+3, Ray head + `vllm serve` on spark1.
#                   Every container runs apply_tp3_patch.py first (|| exit 1).
#   6. wait       — poll /health up to READY_TIMEOUT (320B MoE init is slow)
#
# The TP=3 part (why stock refuses to start, and why our boot is different):
#   GLM-5.3-Flash has 64 attention heads (MLA) and 64 linear-attention heads
#   (KDA). 64 % 3 != 0, so stock vLLM rejects TP=3 during config validation.
#   patches/tp3/apply_tp3_patch.py pads heads 64→66, the MoE intermediate
#   2048→2112, and the vocab 154880→154944, with zeroed pad slabs that
#   contribute exactly nothing. See docs/PATCH-SITES.md.
#
# Usage:
#   ./start.sh                    start (download/sync/patch/launch) — default
#   ./start.sh stop               stop all nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers, API health, Ray cluster
#   ./start.sh logs               follow head logs (driver + API server)
#   ./start.sh logs worker1       follow a worker container's logs
#
# Handy env overrides (all optional, defaults match a 3× Spark kit):
#   MODEL=dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4   serve a geometry-identical
#                                 derivative (see CHAT_TEMPLATE_URL note below)
#   TP=3                          tensor parallel size (= node count)
#   MOE_BACKEND=marlin            marlin | native | auto
#   MAX_MODEL_LEN=1048576         model is 1M-native; default here is 262144
#   GPU_MEM_UTIL=0.80             vLLM memory budget (GB10 UMA; higher can trip Ray memory monitor)
#   MTP_TOKENS=4                  MTP speculative tokens (0 disables)
#   MAX_NUM_SEQS=8                scheduler concurrency — raise this first when
#                                 chasing aggregate throughput (see BENCHMARKS)
#   PORT=8888                     API port on the head
#   EXTRA_ARGS='--max-num-seqs 32'  extra flags appended to `vllm serve`
#   TP3_PATCH=0                   skip the padding patch (TP=1/2 only!)
#   CHAT_TEMPLATE_URL=...         where to fetch the multimodal chat template.
#                                 For derivative checkpoints that snapshotted a
#                                 stale template, point this at the base model's
#                                 (the emit_image guard below refuses a stale one).
#   NCCL_DEBUG=INFO               passed through to all containers
#   HF_HUB_OFFLINE=1              skip HF etag checks at startup
#   SKIP_TEMPLATE_REFRESH=1       use cached chat_template.jinja, still verify emit_image
#   REFRESH_WEIGHTS=1 SKIP_SYNC=1 SKIP_DOWNLOAD=1 PULL=1 TAIL=1
# ============================================================================
set -euo pipefail

# ----------------------------- env file ------------------------------------
# Optional: load every setting from a file instead of exporting by hand:
#   ENV_FILE=my.env ./start.sh
# Everything in the file is exported; every default below still applies for
# anything the file leaves unset.
# Default to ./.env when present: without this, a bare `./start.sh stop` silently
# falls back to the PLACEHOLDER worker identities below and reports success while
# the real workers keep running.
: "${ENV_FILE:=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/.env}"
[ -f "$ENV_FILE" ] || ENV_FILE=""
if [ -n "${ENV_FILE:-}" ]; then
    [ -f "$ENV_FILE" ] || { echo "[glm53-tp3] ENV_FILE=$ENV_FILE not found" >&2; exit 1; }
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

# ----------------------------- configuration -------------------------------
MODEL="${MODEL:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
MODEL_CACHE_NAME="models--${MODEL//\//--}"
IMAGE="${IMAGE:-mia/glm53-flash-spark:mm-ray-v1}"
# IMAGE is MiaAI's serving tag (Ray + MM + :8888 on their glm53-flash-sm121:v8
# kernel layer). Build it from their repo's files/ (see docs/CHANGES-REQUIRED.md);
# the kernel layer is TP-agnostic and is used here exactly as they ship it.
RAY_VERSION="${RAY_VERSION:-2.58.0}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"

# ---- workers ---------------------------------------------------------------
# One block per worker. Worker 1 defaults match MiaAI's 2-node kit; worker 2
# is the third Spark. On a switched fabric all CX7 ports face the switch; on
# a triangle of direct QSFP links set each worker's IF/IB to the port that
# faces the HEAD (NCCL rings run head<->worker per rank pair via Ray).
N_WORKERS="${N_WORKERS:-2}"                     # TP = N_WORKERS + 1

WORKER1_SSH="${WORKER1_SSH:-${WORKER_SSH:-zurih@10.0.0.2}}"
WORKER1_IP="${WORKER1_IP:-${WORKER_IP:-10.0.0.2}}"
WORKER1_HOME="${WORKER1_HOME:-${WORKER_HOME:-/home/zurih}}"
WORKER1_CX7_IF="${WORKER1_CX7_IF:-${WORKER_CX7_IF:-enp1s0f0np0}}"
WORKER1_CX7_IB="${WORKER1_CX7_IB:-${WORKER_CX7_IB:-rocep1s0f0}}"

WORKER2_SSH="${WORKER2_SSH:-zurih@10.0.0.3}"
WORKER2_IP="${WORKER2_IP:-10.0.0.3}"
WORKER2_HOME="${WORKER2_HOME:-/home/zurih}"
WORKER2_CX7_IF="${WORKER2_CX7_IF:-enp1s0f0np0}"
WORKER2_CX7_IB="${WORKER2_CX7_IB:-rocep1s0f0}"

# Head-side CX7 pins. Ray can use IP aliases; NCCL cannot — without these
# pins it busy-waits forever in ncclCommInitRank (~100% CPU, no /health).
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# -1 lets NCCL auto-select the RoCEv2 GID per node. Do NOT hardcode an index:
# the v2/IPv4 GID is NOT guaranteed to land on the same index on every node,
# even across identical hardware (observed: two nodes at 3, one at 4). Because
# this knob is global, any fixed value can point some node at a RoCEv1 GID,
# which fails deep inside ncclCommInitRank as an opaque
# "NCCL error: unhandled system error" AFTER the full ring has been built.
# Verify per node with `show_gids` before overriding.
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:--1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
USE_HOST_NCCL="${USE_HOST_NCCL:-1}"
# GB10 is UMA: Ray's default object store (~30% of RAM) steals from the GPU
# budget and vLLM then refuses to start (free < gpu-memory-utilization).
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"  # 4 GiB
# GB10 GPU allocations are backed by unified system memory. During vLLM init,
# Ray's node-memory monitor can see the GPU/JIT transient as host pressure and
# kill the worker actor even though the kernel is not OOM-killing anything.
RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

TP="${TP:-$((N_WORKERS + 1))}"           # 1 GB10 per node
RAY_PORT="${RAY_PORT:-6379}"
PORT="${PORT:-8888}"

MTP_TOKENS="${MTP_TOKENS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LIMIT_MM="${LIMIT_MM:-{\"image\":4,\"video\":1}}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"      # auto | native | marlin
# The vision tower has 16 attention heads and 16 % 3 != 0. Rather than
# padding the ViT, run the multimodal encoder DATA-parallel — fully
# replicated on every rank (~0.9 GiB BF16). The model class inherits
# supports_encoder_tp_data from the GLM-4V family, so this is a plain
# config switch, not a patch.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"          # = VLLM_ENGINE_READY_TIMEOUT_S
CLUSTER_WAIT_ITERS="${CLUSTER_WAIT_ITERS:-120}" # x5s to join Ray

CONTAINER_HEAD="glm53-flash-head"
CACHE_VOLUME="${CACHE_VOLUME:-glm53-flash-cache-sm121}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-worker.inner.sh"
KERNEL_ERR_PAT='NoKernelImageForDevice|no kernel image is available'

# The TP=3 padding patch. Mounted read-only into every container and applied
# before anything imports vllm. TP3_PATCH=0 skips it (only legal when TP
# already divides the model's 64 heads, i.e. TP=1/2/4).
TP3_PATCH="${TP3_PATCH:-1}"
TP3_PATCH_DIR="${TP3_PATCH_DIR:-$SCRIPT_DIR/patches/tp3}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-tp3]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-tp3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-tp3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# worker accessors — $1 is the worker index (1..N_WORKERS)
w_ssh_target() { local v="WORKER$1_SSH";    printf '%s' "${!v}"; }
w_ip()         { local v="WORKER$1_IP";     printf '%s' "${!v}"; }
w_home()       { local v="WORKER$1_HOME";   printf '%s' "${!v}"; }
w_if()         { local v="WORKER$1_CX7_IF"; printf '%s' "${!v}"; }
w_ib()         { local v="WORKER$1_CX7_IB"; printf '%s' "${!v}"; }
w_cache()      { printf '%s/.cache/huggingface' "$(w_home "$1")"; }
w_nccl_dir()   {
    local v="WORKER$1_NCCL_HOST_DIR"
    printf '%s' "${!v:-$(w_home "$1")/nccl-2.30.7}"
}
w_container()  { printf 'glm53-flash-worker%s' "$1"; }

worker_ssh() {  # worker_ssh <idx> <cmd...>
    local i="$1"; shift
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$(w_ssh_target "$i")" "$@"
}

each_worker() { seq 1 "$N_WORKERS"; }

usage() { sed -n '2,68p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# Resolve the HF-cache snapshot directory into the IN-CONTAINER path that
# vllm serve should load from. We pass this instead of the repo id because
# vllm's custom Glm5NextProcessor.from_pretrained() does a raw
# open(os.path.join(model_path, "processor_config.json")) and does NOT
# resolve HF repo ids through the cache. (MiaAI's finding, kept verbatim.)
resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run download first"
    hash="$(<"$ref")"
    [ -n "$hash" ] || die "empty refs/main at $ref"
    local dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/processor_config.json" ] \
        || die "processor_config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use by another service — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running in the background (./start.sh logs to watch, ./start.sh stop to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    command -v python3 >/dev/null 2>&1 || die "python3 not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set HEAD_IP=<ip the workers can reach>"

    # TP-geometry sanity: 64 heads. TP in {1,2,4,8...} needs no patch; TP=3
    # (or any non-divisor) REQUIRES it. Refusing here beats a silent mis-shard.
    if [ "$TP" -ne "$((N_WORKERS + 1))" ]; then
        die "TP=${TP} but N_WORKERS=${N_WORKERS} (+1 head) — on 1-GPU-per-node Sparks these must match"
    fi
    if [ $((64 % TP)) -ne 0 ] && [ "$TP3_PATCH" != "1" ]; then
        die "64 heads % TP=${TP} != 0 and TP3_PATCH=0 — stock vLLM will refuse (or worse, mis-shard). Enable the patch."
    fi
    if [ "$TP3_PATCH" = "1" ]; then
        [ -f "$TP3_PATCH_DIR/apply_tp3_patch.py" ] \
            || die "TP3_PATCH_DIR=$TP3_PATCH_DIR has no apply_tp3_patch.py"
    fi

    local i
    for i in $(each_worker); do
        log "checking worker${i} $(w_ssh_target "$i") ..."
        worker_ssh "$i" true 2>/dev/null \
            || die "cannot ssh (key-based) to $(w_ssh_target "$i") — set up passwordless ssh first"
        worker_ssh "$i" "docker info >/dev/null 2>&1" \
            || die "worker${i} cannot talk to its docker daemon (docker group?)"
        worker_ssh "$i" "nvidia-smi -L 2>/dev/null | grep -q GB10" \
            || warn "no GB10 GPU visible on worker${i}"
    done

    # GLM-5.3-Flash-NVFP4 at TP=3 needs ~60 GiB of weights + KV on each GB10 —
    # it cannot coexist with another served model on any node.
    local others
    others=$(docker ps --format '  {{.Names}}  ({{.Image}})' | grep -v "^  ${CONTAINER_HEAD}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the head:"
        echo "$others" >&2
        warn "GLM-5.3-Flash needs the GPU budget of each GB10 — stop GPU containers first"
    fi
    for i in $(each_worker); do
        others=$(worker_ssh "$i" "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  $(w_container "$i")" || true)
        if [ -n "$others" ]; then
            warn "other containers are running on worker${i}:"
            echo "$others" >&2
        fi
    done

    check_port_free "$PORT" PORT
    check_port_free "$RAY_PORT" RAY_PORT

    # disk space: model is ~181 GiB per node
    local need_kb=$((190 * 1024 * 1024)) avail
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~181 GiB model"
    for i in $(each_worker); do
        avail=$(worker_ssh "$i" "df -Pk '$(w_home "$i")' 2>/dev/null" | awk 'NR==2{print $4}' || true)
        [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker${i} for a ~181 GiB model"
    done

    log "preflight OK (head=$(hostname) ${HEAD_IP}, ${N_WORKERS} worker(s), TP=${TP})"
}

# ------------------------------ image pull ---------------------------------
image_is_local() {
    case "$IMAGE" in
        mia/glm53-flash-spark:*|glm53-flash-sm121:*|localhost/*) return 0 ;;
        */*) return 1 ;;
        *) return 0 ;;
    esac
}

ship_image_to_worker() {  # $1 = worker idx
    log "shipping ${IMAGE} to worker$1 via docker save | ssh docker load ..."
    docker save "$IMAGE" | worker_ssh "$1" docker load >/dev/null
}

ensure_local_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 i
    docker image inspect "$IMAGE" >/dev/null 2>&1 && head_ok=1
    if [ "$head_ok" = "0" ] || [ "${PULL:-0}" = "1" ]; then
        # The image is MiaAI's: build it from their repo (files/build.sh).
        # We refuse to guess at a build here — their build script owns the
        # kernel layer (glm53-flash-sm121:v8) and the mm-ray layer.
        local build="${MIA_REPO_DIR:-$SCRIPT_DIR/../GLM-5.3-Flash-NVFP4-Dual-DGX-Spark}/files/build.sh"
        [ -x "$build" ] || die "image $IMAGE missing and MiaAI build.sh not found — clone MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark next to this repo or set MIA_REPO_DIR"
        log "building ${IMAGE} via ${build} (log: $LOGDIR/build-sm121.log) ..."
        SKIP_WORKER_LOAD=1 IMAGE="$IMAGE" "$build" >"$LOGDIR/build-sm121.log" 2>&1 \
            || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
    fi
    for i in $(each_worker); do
        if ! worker_ssh "$i" "docker image inspect '$IMAGE' >/dev/null 2>&1" || [ "${PULL:-0}" = "1" ]; then
            ship_image_to_worker "$i"
        fi
    done
    log "image ready on all nodes"
}

pull_images() {
    mkdir -p "$LOGDIR"
    if image_is_local; then
        ensure_local_image
        return
    fi
    local i ok=1
    docker image inspect "$IMAGE" >/dev/null 2>&1 || ok=0
    for i in $(each_worker); do
        worker_ssh "$i" "docker image inspect '$IMAGE' >/dev/null 2>&1" || ok=0
    done
    if [ "$ok" = "1" ] && [ "${PULL:-0}" != "1" ]; then
        log "image $IMAGE present on all nodes (PULL=1 to refresh)"
        return
    fi
    log "pulling ${IMAGE} on all nodes in parallel (logs: $LOGDIR/pull-*.log) ..."
    local -a pids=()
    docker pull "$IMAGE" >"$LOGDIR/pull-head.log" 2>&1 & pids+=($!)
    for i in $(each_worker); do
        worker_ssh "$i" "docker pull '$IMAGE'" >"$LOGDIR/pull-worker${i}.log" 2>&1 & pids+=($!)
    done
    local fail=0 p
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ "$fail" = "0" ] || { tail -n 20 "$LOGDIR"/pull-*.log >&2; die "image pull failed"; }
    log "image ready on all nodes"
}

# ---------------------------- weight download ------------------------------
download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    local need=0
    if [ ! -d "$MODEL_PATH" ]; then
        need=1
    elif [ -z "$(find "$MODEL_PATH/snapshots" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
        # NOTE: no '-type f' — HF cache snapshot entries are symlinks into blobs/
        need=1
    elif [ "${REFRESH_WEIGHTS:-0}" = "1" ]; then
        need=1
    fi
    [ "$need" = "0" ] && { log "weights already present: $MODEL_PATH"; return; }

    local hf
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"

    log "downloading ${MODEL} (~181 GiB / 120 shards) into ${HF_CACHE_DIR} ..."
    "$hf" download "$MODEL"
    log "download complete"
}

# Latest multimodal chat_template.jinja (emit_image / emit_video).
#
# ⚠ For DERIVATIVE checkpoints (abliterations, fine-tunes) this default —
# the checkpoint's own repo — can re-fetch a STALE template: a quantizer
# snapshots its base repo at some commit, and if that snapshot predates the
# template fix, the broken file is what they ship AND what they serve over
# resolve/main. Point CHAT_TEMPLATE_URL at the BASE model's template instead:
#   CHAT_TEMPLATE_URL=https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/chat_template.jinja
# The emit_image guard below turns a stale fetch into a refusal, not a
# quietly text-blind multimodal deployment.
CHAT_TEMPLATE_URL="${CHAT_TEMPLATE_URL:-https://huggingface.co/${MODEL}/resolve/main/chat_template.jinja}"

refresh_chat_template() {
    local ref="$MODEL_PATH/refs/main" hash dest tmp
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run download first"
    hash="$(<"$ref")"
    dest="$MODEL_PATH/snapshots/$hash/chat_template.jinja"
    if [ "${SKIP_TEMPLATE_REFRESH:-0}" = "1" ]; then
        [ -f "$dest" ] || die "SKIP_TEMPLATE_REFRESH=1 but $dest is missing"
        grep -q 'emit_image' "$dest" \
            || die "SKIP_TEMPLATE_REFRESH=1 but cached chat_template.jinja is missing emit_image"
        log "SKIP_TEMPLATE_REFRESH=1 — using cached chat_template.jinja ($(wc -c < "$dest" | tr -d ' ') bytes)"
        return
    fi
    tmp="$(mktemp)"
    log "fetching chat_template.jinja from ${CHAT_TEMPLATE_URL} ..."
    curl -fsSL "$CHAT_TEMPLATE_URL" -o "$tmp" \
        || { rm -f "$tmp"; die "failed to download $CHAT_TEMPLATE_URL"; }
    grep -q 'emit_image' "$tmp" \
        || { rm -f "$tmp"; die "downloaded chat_template.jinja is missing emit_image — stale template (set CHAT_TEMPLATE_URL to the base model's)"; }
    if [ -L "$dest" ]; then
        cat "$tmp" > "$(readlink -f "$dest")"
    else
        mkdir -p "$(dirname "$dest")"
        cat "$tmp" > "$dest"
    fi
    rm -f "$tmp"
    log "chat_template.jinja updated ($(wc -c < "$dest" | tr -d ' ') bytes, emit_image+emit_video)"
    local real i
    real="$(readlink -f "$dest")"
    for i in $(each_worker); do
        worker_ssh "$i" "mkdir -p '$(w_cache "$i")/hub/$MODEL_CACHE_NAME/snapshots/$hash'"
        rsync -a "$real" "$(w_ssh_target "$i"):$(w_cache "$i")/hub/$MODEL_CACHE_NAME/snapshots/$hash/chat_template.jinja"
    done
    log "chat_template.jinja synced to all workers"
}

# ------------------------------ weight sync --------------------------------
sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to workers"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    local i
    for i in $(each_worker); do
        log "syncing weights to worker${i} (first run moves ~181 GiB) ..."
        worker_ssh "$i" "mkdir -p '$(w_cache "$i")/hub'"
        rsync -a --partial --info=progress2 \
            "$MODEL_PATH/" "$(w_ssh_target "$i"):$(w_cache "$i")/hub/${MODEL_CACHE_NAME}/"
    done
    log "all workers' weights in sync"
}

# ------------------------- derivative-checkpoint check ----------------------
# Derivative NVFP4 checkpoints (abliterations, fine-tunes) sometimes ship a
# SHORTER quantization_config.ignore list than the reference LibertAIDAI
# checkpoint — notably missing the vLLM-side FUSED module names
# (fused_qkvbfg_a_proj, qkv_proj, ...). vLLM matches ignore globs against its
# OWN module names, not the checkpoint's, so a missing fused glob can make
# the loader treat a BF16 module as NVFP4. Warn early; the fix (overlaying
# the reference list into the local snapshot config) is in
# docs/TROUBLESHOOTING.md.
check_quant_ignore() {
    local ref="$MODEL_PATH/refs/main" hash cfg
    [ -f "$ref" ] || return 0
    hash="$(<"$ref")"
    cfg="$MODEL_PATH/snapshots/$hash/config.json"
    [ -f "$cfg" ] || return 0
    python3 - "$cfg" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
ign = (cfg.get("quantization_config") or {}).get("ignore") or []
need = ["*.self_attn.fused_qkvbfg_a_proj", "*.self_attn.qkv_proj",
        "*.self_attn.fused_fg_b_proj", "*.self_attn.q_conv1d"]
missing = [n for n in need if n not in ign]
if missing:
    print(f"[glm53-tp3] WARNING: quantization_config.ignore lacks {len(missing)} "
          f"fused-module globs the reference checkpoint carries:")
    for n in missing:
        print(f"[glm53-tp3]     {n}")
    print("[glm53-tp3] if BF16 modules fail to load as NVFP4, overlay the reference")
    print("[glm53-tp3] (LibertAIDAI) ignore list -- see docs/TROUBLESHOOTING.md")
PYEOF
}

# ------------------------------ patch sync ---------------------------------
sync_patch() {
    [ "$TP3_PATCH" = "1" ] || { log "TP3_PATCH=0 — not shipping the patch"; return; }
    local i
    for i in $(each_worker); do
        # No '|| true' here, on purpose. A failed sync leaves a worker running
        # an OLD patch, and the only symptom is a shape assert on one rank —
        # or silently wrong output. A safeguard that cannot report its own
        # death is not a safeguard.
        rsync -a --delete "$TP3_PATCH_DIR/" "$(w_ssh_target "$i"):/tmp/glm53-tp3/" || {
            die "failed to sync the tp3 patch to worker${i} — refusing to boot TP=${TP}"
        }
    done
    log "tp3 patch synced to all workers"
}

# ------------------------ inner container scripts --------------------------
# Written to disk and mounted into the containers; config comes in via -e env.
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start.sh — runs INSIDE the head container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-head] $*"; }

# ---- TP=3 padding patch: BEFORE anything imports vllm ----
if [ "${TP3_PATCH:-1}" = "1" ]; then
    say "applying the TP=3 padding patch ..."
    python3 /opt/glm53-tp3/apply_tp3_patch.py || {
        say "FATAL: apply_tp3_patch.py failed — refusing to serve a mis-sharded model"
        exit 1
    }
fi

if ! command -v ray >/dev/null 2>&1; then
    say "ray not present in image — pip install ray[default]==${RAY_VERSION}"
    pip install -q --no-cache-dir "ray[default]==${RAY_VERSION}"
fi
say "ray $(ray --version 2>&1 | head -1)"

say "starting Ray head on ${HEAD_IP}:${RAY_PORT}"
ray start --head --port "${RAY_PORT}" --node-ip-address "${HEAD_IP}" \
    --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-4294967296}" \
    --dashboard-host 127.0.0.1 --disable-usage-stats

say "waiting for ${CLUSTER_SIZE} Ray node(s) to join ..."
n=0
for i in $(seq 1 "${CLUSTER_WAIT_ITERS}"); do
    n=$(python3 -c 'import ray; ray.init(logging_level="ERROR"); print(sum(nd["Alive"] for nd in ray.nodes()))' 2>/dev/null || echo 0)
    [ "${n:-0}" -ge "${CLUSTER_SIZE}" ] && break
    sleep 5
done
if [ "${n:-0}" -lt "${CLUSTER_SIZE}" ]; then
    say "FATAL: Ray cluster stuck at ${n:-0}/${CLUSTER_SIZE} node(s)"
    ray status || true
    exit 1
fi
say "Ray cluster ready (${n} node(s))"

# ---- model-card recipe (MiaAI's, kept verbatim) ----
ARGS=(
    --tensor-parallel-size "${TP}"
    --distributed-executor-backend ray
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --host 0.0.0.0
    --port "${PORT}"
)
if [ "${TRUST_REMOTE_CODE:-1}" = "1" ]; then
    ARGS+=(--trust-remote-code)
fi
if [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS}}")
fi
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${BLOCK_SIZE:-}" ]    && ARGS+=(--block-size "${BLOCK_SIZE}")
[ -n "${MAX_NUM_SEQS:-}" ]  && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
if [ -n "${KV_CACHE_DTYPE:-}" ]; then
    ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    say "kv-cache-dtype=${KV_CACHE_DTYPE}"
fi
if [ -n "${KV_CACHE_MEMORY:-}" ]; then
    ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
    say "kv-cache-memory=${KV_CACHE_MEMORY}"
fi
[ -n "${LIMIT_MM:-}" ]      && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
if [ -n "${MM_ENCODER_TP_MODE:-}" ]; then
    ARGS+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE}")
    say "mm-encoder-tp-mode=${MM_ENCODER_TP_MODE} (16 ViT heads % 3 != 0 -> replicate, don't shard)"
fi
if [ "${SKIP_MM_PROFILING:-1}" = "1" ]; then
    ARGS+=(--skip-mm-profiling)
    say "skip-mm-profiling: image+video serving on, no max-size MM dummy forward at init"
fi
ARGS+=(--chat-template "${MODEL_DIR}/chat_template.jinja")
say "chat-template: ${MODEL_DIR}/chat_template.jinja"
if [ "${ENFORCE_EAGER:-1}" = "1" ]; then
    ARGS+=(--enforce-eager)
    say "enforce-eager: no CUDA graph capture at init"
fi

if [ "${MOE_MODE:-native}" = "marlin" ]; then
    ARGS+=(--moe-backend marlin --enforce-eager)
    say "MoE backend: marlin (enforce-eager)"
else
    say "MoE backend: native NVFP4 kernels"
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

if [ ! -f "${MODEL_DIR}/processor_config.json" ]; then
    say "FATAL: ${MODEL_DIR}/processor_config.json missing — Glm5NextProcessor.from_pretrained() opens this as a local file (repo ids fail)."
    ls -la "${MODEL_DIR}" 2>/dev/null | head -n 30 || true
    exit 1
fi

say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]} (served-model-name=${MODEL})"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}" --served-model-name "${MODEL}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start.sh — runs INSIDE a worker container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-worker${WORKER_INDEX:-?}] $*"; }

# ---- TP=3 padding patch: BEFORE ray starts the executor that imports vllm ----
if [ "${TP3_PATCH:-1}" = "1" ]; then
    say "applying the TP=3 padding patch ..."
    python3 /opt/glm53-tp3/apply_tp3_patch.py || {
        say "FATAL: apply_tp3_patch.py failed — refusing to join as a mis-sharded rank"
        exit 1
    }
fi

if ! command -v ray >/dev/null 2>&1; then
    say "ray not present in image — pip install ray[default]==${RAY_VERSION}"
    pip install -q --no-cache-dir "ray[default]==${RAY_VERSION}"
fi
say "ray $(ray --version 2>&1 | head -1)"

say "joining Ray cluster at ${HEAD_IP}:${RAY_PORT} as ${WORKER_IP}"
for i in $(seq 1 "${CLUSTER_WAIT_ITERS}"); do
    if ray start --address "${HEAD_IP}:${RAY_PORT}" --node-ip-address "${WORKER_IP}" \
        --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-4294967296}" \
        --disable-usage-stats --block; then
        exit 0
    fi
    say "head not reachable yet (${HEAD_IP}:${RAY_PORT}), retrying in 5s ..."
    sleep 5
done
say "FATAL: could not join Ray cluster at ${HEAD_IP}:${RAY_PORT}"
exit 1
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
launch_cluster() {
    local moe_mode="$1" i

    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    for i in $(each_worker); do
        worker_ssh "$i" "docker rm -f '$(w_container "$i")'" >/dev/null 2>&1 || true
    done

    for i in $(each_worker); do
        scp -q -o BatchMode=yes "$WORKER_SCRIPT" "$(w_ssh_target "$i"):/tmp/$(w_container "$i").sh"
    done

    # Shared NCCL/RoCE pins (MiaAI's known-good set for GB10 CX7).
    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=()
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local -a head_patch_mount=()
    local worker_patch_mount=""
    if [ "$TP3_PATCH" = "1" ]; then
        head_patch_mount=(-v "$TP3_PATCH_DIR:/opt/glm53-tp3:ro")
        worker_patch_mount="-v /tmp/glm53-tp3:/opt/glm53-tp3:ro"
    fi

    for i in $(each_worker); do
        local worker_preload=""
        if [ "$USE_HOST_NCCL" = "1" ]; then
            if worker_ssh "$i" "test -f '$(w_nccl_dir "$i")/$NCCL_SO_NAME'"; then
                worker_preload="-v '$(w_nccl_dir "$i"):/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
                log "worker${i}: LD_PRELOAD $NCCL_SO_NAME"
            else
                warn "worker${i}: $(w_nccl_dir "$i")/$NCCL_SO_NAME missing — using image NCCL"
            fi
        fi
        log "starting worker${i} container on $(w_ssh_target "$i") (MoE mode: ${moe_mode}; NCCL if=$(w_if "$i") hca=$(w_ib "$i")) ..."
        worker_ssh "$i" "docker run -d --name '$(w_container "$i")' \
            --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
            --device /dev/infiniband --cap-add IPC_LOCK \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v '$(w_cache "$i"):/root/.cache/huggingface' \
            -v '$CACHE_VOLUME:/root/.cache' \
            -v '/tmp/$(w_container "$i").sh:/start.sh:ro' \
            ${worker_patch_mount} \
            ${worker_preload} \
            ${worker_nccl} \
            -e NCCL_SOCKET_IFNAME='$(w_if "$i")' \
            -e GLOO_SOCKET_IFNAME='$(w_if "$i")' \
            -e NCCL_IB_HCA='$(w_ib "$i")' \
            -e HEAD_IP='$HEAD_IP' -e RAY_PORT='$RAY_PORT' -e WORKER_IP='$(w_ip "$i")' \
            -e WORKER_INDEX='$i' \
            -e TP3_PATCH='$TP3_PATCH' \
            -e RAY_VERSION='$RAY_VERSION' \
            -e CLUSTER_WAIT_ITERS=$CLUSTER_WAIT_ITERS \
            -e VLLM_HOST_IP='$(w_ip "$i")' \
            -e RAY_OBJECT_STORE_MEMORY='$RAY_OBJECT_STORE_MEMORY' \
            -e RAY_memory_usage_threshold='$RAY_memory_usage_threshold' \
            -e VLLM_ENGINE_READY_TIMEOUT_S='$READY_TIMEOUT' \
            --entrypoint bash '$IMAGE' /start.sh" >/dev/null
    done

    log "starting head container (Ray head + vLLM API server; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$CACHE_VOLUME:/root/.cache" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        "${head_patch_mount[@]}" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e HEAD_IP="$HEAD_IP" -e RAY_PORT="$RAY_PORT" \
        -e TP3_PATCH="$TP3_PATCH" \
        -e RAY_VERSION="$RAY_VERSION" \
        -e CLUSTER_SIZE="$TP" -e CLUSTER_WAIT_ITERS="$CLUSTER_WAIT_ITERS" \
        -e MODEL="$MODEL" -e MODEL_DIR="$MODEL_DIR" -e TP="$TP" -e PORT="$PORT" -e MTP_TOKENS="$MTP_TOKENS" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e BLOCK_SIZE="$BLOCK_SIZE" -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e KV_CACHE_MEMORY="$KV_CACHE_MEMORY" \
        -e TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
        -e LIMIT_MM="$LIMIT_MM" -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e MM_ENCODER_TP_MODE="$MM_ENCODER_TP_MODE" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e MOE_MODE="$moe_mode" -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
        -e RAY_memory_usage_threshold="$RAY_memory_usage_threshold" \
        -e VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

    log "containers up — head=${CONTAINER_HEAD}, workers=$(for i in $(each_worker); do printf '%s ' "$(w_container "$i")"; done)"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on a 320B MoE is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — containers keep running in the background"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            log "head container exited during startup"
            exited=1; break
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running in the background"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "head container exited after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

# --------------------------- failure logs ----------------------------------
collect_failure_logs() {
    local tag="$1" i
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head-${tag}.log" 2>&1 || true
    for i in $(each_worker); do
        {
            echo "### docker logs $(w_container "$i")"
            worker_ssh "$i" "docker logs '$(w_container "$i")' 2>&1" || true
            echo
            echo "### worker${i} Ray session logs (filtered for CUDA kernel-image errors)"
            worker_ssh "$i" "docker exec '$(w_container "$i")' sh -c 'grep -rhE \"$KERNEL_ERR_PAT\" /tmp/ray/session_latest/logs/ 2>/dev/null | head -n 40'" || true
        } >"$LOGDIR/worker${i}-${tag}.log" 2>&1 || true
    done
}

# ------------------------------ on ready -----------------------------------
on_ready() {
    local mode="$1" how="$2"
    log "======================================================================"
    log "GLM-5.3-Flash-NVFP4 is UP at TP=${TP} (${how}; MoE backend: ${mode})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN ips: $(hostname -I))"
    log "  model name : ${MODEL}"
    log "  features   : tools=glm47+auto, reasoning=glm45, MTP spec-decode (${MTP_TOKENS} tokens), image+video"
    log "  IMPORTANT  : 'it starts' is not 'it is correct' on a padded TP shape."
    log "               Run scripts/validate_tp3.sh 127.0.0.1:${PORT} before trusting output."
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hello!\"}]}'"
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker1 | ./start.sh stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted"; exit 130' INT
        log "detached from logs; server still running"
    fi
}

# ------------------------------- start -------------------------------------
start() {
    preflight
    pull_images
    download_weights
    refresh_chat_template
    sync_weights
    sync_patch
    write_inner_scripts

    MODEL_DIR="$(resolve_model_dir)"
    log "model load path (in-container): ${MODEL_DIR}"
    check_quant_ignore

    local mode="native"
    case "$MOE_BACKEND" in
        native) mode="native" ;;
        marlin) mode="marlin" ;;
        auto)   mode="native" ;;
        *) die "MOE_BACKEND must be auto | native | marlin (got: ${MOE_BACKEND})" ;;
    esac

    log "config: image=${IMAGE} tp=${TP} workers=${N_WORKERS} first-attempt=${mode} mtp=${MTP_TOKENS}" \
        "max-len=${MAX_MODEL_LEN:-<model default>} gpu-util=${GPU_MEM_UTIL} block=${BLOCK_SIZE} kv=${KV_CACHE_DTYPE} patch=${TP3_PATCH} port=${PORT}"

    launch_cluster "$mode"
    if wait_for_health; then
        on_ready "$mode" "first attempt"
        return
    fi

    collect_failure_logs "$mode"
    echo "---- last 60 lines of head log ($LOGDIR/head-${mode}.log) ----"
    tail -n 60 "$LOGDIR/head-${mode}.log" || true

    if [ "$MOE_BACKEND" = "auto" ] && [ "$mode" = "native" ] \
       && grep -qE "$KERNEL_ERR_PAT" "$LOGDIR"/head-native.log "$LOGDIR"/worker*-native.log 2>/dev/null; then
        warn "cudaErrorNoKernelImageForDevice from the native FP4 MoE kernels (expected on sm_121) —"
        warn "falling back to marlin MoE backend: --moe-backend marlin --enforce-eager"
        launch_cluster "marlin"
        if wait_for_health; then
            on_ready "marlin" "after sm_121 fallback"
            return
        fi
        collect_failure_logs "marlin"
        die "server failed in marlin mode too — full logs in $LOGDIR/"
    fi
    die "server did not become healthy — full logs in $LOGDIR/ (read the FIRST-dying rank's log; the others report the consequence)"
}

# ------------------------------- stop --------------------------------------
stop() {
    local i
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    for i in $(each_worker); do
        log "stopping worker${i} container on $(w_ssh_target "$i") ..."
        worker_ssh "$i" "docker rm -f '$(w_container "$i")'" >/dev/null 2>&1 \
            || log "  (no worker${i} container was running)"
    done
    log "stopped."
}

# ------------------------------ status -------------------------------------
status() {
    local i
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
        docker exec "$CONTAINER_HEAD" ray status 2>/dev/null | sed 's/^/  /' | head -n 25 || true
    fi
    for i in $(each_worker); do
        log "worker${i} ($(w_container "$i") on $(w_ssh_target "$i")):"
        worker_ssh "$i" "docker ps -a --filter name=$(w_container "$i") --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
            || log "  (worker${i} unreachable)"
    done
}

# ------------------------------- logs --------------------------------------
logs() {
    local which="${1:-head}"
    case "$which" in
        worker*)
            local i="${which#worker}"
            [ -n "$i" ] && [ "$i" != "worker" ] || i=1
            log "following worker${i} container logs on $(w_ssh_target "$i") ..."
            log "(per-rank engine stdout lives inside the container: /tmp/ray/session_latest/logs/worker-*.out)"
            trap '' INT
            worker_ssh "$i" "docker logs -f --tail 100 '$(w_container "$i")'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

# ------------------------------- main --------------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start)   shift || true; start ;;
        stop)    stop ;;
        restart) stop; start ;;
        status)  status ;;
        logs)    shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
