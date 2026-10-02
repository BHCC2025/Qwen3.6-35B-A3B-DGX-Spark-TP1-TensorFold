#!/usr/bin/env bash
# recipes/tp1.sh — Qwen3.6-35B-A3B on ONE DGX Spark (TP1), TensorFold. Normally started via ./run.sh tp1.
#
# The MLX 4-bit checkpoint (19.5 GiB of weights) with its own MTP layer as the drafter, so there is no draft model.
# PARALLEL=8 decodes up to eight requests in shared rounds (each reply equal to the request alone); a lone request
# replays the one-stream CUDA graphs, so it is as fast as with PARALLEL=1. Memory profile by MEM_GB (default: from
# /proc/meminfo, 64 below 80 GiB):
#   128  CONTEXT=auto: TensorFold sizes the window to what it can afford (the native 262,144 tokens a stream at 8
#        streams on a 128 GB Spark; startup estimate 95.6 GiB if every stream fills its window).
#   64   PARALLEL=4, CONTEXT=65536: startup estimate 34.7 GiB, so a 64 GB Spark keeps a wide margin for the OS.
# PARALLEL and CONTEXT set explicitly always win over the profile.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

if [ "${MEM_GB:-auto}" = auto ]; then
  MEM_GB=$(awk '/^MemTotal:/ {print ($2 < 80 * 1048576) ? 64 : 128}' /proc/meminfo)
fi
case "$MEM_GB" in
  128) PARALLEL="${PARALLEL:-8}"; CONTEXT="${CONTEXT:-auto}" ;;
  64)  PARALLEL="${PARALLEL:-4}"; CONTEXT="${CONTEXT:-65536}" ;;
  *)   echo "MEM_GB must be auto, 128 or 64" >&2; exit 2 ;;
esac
build_all

check_model "$MODEL_DIR"; [ "${DRAFTS:-1}" = 0 ] || check_model "$DRAFT_DIR"
run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR:/models/qwen36-35b:ro" -v "$CACHE_DIR:/root/.cache" "${SPEC_MOUNT[@]}" \
  "${BASE_ENV[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    tensorfold serve /models/qwen36-35b "${ENDPOINT_ARGS[@]}" "${SERVE_ARGS[@]}" "${SPEC_ARGS[@]}" ${EXTRA:-}
[ "${DRY_RUN:-0}" = 1 ] || echo "launched $NAME tp=1 mem_gb=$MEM_GB parallel=$PARALLEL context=${CONTEXT} drafts=${DRAFTS:-1}"
