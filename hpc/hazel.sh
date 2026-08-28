#!/usr/bin/env bash
# Helper for working with NC State's Hazel cluster from this repo.
#
#   ./hpc/hazel.sh connect          open a shell on the login node
#   ./hpc/hazel.sh push             upload code (rsync, no GitHub auth needed)
#   ./hpc/hazel.sh pull             download results/ into ./results
#   ./hpc/hazel.sh submit <py> [gpu] [hrs]   sbatch a script (default a30, 4h)
#   ./hpc/hazel.sh gpu [hours]      interactive L40S shell (fp64-poor; debug only)
#   ./hpc/hazel.sh smoke            push + submit the GPU smoke test
#   ./hpc/hazel.sh status           your queued/running jobs
#   ./hpc/hazel.sh log <jobid>      follow a job's output
#   ./hpc/hazel.sh gpus             what GPUs are free right now
#
# Remote directory is auto-detected on first use and cached in hpc/.hazel_env.
# Override with:  export HAZEL_REMOTE_DIR=/share/<group>/smohapa5/hybridlane-benchmarking

set -euo pipefail

HOST=hazel
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$REPO_ROOT/hpc/.hazel_env"

resolve_remote_dir() {
    if [[ -n "${HAZEL_REMOTE_DIR:-}" ]]; then
        REMOTE_DIR="$HAZEL_REMOTE_DIR"
        return
    fi
    if [[ -f "$CACHE" ]]; then
        # shellcheck disable=SC1090
        source "$CACHE"
        [[ -n "${REMOTE_DIR:-}" ]] && return
    fi
    echo "Detecting your project directory on Hazel (Duo may prompt)..." >&2
    REMOTE_DIR=$(ssh "$HOST" 'echo /share/$(id -gn)/$USER/hybridlane-benchmarking')
    printf 'REMOTE_DIR=%s\n' "$REMOTE_DIR" > "$CACHE"
    echo "Using $REMOTE_DIR  (cached; override with HAZEL_REMOTE_DIR)" >&2
}

cmd_connect() { exec ssh "$HOST"; }

cmd_push() {
    resolve_remote_dir
    ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
    rsync -avh --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude 'results' --exclude '.venv' --exclude '.hazel_env' \
        --exclude '*.npz' --exclude '*.npy' --exclude '*.png' \
        "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"
}

cmd_pull() {
    resolve_remote_dir
    mkdir -p "$REPO_ROOT/results"
    rsync -avhP "$HOST:$REMOTE_DIR/results/" "$REPO_ROOT/results/"
}

# Interactive GPU shell. Hazel forces interactive jobs onto QOS short_gpu, which
# is tied to gpu_partners -- and that partition only has A10/L40S, both running
# fp64 at ~1/32 rate. Fine for checking imports and shapes; useless for timing
# the float64 propagator. Use `submit` for anything numeric.
cmd_gpu() {
    local hours="${1:-1}"
    (( hours > 2 )) && { echo "short_gpu caps interactive at 2h; clamping." >&2; hours=2; }
    echo "Interactive L40S for ${hours}h -- fp64-crippled, debugging only."
    exec ssh -t "$HOST" "salloc --ntasks=4 --partition=gpu_partners --qos=short_gpu \
        --gres=gpu:l40s:1 --mem=32G --time=${hours}:00:00"
}

# Batch submission -- the only route to fp64-capable cards, since batch jobs may
# use any QOS. Defaults to A30: 9.5 TFLOP/s fp64 via tensor cores, 8 cards, and
# effectively never contended (vs 4 A100s on one node).
cmd_submit() {
    resolve_remote_dir
    local script="${1:?usage: hazel.sh submit <script.py> [gpu_type] [hours]}"
    local gpu="${2:-a30}" hours="${3:-4}"
    local name; name=$(basename "$script" .py)
    ssh "$HOST" "cd '$REMOTE_DIR' && mkdir -p logs && sbatch \
        --partition=gpu --gres=gpu:${gpu}:1 --ntasks=4 --mem=32G \
        --time=${hours}:00:00 --export=ALL,JAX_ENABLE_X64=1 \
        --output=logs/%x_%j.out --job-name='${name}' \
        --wrap='python ${script}'"
}

cmd_smoke() {
    cmd_push
    cmd_submit hpc/gpu_smoke_test.py a30 1
    echo "Watch with: $0 status    Read with: $0 log <jobid>"
}

cmd_status() { ssh "$HOST" 'squeue -u $USER'; }

cmd_log() {
    resolve_remote_dir
    local jobid="${1:?usage: hazel.sh log <jobid>}"
    ssh -t "$HOST" "tail -f '$REMOTE_DIR'/logs/*${jobid}*.out"
}

cmd_gpus() { ssh "$HOST" 'si --gpus'; }

case "${1:-connect}" in
    connect) cmd_connect ;;
    push)    cmd_push ;;
    pull)    cmd_pull ;;
    gpu)     cmd_gpu "${2:-1}" ;;
    smoke)   cmd_smoke ;;
    submit)  cmd_submit "${2:-}" "${3:-}" "${4:-}" ;;
    status)  cmd_status ;;
    log)     cmd_log "${2:-}" ;;
    gpus)    cmd_gpus ;;
    *)       sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
