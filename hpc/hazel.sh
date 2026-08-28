#!/usr/bin/env bash
# Helper for working with NC State's Hazel cluster from this repo.
#
#   ./hpc/hazel.sh connect          open a shell on the login node
#   ./hpc/hazel.sh push             upload code (rsync, no GitHub auth needed)
#   ./hpc/hazel.sh pull             download results/ into ./results
#   ./hpc/hazel.sh gpu [hours]      interactive A100 session (default 1h)
#   ./hpc/hazel.sh smoke            push + run the GPU smoke test on an A100
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

cmd_gpu() {
    local hours="${1:-1}"
    echo "Requesting 1x A100 for ${hours}h. This waits in the queue, then drops you"
    echo "into a shell ON the GPU node. Type 'exit' to release it."
    exec ssh -t "$HOST" \
        "salloc --ntasks=4 --partition=gpu --gres=gpu:a100:1 --mem=32G --time=${hours}:00:00"
}

cmd_smoke() {
    resolve_remote_dir
    cmd_push
    echo "Running smoke test on an A100 (queue wait, then ~30s)..."
    ssh -t "$HOST" "cd '$REMOTE_DIR' && srun --ntasks=1 --partition=gpu \
        --gres=gpu:a100:1 --mem=16G --time=00:15:00 \
        python hpc/gpu_smoke_test.py"
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
    status)  cmd_status ;;
    log)     cmd_log "${2:-}" ;;
    gpus)    cmd_gpus ;;
    *)       sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
