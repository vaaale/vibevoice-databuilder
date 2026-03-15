#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IMAGE="${VIBEVOICE_IMAGE:-vibevoice-databuilder}"

usage() {
    cat <<'EOF'
Usage: databuilder.sh <command> [args...]

Docker wrapper for VibeVoice tools. Automatically mounts host paths into the container.

Commands:
  databuilder  INPUT_DIR REPO_ID [options]
      Build a speech dataset. INPUT_DIR is mounted read-only at /app/input.
      If REPO_ID is a local path it is mounted read-write at /app/output.

  enhance batch INPUT_DIR [--output-dir DIR] [options]
      Batch-enhance all audio files in INPUT_DIR.

  enhance sweep INPUT_FILE [--output-dir DIR] [options]
      Parameter sweep on a single audio file.

Environment:
  VIBEVOICE_IMAGE   Docker image name (default: vibevoice-data)

Examples:
  databuilder.sh databuilder /data/raw ./output/dataset --device cuda --batch-size 32
  databuilder.sh enhance batch /data/input --output-dir /data/output --device cuda
  databuilder.sh enhance sweep /data/file.wav --output-dir ./sweep_out --nfe 8,16,32,64
EOF
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

resolve_path() {
    local p="$1"
    if [[ "$p" != /* ]]; then
        p="$(pwd)/$p"
    fi
    # Normalise via dirname/basename (works even if path doesn't exist yet)
    local dir base
    dir="$(cd "$(dirname "$p")" 2>/dev/null && pwd)" || dir="$(dirname "$p")"
    base="$(basename "$p")"
    echo "${dir}/${base}"
}

# Options that consume the next token as a value
VALUED_OPTS=(
    --model-id --device --hf-token --speaker-prefix --max-duration
    --max-num-speakers --batch-size --chunk-size --work-dir --output-dir
    --nfe --lambd --tau
)

is_valued_opt() {
    local opt="$1"
    for v in "${VALUED_OPTS[@]}"; do
        [[ "$opt" == "$v" ]] && return 0
    done
    return 1
}

is_local_path() {
    local arg="$1"
    [[ "$arg" == /* || "$arg" == ./* || "$arg" == ../* ]]
}

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

declare -a VOLUMES=()
declare -a DOCKER_EXTRA=("--ipc=host" "--ulimit" "stack=67108864" "-v" "$HOME/.cache/huggingface:/root/.cache/huggingface")
declare -a CONTAINER_ARGS=()

add_volume() {
    local host_path container_path mode
    host_path="$(resolve_path "$1")"
    container_path="$2"
    mode="${3:-rw}"
    # Create host directory if needed (for output mounts)
    if [[ "$mode" == "rw" && ! -e "$host_path" ]]; then
        mkdir -p "$host_path"
    fi
    VOLUMES+=("-v" "${host_path}:${container_path}:${mode}")
}

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

handle_databuilder() {
    local -a positionals=()
    local -a args=("$@")
    local i=0

    while [[ $i -lt ${#args[@]} ]]; do
        local arg="${args[$i]}"
        if is_valued_opt "$arg" && [[ $((i + 1)) -lt ${#args[@]} ]]; then
            local val="${args[$((i + 1))]}"
            if [[ "$arg" == "--work-dir" ]]; then
                add_volume "$val" "/app/work" "rw"
                CONTAINER_ARGS+=("$arg" "/app/work")
            else
                CONTAINER_ARGS+=("$arg" "$val")
            fi
            i=$((i + 2))
        elif [[ "$arg" == --* ]]; then
            CONTAINER_ARGS+=("$arg")
            i=$((i + 1))
        else
            positionals+=("$arg")
            i=$((i + 1))
        fi
    done

    if [[ ${#positionals[@]} -lt 2 ]]; then
        echo "Error: databuilder requires INPUT_DIR and REPO_ID" >&2
        echo "Usage: databuilder.sh databuilder INPUT_DIR REPO_ID [options]" >&2
        exit 1
    fi

    local input_dir="${positionals[0]}"
    local repo_id="${positionals[1]}"

    add_volume "$input_dir" "/app/input" "ro"

    if is_local_path "$repo_id"; then
        add_volume "$repo_id" "/app/output" "rw"
        CONTAINER_ARGS+=("/app/input" "/app/output")
    else
        CONTAINER_ARGS+=("/app/input" "$repo_id")
    fi
}

handle_enhance() {
    if [[ $# -eq 0 ]]; then
        echo "Error: enhance requires a subcommand (batch or sweep)" >&2
        exit 1
    fi

    local subcmd="$1"; shift
    CONTAINER_ARGS+=("$subcmd")

    local -a positionals=()
    local -a args=("$@")
    local i=0

    while [[ $i -lt ${#args[@]} ]]; do
        local arg="${args[$i]}"
        if is_valued_opt "$arg" && [[ $((i + 1)) -lt ${#args[@]} ]]; then
            local val="${args[$((i + 1))]}"
            if [[ "$arg" == "--output-dir" ]]; then
                add_volume "$val" "/app/output" "rw"
                CONTAINER_ARGS+=("$arg" "/app/output")
            else
                CONTAINER_ARGS+=("$arg" "$val")
            fi
            i=$((i + 2))
        elif [[ "$arg" == --* ]]; then
            CONTAINER_ARGS+=("$arg")
            i=$((i + 1))
        else
            positionals+=("$arg")
            i=$((i + 1))
        fi
    done

    case "$subcmd" in
        batch)
            if [[ ${#positionals[@]} -lt 1 ]]; then
                echo "Error: enhance batch requires INPUT_DIR" >&2
                exit 1
            fi
            add_volume "${positionals[0]}" "/app/input" "ro"
            CONTAINER_ARGS+=("/app/input")
            ;;
        sweep)
            if [[ ${#positionals[@]} -lt 1 ]]; then
                echo "Error: enhance sweep requires INPUT_FILE" >&2
                exit 1
            fi
            local input_file="${positionals[0]}"
            local host_dir
            host_dir="$(dirname "$(resolve_path "$input_file")")"
            local filename
            filename="$(basename "$input_file")"
            VOLUMES+=("-v" "${host_dir}:/app/input:ro")
            CONTAINER_ARGS+=("/app/input/${filename}")
            ;;
        *)
            echo "Error: Unknown enhance subcommand '$subcmd'" >&2
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]] || [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    usage
    exit 0
fi

COMMAND="$1"; shift

# Use GPU if available
if command -v nvidia-smi &>/dev/null; then
    DOCKER_EXTRA+=("--gpus" "all")
fi

case "$COMMAND" in
    databuilder)
        handle_databuilder "$@"
        ;;
    enhance)
        handle_enhance "$@"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'" >&2
        usage >&2
        exit 1
        ;;
esac

CMD=(docker run --rm "${DOCKER_EXTRA[@]}" "${VOLUMES[@]}" "$IMAGE" "$COMMAND" "${CONTAINER_ARGS[@]}")

echo ">>> ${CMD[*]}"
exec "${CMD[@]}"
