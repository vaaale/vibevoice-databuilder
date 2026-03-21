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
  databuilder  INPUT_DIR OUTPUT_DIR [options]
      Build a speech dataset. INPUT_DIR is mounted read-only at /app/input.
      OUTPUT_DIR is mounted read-write at /app/output.

  enhance batch INPUT_DIR OUTPUT_DIR [options]
      Batch-enhance all audio files in INPUT_DIR.

  enhance sweep INPUT_FILE OUTPUT_DIR [options]
      Parameter sweep on a single audio file.

  export <subcommand> DATASET_PATHS... [options]
      Export / analyse / merge datasets.
      Subcommands: analyse, merge, export

  stortinget [options]
      Build a dataset from the NPSC Stortinget V1.0 corpus.

Environment:
  VIBEVOICE_IMAGE   Docker image name (default: vibevoice-data)

Examples:
  databuilder.sh databuilder /data/raw ./output/dataset --device cuda --batch-size 32
  databuilder.sh enhance batch /data/input --output-dir /data/output --device cuda
  databuilder.sh enhance sweep /data/file.wav --output-dir ./sweep_out --nfe 8,16,32,64
  databuilder.sh export export /data/dataset --output-path ./export_out --full
  databuilder.sh stortinget --input-path /data/stortinget --output-path ./stortinget_out --device cuda
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
    --nfe --lambd --tau --uid --gid
    --output-path --input-path --whisper-model --num-workers
    --num-audio-workers --limit --min-duration --split --jsonl
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
declare -a DOCKER_EXTRA=("--ipc=host" "--ulimit" "stack=67108864" "-v" "${HOME}/.cache/huggingface:/root/.cache/huggingface")
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
        echo "Error: databuilder requires INPUT_DIR and OUTPUT_DIR" >&2
        echo "Usage: databuilder.sh databuilder INPUT_DIR OUTPUT_DIR [options]" >&2
        exit 1
    fi

    local input_dir="${positionals[0]}"
    local output_dir="${positionals[1]}"

#    add_volume "$input_dir" "/app/input" "ro"
#    add_volume "$output_dir" "/app/output" "rw"
#    CONTAINER_ARGS+=("/app/input" "/app/output")
    add_volume "$input_dir" "$input_dir" "ro"
    add_volume "$output_dir" "$output_dir" "rw"
    CONTAINER_ARGS+=("$input_dir" "$output_dir")

    # Pass current user's uid/gid so the container can chown output files
    CONTAINER_ARGS+=("--uid" "$(id -u)" "--gid" "$(id -g)")
}

handle_export() {
    if [[ $# -eq 0 ]]; then
        echo "Error: export requires a subcommand (analyse, merge, export)" >&2
        exit 1
    fi

    local subcmd="$1"; shift
    CONTAINER_ARGS+=("$subcmd")

    local -a args=("$@")
    local i=0

    while [[ $i -lt ${#args[@]} ]]; do
        local arg="${args[$i]}"
        if is_valued_opt "$arg" && [[ $((i + 1)) -lt ${#args[@]} ]]; then
            local val="${args[$((i + 1))]}"
            if [[ "$arg" == "--output-path" ]]; then
                add_volume "$val" "$val" "rw"
                CONTAINER_ARGS+=("$arg" "$val")
            else
                CONTAINER_ARGS+=("$arg" "$val")
            fi
            i=$((i + 2))
        elif [[ "$arg" == --* ]]; then
            CONTAINER_ARGS+=("$arg")
            i=$((i + 1))
        else
            # Positional arg (dataset path) — mount if local
            if is_local_path "$arg"; then
                local resolved
                resolved="$(resolve_path "$arg")"
                add_volume "$resolved" "$resolved" "ro"
                CONTAINER_ARGS+=("$resolved")
            else
                CONTAINER_ARGS+=("$arg")
            fi
            i=$((i + 1))
        fi
    done
}

handle_stortinget() {
    local -a args=("$@")
    local i=0

    while [[ $i -lt ${#args[@]} ]]; do
        local arg="${args[$i]}"
        if is_valued_opt "$arg" && [[ $((i + 1)) -lt ${#args[@]} ]]; then
            local val="${args[$((i + 1))]}"
            case "$arg" in
                --input-path)
                    add_volume "$val" "$val" "ro"
                    CONTAINER_ARGS+=("$arg" "$val")
                    ;;
                --output-path|--work-dir)
                    add_volume "$val" "$val" "rw"
                    CONTAINER_ARGS+=("$arg" "$val")
                    ;;
                *)
                    CONTAINER_ARGS+=("$arg" "$val")
                    ;;
            esac
            i=$((i + 2))
        elif [[ "$arg" == --* ]]; then
            CONTAINER_ARGS+=("$arg")
            i=$((i + 1))
        else
            CONTAINER_ARGS+=("$arg")
            i=$((i + 1))
        fi
    done
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
                add_volume "$val" "$val" "rw"
                CONTAINER_ARGS+=("$arg" "$val")
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
            if [[ ${#positionals[@]} -lt 2 ]]; then
                echo "Error: enhance batch requires INPUT_DIR OUTPUT_DIR" >&2
                exit 1
            fi
            add_volume "${positionals[0]}" "${positionals[0]}" "ro"
            add_volume "${positionals[1]}" "${positionals[1]}" "rw"
            CONTAINER_ARGS+=("${positionals[0]}" "--output-dir" "${positionals[1]}")
            ;;
        sweep)
            if [[ ${#positionals[@]} -lt 2 ]]; then
                echo "Error: enhance sweep requires INPUT_FILE OUTPUT_DIR" >&2
                exit 1
            fi
            local input_file="${positionals[0]}"
            local resolved
            resolved="$(resolve_path "$input_file")"
            local host_dir
            host_dir="$(dirname "$resolved")"
            local filename
            filename="$(basename "$resolved")"
            add_volume "$host_dir" "$host_dir" "ro"
            add_volume "${positionals[1]}" "${positionals[1]}" "rw"
            CONTAINER_ARGS+=("$host_dir/${filename}" "--output-dir" "${positionals[1]}")
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
    export)
        handle_export "$@"
        ;;
    stortinget)
        handle_stortinget "$@"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'" >&2
        usage >&2
        exit 1
        ;;
esac

CMD=(docker run --rm "${DOCKER_EXTRA[@]}" "${VOLUMES[@]}" "$IMAGE" "$COMMAND" "${CONTAINER_ARGS[@]}")

printf '>>> %s\n' "$(printf '%q ' "${CMD[@]}")"
exec "${CMD[@]}"
