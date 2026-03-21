#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: run.sh <command> [args...]

Commands:
  databuilder    Run the dataset builder pipeline (Whisper ASR + diarization)
  enhance        Run audio enhancement (batch or sweep mode)
  export         Export / analyse / merge datasets (subcommands: analyse, merge, export)
  stortinget     Build a dataset from the NPSC Stortinget V1.0 corpus
  help           Show this help message

Examples:
  # Dataset builder pipeline
  run.sh databuilder /data/input my-org/my-dataset --device cuda

  # Enhance all files in a directory
  run.sh enhance batch /data/input --output-dir /data/output --device cuda --batch-size 4

  # Parameter sweep on a single file
  run.sh enhance sweep /data/input/file.wav --output-dir /data/output --nfe 8,16,32,64 --lambd 0.5 --tau 0.5

  # Export a dataset to JSONL
  run.sh export export /data/dataset --output-path /data/export --full

  # Build Stortinget dataset
  run.sh stortinget --input-path /data/stortinget --output-path /data/output --device cuda

  # Pass --help to any command for full options
  run.sh databuilder --help
  run.sh enhance --help
EOF
}

if [ $# -eq 0 ] || [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
    exit 0
fi

COMMAND="$1"
shift

cd /app

case "$COMMAND" in
    databuilder)
        exec ./.venv/bin/python -m databuilder "$@"
        ;;
    enhance)
        exec ./.venv/bin/python -m databuilder.run_enhance_dir "$@"
        ;;
    export)
        exec ./.venv/bin/python -m databuilder.export "$@"
        ;;
    stortinget)
        exec ./.venv/bin/python -m databuilder.stortinget_v1 "$@"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'" >&2
        echo "" >&2
        usage >&2
        exit 1
        ;;
esac
