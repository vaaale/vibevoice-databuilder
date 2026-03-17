"""Console entry point powered by Click."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click

from .pipeline import run_pipeline


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, resolve_path=True),
)
@click.argument("repo_id")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for intermediate files (defaults to ./output).",
)
@click.option("--model-id", default="openai/whisper-base", show_default=True)
@click.option("--device", default=None, metavar="DEVICE", help="Force compute device (cuda, mps, cpu).")
@click.option(
    "--hf-token",
    default=None,
    envvar="HUGGINGFACEHUB_API_TOKEN",
    help="Hugging Face token (uses HUGGINGFACEHUB_API_TOKEN if unset).",
)
@click.option(
    "--speaker-prefix",
    default="Speaker 0: ",
    show_default=True,
    help="Prefix prepended to each transcript.",
)
@click.option(
    "--max-duration",
    type=float,
    default=25,
    show_default=True,
    help="Maximum duration (seconds) to process per input file (trims audio beyond this length).",
)
@click.option(
    "--max-num-speakers",
    type=int,
    default=4,
    show_default=True,
    help="Optional upper bound for diarization speaker count.",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Optional batch size.",
)
@click.option(
    "--chunk-size",
    type=float,
    default=300.0,
    show_default=True,
    help="Maximum duration in seconds for VAD-built diarization chunks.",
)

def main(
    input_dir: Path,
    repo_id: str,
    work_dir: Optional[Path],
    model_id: str,
    device: Optional[str],
    hf_token: Optional[str],
    speaker_prefix: str,
    max_duration: Optional[float],
    max_num_speakers: Optional[int],
    batch_size: int,
    chunk_size: float,
) -> None:
    """Segment speech, run Whisper ASR, and push a dataset to the HF Hub."""
    start = time.perf_counter()

    if repo_id.startswith("/") or repo_id.startswith("./") or Path(repo_id).exists():
        repo_id = Path(repo_id).resolve()
        resolved_work_dir = repo_id
    else:
        resolved_work_dir = (work_dir or Path.cwd() / "output").expanduser().resolve()

    input_dir = input_dir.expanduser().resolve()

    if not input_dir.exists():  # Defensive: Click already checks, but keep for clarity
        raise click.UsageError(f"Input directory not found: {input_dir}")
    dataset = run_pipeline(
        input_dir=input_dir,
        repo_id=repo_id,
        work_dir=resolved_work_dir,
        model_id=model_id,
        device=device,
        token=hf_token,
        speaker_prefix=speaker_prefix,
        enable_diarization=True,
        max_duration=max_duration,
        max_num_speakers=max_num_speakers,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    click.echo(f"Uploaded {len(dataset)} records to {repo_id}")
    click.echo(f"Total time: {time.perf_counter() - start:.2f} seconds")

    # # Print 4 random samples from dataset
    # debug_samples = [s["text"] for s in dataset.select(range(40))]
    # for sample in dataset.shuffle().select(range(4)):
    #     click.echo(sample)
