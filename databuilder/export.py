"""VibeVoice dataset export utilities: analyse, merge, and export to JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm.auto import tqdm


# ── Dataset loading helpers ───────────────────────────────────────────────

def _is_local_dataset(path_str: str) -> bool:
    """Return True when *path_str* points to a local directory."""
    return Path(path_str).is_dir()


def _load_single(path_str: str) -> Dataset:
    if _is_local_dataset(path_str):
        click.echo(f"  Loading local dataset: {path_str}")
        return load_from_disk(path_str)
    click.echo(f"  Loading HF dataset: {path_str}")
    return load_dataset(path_str, split="train")


def _load_and_concat(paths: tuple[str, ...]) -> Dataset:
    datasets_list = [_load_single(p) for p in paths]
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)


def _display_name(paths: tuple[str, ...]) -> str:
    parts = []
    for p in paths:
        parts.append(Path(p).name if _is_local_dataset(p) else p.replace("/", "_"))
    return "_".join(parts)


# ── Audio file helpers ────────────────────────────────────────────────────

def _audio_filename(audio, idx: int, *, prompt_index: int | None = None) -> str:
    """Extract or generate a .wav filename from a decoded HF Audio dict."""
    path = audio.get("path") if isinstance(audio, dict) else None
    if path:
        name = Path(path).name
        if name and name != ".":
            return name
    if prompt_index is not None:
        return f"prompt_{idx:06d}_{prompt_index:03d}.wav"
    return f"sample_{idx:06d}.wav"


def _write_audio(audio, dst: Path) -> None:
    """Write a decoded HF Audio dict to a WAV file."""
    import soundfile as sf

    if not isinstance(audio, dict):
        return
    arr = audio.get("array")
    sr = audio.get("sampling_rate")
    if arr is not None and sr is not None:
        sf.write(str(dst), arr, sr)


# ── CLI ───────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """VibeVoice dataset export utilities."""


@cli.command()
@click.argument("dataset_paths", nargs=-1, required=True)
@click.option("--min-duration", type=float, default=None, help="Exclude samples shorter than this duration (seconds).")
@click.option("--max-duration", type=float, default=None, help="Exclude samples longer than this duration (seconds).")
def analyse(dataset_paths: tuple[str, ...], min_duration: float | None, max_duration: float | None):
    """Show sample count, duration histogram, and min/max/mean duration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        click.echo(
            "matplotlib is required for the analyse command.\n"
            "Install it with:  pip install matplotlib",
            err=True,
        )
        sys.exit(1)

    ds = _load_and_concat(dataset_paths)

    click.echo("Extracting durations …")
    raw_durations = ds["duration"]
    durations = [float(d) for d in tqdm(raw_durations, desc="Durations", unit="val") if d is not None]

    if min_duration is not None:
        durations = [d for d in durations if d >= min_duration]
    if max_duration is not None:
        durations = [d for d in durations if d <= max_duration]

    if not durations:
        click.echo("No duration data found in the dataset.", err=True)
        return

    n = len(durations)
    min_d, max_d = min(durations), max(durations)
    mean_d = sum(durations) / n
    total_d = sum(durations)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(durations, bins=100, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(f"Duration Distribution — {_display_name(dataset_paths)}")

    stats_text = (
        f"Samples : {n}\n"
        f"Total   : {total_d:.1f}s ({total_d / 3600:.2f}h)\n"
        f"Min     : {min_d:.2f}s\n"
        f"Max     : {max_d:.2f}s\n"
        f"Mean    : {mean_d:.2f}s"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
    )
    plt.tight_layout()

    name = _display_name(dataset_paths)
    chart_file = f"{name}_analysis.png"
    chart_path = Path.cwd() / chart_file
    fig.savefig(str(chart_path), dpi=150)
    click.echo(f"Chart saved to {chart_path}")

    import subprocess, platform
    try:
        if platform.system() == "Linux":
            subprocess.Popen(["xdg-open", str(chart_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(chart_path)])
        else:
            subprocess.Popen(["start", str(chart_path)], shell=True)
    except Exception:
        click.echo("Could not open the chart automatically. Please open it manually.")


@cli.command()
@click.argument("dataset_paths", nargs=-1, required=True)
@click.option(
    "--output-path", required=True,
    type=click.Path(path_type=Path),
    help="Destination directory for the merged dataset.",
)
def merge(dataset_paths: tuple[str, ...], output_path: Path):
    """Concatenate several datasets into one and save to disk."""
    if len(dataset_paths) < 2:
        click.echo("Error: merge requires at least two dataset paths.", err=True)
        sys.exit(1)

    ds = _load_and_concat(dataset_paths)
    click.echo(f"Merged dataset: {len(ds)} samples.")

    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving to {output_path} …")
    ds.save_to_disk(str(output_path))
    click.echo("Done.")


@cli.command(name="export")
@click.argument("dataset_paths", nargs=-1, required=True)
@click.option(
    "--output-path", required=True,
    type=click.Path(path_type=Path),
    help="Target directory for the exported files.",
)
@click.option(
    "--jsonl", "jsonl_name",
    default="vibevoice_dataset.jsonl", show_default=True,
    help="Name of the JSONL file.",
)
@click.option(
    "--full", is_flag=True, default=False,
    help="Copy audio samples and voice prompts to the output directory.",
)
def export_cmd(
    dataset_paths: tuple[str, ...],
    output_path: Path,
    jsonl_name: str,
    full: bool,
):
    """Export a dataset to JSONL format."""
    ds = _load_and_concat(dataset_paths)

    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path / jsonl_name
    if jsonl_path.exists():
        if not click.confirm(f"'{jsonl_path}' already exists. Overwrite?"):
            click.echo("Aborted.")
            return

    samples_dir = output_path / "samples"
    voice_prompts_dir = output_path / "voice_prompts"
    has_voice_prompts = "voice_prompts" in ds.column_names

    if full:
        samples_dir.mkdir(parents=True, exist_ok=True)
        if has_voice_prompts:
            voice_prompts_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not samples_dir.is_dir():
            click.echo(
                f"Error: '{samples_dir}' does not exist. "
                "Use --full to export all files.",
                err=True,
            )
            sys.exit(1)
        if has_voice_prompts and not voice_prompts_dir.is_dir():
            click.echo(
                f"Error: '{voice_prompts_dir}' does not exist. "
                "Use --full to export all files.",
                err=True,
            )
            sys.exit(1)

    records: list[dict] = []
    missing: list[str] = []

    for idx, sample in enumerate(tqdm(ds, desc="Processing", unit="sample")):
        audio = sample["audio"]
        text = sample.get("text", "")
        duration = sample.get("duration")
        vps = (sample.get("voice_prompts") or []) if has_voice_prompts else []

        # ── audio ─────────────────────────────────────────────────────
        audio_fn = _audio_filename(audio, idx)
        rel_audio = str(Path("samples") / audio_fn)
        abs_audio = samples_dir / audio_fn

        if full:
            _write_audio(audio, abs_audio)
        elif not abs_audio.exists():
            missing.append(str(abs_audio))

        # ── voice prompts ─────────────────────────────────────────────
        prompt_rels: list[str] = []
        for vp_idx, vp in enumerate(vps):
            vp_fn = _audio_filename(vp, idx, prompt_index=vp_idx)
            rel_vp = str(Path("voice_prompts") / vp_fn)
            abs_vp = voice_prompts_dir / vp_fn

            if full:
                _write_audio(vp, abs_vp)
            elif not abs_vp.exists():
                missing.append(str(abs_vp))

            prompt_rels.append(rel_vp)

        records.append({
            "audio": rel_audio,
            "text": text,
            "duration": duration,
            "voice_prompts": prompt_rels,
        })

    if missing:
        click.echo(f"Error: {len(missing)} required file(s) not found:", err=True)
        for m in missing[:20]:
            click.echo(f"  {m}", err=True)
        if len(missing) > 20:
            click.echo(f"  … and {len(missing) - 20} more", err=True)
        sys.exit(1)

    click.echo(f"Writing {len(records)} records to {jsonl_path} …")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="Writing JSONL", unit="record"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    click.echo(f"Export complete → {jsonl_path}")


def main():
    cli()


if __name__ == "__main__":
    main()
