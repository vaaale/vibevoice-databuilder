import argparse
import itertools
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torchaudio.transforms as T

from resemble_enhance.enhancer.inference import (
    denoise,
    denoise_batch,
    enhance,
    enhance_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}


def get_audio_duration(file_path: Path) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        waveform, sr = torchaudio.load(file_path)
    return waveform.shape[-1] / sr


def collect_audio_files(directory: Path) -> list[Path]:
    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def load_audio_files(audio_files: List[Path]) -> Tuple[List[torch.Tensor], List[int]]:
    dwavs = []
    srs = []
    for audio_file in audio_files:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            dwav, sr = torchaudio.load(audio_file)
        dwavs.append(dwav.mean(dim=0))
        srs.append(sr)
    return dwavs, srs


def save_audio_files(
    audio_files: List[Path],
    dwavs: List[torch.Tensor],
    srs: List[int],
    original_srs: List[int],
    output_dir: Path,
    resample: bool = False,
) -> List[Path]:
    output_files = []
    for audio_file, wav, new_sr, orig_sr in zip(audio_files, dwavs, srs, original_srs):
        output_file = output_dir / audio_file.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if resample:
                resampler = T.Resample(new_sr, orig_sr, dtype=wav.dtype)
                wav = resampler(wav)
                torchaudio.save(output_file, wav.unsqueeze(0), sample_rate=orig_sr)
            else:
                torchaudio.save(output_file, wav.unsqueeze(0), sample_rate=new_sr)
        output_files.append(output_file)
    return output_files


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",")]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",")]


def run_batch(
    audio_files: List[Path],
    output_dir: Path,
    do_denoise: bool,
    do_enhance: bool,
    nfe: int,
    lambd: float,
    tau: float,
    device: str,
    batch_size: int,
    resample: bool,
):
    # Measure total audio duration
    total_audio_duration = 0.0
    for f in audio_files:
        total_audio_duration += get_audio_duration(f)

    # Load
    t_load_start = time.perf_counter()
    dwavs, srs = load_audio_files(audio_files)
    original_srs = list(srs)
    t_load_elapsed = time.perf_counter() - t_load_start
    logger.info(f"Audio loading: {t_load_elapsed:.3f}s")

    # Denoise
    t_denoise_elapsed = 0.0
    if do_denoise:
        t_denoise_start = time.perf_counter()
        results = denoise_batch(
            dwavs, srs, device, run_dir="resemble_ai", batch_size=batch_size,
        )
        dwavs = [r[0] for r in results]
        srs = [r[1] for r in results]
        t_denoise_elapsed = time.perf_counter() - t_denoise_start
        logger.info(f"Denoising: {t_denoise_elapsed:.2f}s")

    # Enhance
    t_enhance_elapsed = 0.0
    if do_enhance:
        t_enhance_start = time.perf_counter()
        results = enhance_batch(
            dwavs, srs, device,
            nfe=nfe, solver="midpoint", lambd=lambd, tau=tau,
            batch_size=batch_size,
        )
        dwavs = [r[0] for r in results]
        srs = [r[1] for r in results]
        t_enhance_elapsed = time.perf_counter() - t_enhance_start
        logger.info(f"Enhancement: {t_enhance_elapsed:.2f}s")

    # Save
    t_save_start = time.perf_counter()
    output_files = save_audio_files(
        audio_files, dwavs, srs, original_srs, output_dir, resample=resample,
    )
    t_save_elapsed = time.perf_counter() - t_save_start
    logger.info(f"Saving: {t_save_elapsed:.3f}s")

    t_total = t_load_elapsed + t_denoise_elapsed + t_enhance_elapsed + t_save_elapsed

    return {
        "t_load": t_load_elapsed,
        "t_denoise": t_denoise_elapsed,
        "t_enhance": t_enhance_elapsed,
        "t_save": t_save_elapsed,
        "t_total": t_total,
        "audio_duration": total_audio_duration,
        "n_files": len(output_files),
        "output_files": output_files,
    }


def run_sweep(
    input_file: Path,
    output_dir: Path,
    do_denoise: bool,
    nfe_values: List[int],
    lambd_values: List[float],
    tau_values: List[float],
    device: str,
    resample: bool,
):
    combos = list(itertools.product(nfe_values, lambd_values, tau_values))
    logger.info(f"Parameter sweep: {len(combos)} combinations")
    logger.info(f"  nfe:   {nfe_values}")
    logger.info(f"  lambd: {lambd_values}")
    logger.info(f"  tau:   {tau_values}")

    # Load input file once
    audio_duration = get_audio_duration(input_file)
    logger.info(f"Input: {input_file.name} ({audio_duration:.1f}s)")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        dwav, sr = torchaudio.load(input_file)
    dwav = dwav.mean(dim=0)

    # Denoise once (shared across all sweep runs)
    if do_denoise:
        t_denoise_start = time.perf_counter()
        dwav, sr = denoise(dwav, sr, device, run_dir="resemble_ai")
        t_denoise = time.perf_counter() - t_denoise_start
        logger.info(f"Denoising (shared): {t_denoise:.2f}s")
    else:
        t_denoise = 0.0

    stem = input_file.stem
    sweep_results = []

    for i, (nfe, lambd, tau) in enumerate(combos, 1):
        label = f"nfe{nfe}_lambd{lambd:.2f}_tau{tau:.2f}"
        logger.info(f"[{i}/{len(combos)}] {label}")

        t_start = time.perf_counter()
        hwav, new_sr = enhance(dwav, sr, device, nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)
        t_enhance = time.perf_counter() - t_start

        out_name = f"{stem}_{label}.wav"
        out_path = output_dir / out_name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if resample:
                orig_sr = sr
                resampler = T.Resample(new_sr, orig_sr, dtype=hwav.dtype)
                hwav = resampler(hwav)
                torchaudio.save(out_path, hwav.unsqueeze(0), sample_rate=orig_sr)
            else:
                torchaudio.save(out_path, hwav.unsqueeze(0), sample_rate=new_sr)

        rtf = t_enhance / audio_duration if audio_duration > 0 else float("inf")
        logger.info(f"  -> {out_name}  ({t_enhance:.2f}s, RTF={rtf:.3f}x)")

        sweep_results.append({
            "nfe": nfe,
            "lambd": lambd,
            "tau": tau,
            "t_enhance": t_enhance,
            "rtf": rtf,
            "output": out_path,
        })

    # Summary table
    logger.info("")
    logger.info("=== Sweep Summary ===")
    logger.info(f"  Input:          {input_file.name} ({audio_duration:.1f}s)")
    if do_denoise:
        logger.info(f"  Denoise time:   {t_denoise:.2f}s")
    logger.info(f"  Output dir:     {output_dir}")
    logger.info("")
    header = f"  {'nfe':>5}  {'lambd':>7}  {'tau':>7}  {'time(s)':>8}  {'RTF':>7}"
    logger.info(header)
    logger.info(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for r in sweep_results:
        logger.info(
            f"  {r['nfe']:>5}  {r['lambd']:>7.2f}  {r['tau']:>7.2f}"
            f"  {r['t_enhance']:>8.2f}  {r['rtf']:>7.3f}"
        )

    t_total_enhance = sum(r["t_enhance"] for r in sweep_results)
    logger.info("")
    logger.info(f"  Total enhance time: {t_total_enhance:.2f}s")
    logger.info(f"  Total wall time:    {t_denoise + t_total_enhance:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run denoising and enhancement on all audio files in a directory, "
                    "or run a parameter sweep on a single file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- batch subcommand ---
    batch_parser = subparsers.add_parser("batch", help="Process all files in a directory")
    batch_parser.add_argument("input_dir", type=Path, help="Directory containing audio files")
    batch_parser.add_argument("--output-dir", type=Path, default=None,
                              help="Output directory (default: <input_dir>_enhanced)")
    batch_parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    batch_parser.add_argument("--no-enhance", action="store_true", help="Skip enhancement")
    batch_parser.add_argument("--nfe", type=int, default=32, help="Number of function evaluations (default: 32)")
    batch_parser.add_argument("--lambd", type=float, default=0.5, help="Lambda parameter (default: 0.5)")
    batch_parser.add_argument("--tau", type=float, default=0.5, help="Tau parameter (default: 0.5)")
    batch_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference (default: 4)")
    batch_parser.add_argument("--device", type=str, default=None,
                              help="Torch device (default: cuda if available, else cpu)")
    batch_parser.add_argument("--resample", action="store_true",
                              help="Resample output back to original sample rate")

    # --- sweep subcommand ---
    sweep_parser = subparsers.add_parser("sweep", help="Parameter sweep on a single audio file")
    sweep_parser.add_argument("input_file", type=Path, help="Single audio file to process")
    sweep_parser.add_argument("--output-dir", type=Path, default=None,
                              help="Output directory (default: <input_file_stem>_sweep)")
    sweep_parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    sweep_parser.add_argument("--nfe", type=parse_int_list, default="16,32,64",
                              help="Comma-separated NFE values (default: 16,32,64)")
    sweep_parser.add_argument("--lambd", type=parse_float_list, default="0.1,0.5,0.9",
                              help="Comma-separated lambda values (default: 0.1,0.5,0.9)")
    sweep_parser.add_argument("--tau", type=parse_float_list, default="0.0,0.5,1.0",
                              help="Comma-separated tau values (default: 0.0,0.5,1.0)")
    sweep_parser.add_argument("--device", type=str, default=None,
                              help="Torch device (default: cuda if available, else cpu)")
    sweep_parser.add_argument("--resample", action="store_true",
                              help="Resample output back to original sample rate")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.command == "batch":
        if not args.input_dir.is_dir():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            sys.exit(1)

        do_denoise = not args.no_denoise
        do_enhance = not args.no_enhance

        if not do_denoise and not do_enhance:
            logger.error("Nothing to do: both --no-denoise and --no-enhance are set")
            sys.exit(1)

        output_dir = args.output_dir or args.input_dir.parent / f"{args.input_dir.name}_enhanced"
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = collect_audio_files(args.input_dir)
        if not audio_files:
            logger.warning(f"No audio files found in {args.input_dir}")
            sys.exit(0)

        logger.info(f"Found {len(audio_files)} audio files in {args.input_dir}")
        logger.info(f"Device: {device}, Batch size: {args.batch_size}")
        logger.info(f"Denoise: {do_denoise}, Enhance: {do_enhance}")
        logger.info(f"Output directory: {output_dir}")

        # Measure total audio duration
        total_audio_duration = 0.0
        for f in audio_files:
            total_audio_duration += get_audio_duration(f)
        logger.info(f"Total audio duration: {total_audio_duration:.1f}s")

        stats = run_batch(
            audio_files=audio_files,
            output_dir=output_dir,
            do_denoise=do_denoise,
            do_enhance=do_enhance,
            nfe=args.nfe,
            lambd=args.lambd,
            tau=args.tau,
            device=device,
            batch_size=args.batch_size,
            resample=args.resample,
        )

        n_files = stats["n_files"]
        t_total = stats["t_total"]
        avg_per_file = t_total / n_files if n_files > 0 else 0
        rtf = t_total / total_audio_duration if total_audio_duration > 0 else float("inf")

        logger.info(f"Processed {n_files} files -> {output_dir}")

        logger.info("=== Timing Summary ===")
        logger.info(f"  Loading:           {stats['t_load']:.3f}s")
        if do_denoise:
            logger.info(f"  Denoising:         {stats['t_denoise']:.2f}s")
        if do_enhance:
            logger.info(f"  Enhancement:       {stats['t_enhance']:.2f}s")
        logger.info(f"  Saving:            {stats['t_save']:.3f}s")
        logger.info(f"  Total:             {t_total:.2f}s")
        logger.info(f"  Avg per file:      {avg_per_file:.2f}s")
        logger.info(f"  Audio duration:    {total_audio_duration:.1f}s")
        logger.info(f"  Real-time factor:  {rtf:.3f}x (lower is faster)")

    elif args.command == "sweep":
        if not args.input_file.is_file():
            logger.error(f"Input file does not exist: {args.input_file}")
            sys.exit(1)

        output_dir = args.output_dir or args.input_file.parent / f"{args.input_file.stem}_sweep"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_sweep(
            input_file=args.input_file,
            output_dir=output_dir,
            do_denoise=not args.no_denoise,
            nfe_values=args.nfe,
            lambd_values=args.lambd,
            tau_values=args.tau,
            device=device,
            resample=args.resample,
        )


if __name__ == "__main__":
    main()
