"""Build a HuggingFace DatasetDict from the NPSC Stortinget V1.0 corpus."""
from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import click
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import Audio, Dataset, DatasetDict, Sequence as HFSequence
from tqdm.auto import tqdm

from databuilder.voice_prompts import (
    VOICE_PROMPT_MIN_DURATION,
    VOICE_PROMPT_MAX_DURATION,
    SpeakerPromptCache,
    build_speaker_prompt_index,
    select_voice_prompts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

TARGET_SR = 24_000
ENHANCE_NFE_DEFAULT = 32
ENHANCE_LAMBD_DEFAULT = 0.9
ENHANCE_TAU_DEFAULT = 0.7

EXCLUDED_FIELDS = frozenset({
    "context_after",
    "context_before",
    "meeting_date",
    "proceedings_text",
    "proceedingsfile",
    "sessionid",
    "transcriptionfile",
    "score",
})


# ── Text helpers ──────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"([.?!])\s+([a-zæøåäöü])")


def normalize_text(text: str) -> str:
    """Capitalize the first character and first letter after sentence-ending punctuation."""
    if not text:
        return text
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = _SENTENCE_END_RE.sub(
        lambda m: m.group(1) + " " + m.group(2).upper(), text
    )
    return text


# ── Audio enhancement ─────────────────────────────────────────────────────


def _enhance_single(src: Path, dst: Path, device: str, nfe: int = ENHANCE_NFE_DEFAULT, lambd: float = ENHANCE_LAMBD_DEFAULT, tau: float = ENHANCE_TAU_DEFAULT) -> bool:
    """Denoise, enhance, and resample one audio file to TARGET_SR WAV.

    Returns True on success.
    """
    from resemble_enhance.enhancer.inference import denoise, enhance

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            dwav, sr = torchaudio.load(src)
        dwav = dwav.mean(dim=0)

        dwav, sr = denoise(dwav, sr, device, run_dir="resemble_ai")
        dwav, sr = enhance(
            dwav, sr, device,
            nfe=nfe,
            solver="midpoint",
            lambd=lambd,
            tau=tau,
        )

        if sr != TARGET_SR:
            dwav = T.Resample(sr, TARGET_SR, dtype=dwav.dtype)(dwav)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            torchaudio.save(str(dst), dwav.unsqueeze(0), sample_rate=TARGET_SR)

        return True
    except Exception:
        logger.exception(f"Enhancement failed for {src.name}")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _enhance_all(
    file_pairs: List[tuple[Path, Path]],
    device: str,
    batch_size: int = 1,
    nfe: int = ENHANCE_NFE_DEFAULT,
    lambd: float = ENHANCE_LAMBD_DEFAULT,
    tau: float = ENHANCE_TAU_DEFAULT,
) -> None:
    """Enhance audio files using GPU-batched inference, skipping those already done."""
    from resemble_enhance.enhancer.inference import denoise_batch, enhance, enhance_batch

    todo = [(s, d) for s, d in file_pairs if not d.exists()]
    done = len(file_pairs) - len(todo)
    logger.info(f"Audio enhancement: {len(todo)} pending, {done} already done")
    if not todo:
        return

    batch_size = max(1, batch_size)
    file_batch_size = 10
    n_batches = (len(todo) + file_batch_size - 1) // file_batch_size
    processed = 0

    for batch_idx in range(n_batches):
        batch = todo[batch_idx * file_batch_size : (batch_idx + 1) * file_batch_size]
        click.echo(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch)} files)")

        # ── Load waveforms ────────────────────────────────────
        dwavs: list[torch.Tensor] = []
        srs: list[int] = []
        valid: list[tuple[Path, Path]] = []

        for src, dst in batch:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    dwav, sr = torchaudio.load(src)
                dwavs.append(dwav.mean(dim=0))
                srs.append(sr)
                valid.append((src, dst))
            except Exception:
                logger.exception(f"Failed to load {src.name}")

        if not dwavs:
            continue

        # ── Denoise (batched GPU inference) ───────────────────
        try:
            denoised = denoise_batch(dwavs, srs, device, run_dir="resemble_ai", batch_size=batch_size)
            dwavs = [d for d, s in denoised]
            srs = [s for d, s in denoised]
        except Exception:
            logger.exception(f"Batch denoise failed (batch {batch_idx + 1}), falling back to sequential")
            for src, dst in valid:
                _enhance_single(src, dst, device, nfe=nfe, lambd=lambd, tau=tau)
            processed += len(valid)
            click.echo(f"  Progress: {processed}/{len(todo)} files")
            continue

        # ── Enhance (batched GPU inference) ───────────────────
        try:
            enhanced = enhance_batch(
                dwavs, srs, device,
                nfe=nfe,
                solver="midpoint",
                lambd=lambd,
                tau=tau,
                batch_size=batch_size,
            )
        except Exception:
            logger.exception(f"Batch enhance failed (batch {batch_idx + 1}), falling back to sequential")
            enhanced = []
            for dwav, sr_val in zip(dwavs, srs):
                try:
                    enhanced.append(enhance(dwav, sr_val, device, nfe=nfe, solver="midpoint", lambd=lambd, tau=tau))
                except Exception:
                    enhanced.append(None)
            for (src, dst), result in zip(valid, enhanced):
                if result is None:
                    logger.error(f"Sequential enhance also failed for {src.name}")
                    continue
                hwav, sr = result
                try:
                    if sr != TARGET_SR:
                        hwav = T.Resample(sr, TARGET_SR, dtype=hwav.dtype)(hwav)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        torchaudio.save(str(dst), hwav.unsqueeze(0), sample_rate=TARGET_SR)
                except Exception:
                    logger.exception(f"Failed to save {dst.name}")
            processed += len(valid)
            click.echo(f"  Progress: {processed}/{len(todo)} files")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ── Resample & save ───────────────────────────────────
        for (src, dst), (hwav, sr) in zip(valid, enhanced):
            try:
                if sr != TARGET_SR:
                    hwav = T.Resample(sr, TARGET_SR, dtype=hwav.dtype)(hwav)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    torchaudio.save(str(dst), hwav.unsqueeze(0), sample_rate=TARGET_SR)
            except Exception:
                logger.exception(f"Failed to save {dst.name}")

        processed += len(valid)
        click.echo(f"  Progress: {processed}/{len(todo)} files")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Multi-speaker diarization helpers ─────────────────────────────────────


def _diarize_and_cut_segments(
    audio_path: Path,
    work_dir: Path,
    vad_model,
    diarization_state: dict,
) -> List[tuple[Path, str]]:
    """Diarize audio and return list of (segment_wav_path, canonical_speaker_id)."""
    from databuilder.pipeline import (
        diarize_audio,
        _smooth_speaker_turns,
        _merge_adjacent_speaker_segments,
        _canonicalize_speaker_id,
    )

    segments = diarize_audio(
        audio_path,
        work_dir,
        vad_model=vad_model,
        diarization_state=diarization_state,
    )
    if not segments:
        return []

    segments = _smooth_speaker_turns(segments)
    utterances = _merge_adjacent_speaker_segments(segments, max_duration=None)
    if not utterances:
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    seg_dir = work_dir / "diar_segments" / audio_path.stem
    seg_dir.mkdir(parents=True, exist_ok=True)

    result: List[tuple[Path, str]] = []
    for i, utt in enumerate(utterances):
        s = int(float(utt["start_time"]) * sr)
        e = min(int(float(utt["end_time"]) * sr), waveform.shape[1])
        if e <= s:
            continue
        chunk = waveform[:, s:e]
        if sr != 16000:
            chunk = torchaudio.functional.resample(chunk, int(sr), 16000)
        seg_path = seg_dir / f"seg_{i:04d}.wav"
        torchaudio.save(str(seg_path), chunk, 16000)
        speaker_id = _canonicalize_speaker_id(str(utt["speaker_id"]))
        result.append((seg_path, speaker_id))

    return result


def _assemble_speaker_text(
    segments: List[tuple[Path, str]],
    transcripts: Dict[Path, str],
) -> tuple[str, List[str]]:
    """Build Speaker-prefixed text from diarized segments and their transcripts.

    Returns ``(assembled_text, ordered_speaker_ids)`` where
    *ordered_speaker_ids* lists each canonical speaker id in the order it
    first appears in the text.
    """
    speaker_map: dict[str, int] = {}
    next_idx = 0
    parts: List[str] = []
    ordered_speakers: List[str] = []

    for seg_path, speaker_id in segments:
        text = transcripts.get(seg_path, "").strip()
        if not text:
            continue
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = next_idx
            next_idx += 1
            ordered_speakers.append(speaker_id)
        idx = speaker_map[speaker_id]
        text = normalize_text(text)
        parts.append(f"Speaker {idx}: {text}")

    return "\n".join(parts), ordered_speakers


# ── Main CLI ──────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    default=Path("/mnt/storage/data/Audio/Stortinget V1.0"),
    show_default=True,
    help="Root of the Stortinget V1.0 dataset.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to save the final HF DatasetDict.",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory for intermediate files (default: <output-path>/work).",
)
@click.option("--device", default=None, help="Torch device (default: auto-detect).")
@click.option(
    "--whisper-model",
    default="openai/whisper-large-v3",
    show_default=True,
    help="Whisper model for multi-speaker transcription.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    show_default=True,
    help="Batch size for Whisper transcription.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Workers for CPU-bound tasks.",
)
@click.option(
    "--batch-size-enhance",
    type=int,
    default=32,
    show_default=True,
    help="Number of chunks per GPU inference batch for audio enhancement.",
)
@click.option("--nfe", type=int, default=ENHANCE_NFE_DEFAULT, show_default=True, help="Number of function evaluations for enhancement.")
@click.option("--lambd", type=float, default=ENHANCE_LAMBD_DEFAULT, show_default=True, help="Lambda parameter for enhancement.")
@click.option("--tau", type=float, default=ENHANCE_TAU_DEFAULT, show_default=True, help="Tau parameter for enhancement.")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N records (for testing).",
)
def main(
    input_path: Path,
    output_path: Path,
    work_dir: Path | None,
    device: str | None,
    whisper_model: str,
    batch_size: int,
    num_workers: int,
    batch_size_enhance: int,
    nfe: int,
    lambd: float,
    tau: float,
    limit: int | None,
) -> None:
    """Build a HuggingFace DatasetDict from the NPSC Stortinget V1.0 corpus."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = output_path.resolve()
    work_dir = (work_dir or output_path / "work").resolve()
    enhanced_dir = work_dir / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = input_path / "ssc_v1_0.jsonl"
    if not jsonl_path.exists():
        click.echo(f"Error: {jsonl_path} not found.", err=True)
        sys.exit(1)

    # ── Phase 1: Load and filter records ──────────────────────────────

    click.echo("Phase 1: Loading metadata …")
    records: List[dict] = []
    with jsonl_path.open() as f:
        for line in tqdm(f, desc="Reading JSONL", unit="rec"):
            rec = json.loads(line)
            if rec["num_speakers"] > 4:
                continue
            records.append(rec)

    if limit:
        records = records[:limit]

    n_single = sum(1 for r in records if r["num_speakers"] == 1)
    n_multi = len(records) - n_single
    click.echo(f"  {len(records)} records (single-speaker={n_single}, multi-speaker={n_multi})")

    # ── Phase 2: Enhance audio ────────────────────────────────────────

    click.echo("Phase 2: Audio enhancement …")
    seen: set[str] = set()
    file_pairs: List[tuple[Path, Path]] = []
    for rec in records:
        rel = rec["audio_path"]
        if rel in seen:
            continue
        seen.add(rel)
        src = input_path / rel
        dst = enhanced_dir / f"{Path(rel).stem}.wav"
        file_pairs.append((src, dst))

    click.echo(f"  {len(file_pairs)} unique audio files")
    _enhance_all(file_pairs, device, batch_size=batch_size_enhance, nfe=nfe, lambd=lambd, tau=tau)

    # ── Phase 3: Single-speaker records (CPU-parallel) ────────────────

    click.echo("Phase 3: Processing single-speaker records …")
    single_records = [r for r in records if r["num_speakers"] == 1]
    multi_records = [r for r in records if r["num_speakers"] > 1]

    def _process_single(rec: dict) -> dict | None:
        rel = rec["audio_path"]
        enh = enhanced_dir / f"{Path(rel).stem}.wav"
        if not enh.exists():
            return None
        text = f"Speaker 0: {normalize_text(rec['transcription_text'])}"
        try:
            info = torchaudio.info(str(enh))
            duration = info.num_frames / info.sample_rate
        except Exception:
            duration = rec.get("duration")
        return {
            "segment_id": rec["segment_id"],
            "audio": str(enh),
            "text": text,
            "duration": duration,
            "num_speakers": rec["num_speakers"],
            "speakers": json.dumps(rec["speakers"], ensure_ascii=False),
            "split": rec["split"],
            "voice_prompts": [],
        }

    results: List[dict] = []

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = [pool.submit(_process_single, r) for r in single_records]
        for f in tqdm(
            as_completed(futs), total=len(futs),
            desc="Single-speaker", unit="rec",
        ):
            r = f.result()
            if r:
                results.append(r)

    # ── Phase 4 + 5: Multi-speaker diarization & transcription ────────

    if multi_records:
        click.echo(f"Phase 4: Diarizing {len(multi_records)} multi-speaker records …")
        from databuilder.pipeline import (
            _load_silero_vad_model,
            _resolve_diarization_device,
        )

        vad_model = _load_silero_vad_model()
        diar_state: dict = {"device": _resolve_diarization_device(device)}

        record_segments: dict[int, List[tuple[Path, str]]] = {}
        all_seg_paths: List[Path] = []

        for idx, rec in enumerate(
            tqdm(multi_records, desc="Diarizing", unit="rec")
        ):
            rel = rec["audio_path"]
            enh = enhanced_dir / f"{Path(rel).stem}.wav"
            if not enh.exists():
                continue
            try:
                segs = _diarize_and_cut_segments(
                    enh, work_dir, vad_model, diar_state,
                )
            except Exception:
                logger.exception(f"Diarization failed for {enh.name}")
                continue
            if segs:
                record_segments[idx] = segs
                all_seg_paths.extend(p for p, _ in segs)

        # Batch-transcribe all segments at once
        all_transcripts: Dict[Path, str] = {}
        if all_seg_paths:
            click.echo(
                f"Phase 5: Transcribing {len(all_seg_paths)} diarized segments …"
            )
            from databuilder.transcribe import transcribe_audio as _transcribe

            all_transcripts = _transcribe(
                all_seg_paths,
                model_id=whisper_model,
                device=device,
                batch_size=batch_size,
                language="no",
            )

        # Assemble multi-speaker text
        click.echo("Phase 6: Assembling multi-speaker text …")
        for idx, rec in enumerate(multi_records):
            rel = rec["audio_path"]
            enh = enhanced_dir / f"{Path(rel).stem}.wav"

            segs = record_segments.get(idx)
            if segs:
                text, ordered_speakers = _assemble_speaker_text(segs, all_transcripts)
            else:
                text, ordered_speakers = "", []

            # ── voice prompts from diarized segments ──────────────
            voice_prompts: list[str] = []
            if ordered_speakers and segs:
                seg_candidates = []
                for seg_path, speaker_id in segs:
                    try:
                        seg_info = torchaudio.info(str(seg_path))
                        seg_dur = seg_info.num_frames / seg_info.sample_rate
                    except Exception:
                        continue
                    seg_candidates.append({
                        "speaker_id": speaker_id,
                        "path": str(seg_path),
                        "duration": seg_dur,
                    })
                if seg_candidates:
                    prompt_idx = build_speaker_prompt_index(seg_candidates)
                    prompts = select_voice_prompts(ordered_speakers, prompt_idx)
                    voice_prompts = prompts if prompts else []

            if not text:
                text = f"Speaker 0: {normalize_text(rec['transcription_text'])}"

            try:
                info = torchaudio.info(str(enh))
                duration = info.num_frames / info.sample_rate
            except Exception:
                duration = rec.get("duration")

            results.append({
                "segment_id": rec["segment_id"],
                "audio": str(enh),
                "text": text,
                "duration": duration,
                "num_speakers": rec["num_speakers"],
                "speakers": json.dumps(rec["speakers"], ensure_ascii=False),
                "split": rec["split"],
                "voice_prompts": voice_prompts,
            })

    click.echo(f"Total processed: {len(results)}")

    # ── Voice prompts for single-speaker records ─────────────────────

    click.echo("Assigning voice prompts to single-speaker records …")
    cache = SpeakerPromptCache(max_size=50)
    n_single = 0
    n_assigned = 0

    for r in results:
        if r["num_speakers"] != 1:
            continue
        n_single += 1
        speakers_data = json.loads(r["speakers"])
        if not speakers_data:
            continue
        speaker_id = speakers_data[0]["speaker_id"]
        audio_path = r["audio"]
        duration = r.get("duration")

        # Populate cache with clips whose duration is suitable
        if duration is not None and VOICE_PROMPT_MIN_DURATION <= float(duration) <= VOICE_PROMPT_MAX_DURATION:
            cache.add(speaker_id, audio_path)

        # Try to pick a different clip from the same speaker
        prompt = cache.select(speaker_id, exclude={audio_path})
        if prompt:
            r["voice_prompts"] = [prompt]
            n_assigned += 1
        else:
            # Self-fallback: use the sample's own audio as the prompt
            r["voice_prompts"] = [audio_path]
            n_assigned += 1

    click.echo(f"  Assigned voice prompts to {n_assigned}/{n_single} single-speaker records")

    # ── Phase 7: Build DatasetDict ────────────────────────────────────

    click.echo("Phase 7: Building HF DatasetDict …")
    split_data: dict[str, List[dict]] = {}
    for r in results:
        split_name = r.pop("split")
        split_data.setdefault(split_name, []).append(r)

    ds_dict = {}
    for split_name, items in sorted(split_data.items()):
        click.echo(f"  Split '{split_name}': {len(items)} samples")
        ds = Dataset.from_list(items)
        ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        ds = ds.cast_column("voice_prompts", HFSequence(Audio(sampling_rate=TARGET_SR)))
        ds_dict[split_name] = ds

    dataset_dict = DatasetDict(ds_dict)

    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving to {output_path} …")
    dataset_dict.save_to_disk(str(output_path))
    click.echo("Done.")


if __name__ == "__main__":
    main()
