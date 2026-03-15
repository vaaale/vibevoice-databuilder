"""Core audio segmentation, transcription, and dataset utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
from datasets import Audio, Dataset, Sequence as HFSequence
try:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
except ModuleNotFoundError:  # pragma: no cover
    get_speech_timestamps = None  # type: ignore[assignment]
    load_silero_vad = None  # type: ignore[assignment]
    read_audio = None  # type: ignore[assignment]
from tqdm.auto import tqdm
try:
    from transformers import pipeline as hf_pipeline
except ModuleNotFoundError:  # pragma: no cover
    hf_pipeline = None  # type: ignore[assignment]


_SUPPORTED_EXTENSIONS = {".wav", ".mp3"}
_SPEAKER_DELIMITER = "__spk__"
_OUTPUT_SAMPLE_RATE = 24_000
_DEFAULT_DIARIZATION_CHUNK_SIZE = 30.0
_SORTFORMER_SESSION_LEN_SEC = 90.0
_SENTENCE_END_PUNCTUATION = (".", "?", "!")


def _require_torch() -> None:
    if torch is None or torchaudio is None:  # pragma: no cover
        raise RuntimeError("torch and torchaudio must be installed to process audio")


def _require_silero_vad() -> None:
    if get_speech_timestamps is None or load_silero_vad is None or read_audio is None:  # pragma: no cover
        raise RuntimeError("silero-vad must be installed to run VAD segmentation")


def _require_transformers() -> None:
    if hf_pipeline is None:  # pragma: no cover
        raise RuntimeError("transformers must be installed to run transcription")


def _load_silero_vad_model():
    _require_silero_vad()
    return load_silero_vad()  # type: ignore[misc]


def _get_speech_timestamps_for_audio(
    audio_path: Path,
    *,
    vad_model,
    vad_sampling_rate: int = 16_000,
    max_duration: float | None = None,
) -> List[dict]:
    _require_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=vad_sampling_rate)  # type: ignore[misc]
    if max_duration is not None:
        max_samples = int(float(max_duration) * vad_sampling_rate)
        wav = wav[:max_samples]
    return get_speech_timestamps(wav, vad_model, sampling_rate=vad_sampling_rate)  # type: ignore[misc]


def _resolve_diarization_device(device: str | None) -> str:
    if device is not None:
        normalized = device.strip().lower()
        if normalized.startswith("cuda") and torch is not None and torch.cuda.is_available():
            return device
        if normalized.startswith("cpu"):
            return "cpu"
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _load_sortformer_diarization_model(*, device: str):
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "NeMo diarization dependencies are not available. "
            "Install nemo_toolkit[asr] and omegaconf to enable diarization."
        ) from exc

    try:
        model = SortformerEncLabelModel.from_pretrained(
            model_name="nvidia/diar_sortformer_4spk-v1",
            map_location=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load nvidia/diar_sortformer_4spk-v1 for diarization. "
            "Ensure NeMo can access the model from Hugging Face."
        ) from exc

    model.eval()
    return model


def _build_vad_chunks(
    timestamps: Sequence[dict],
    *,
    chunk_size: float,
    vad_sampling_rate: int = 16_000,
) -> List[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: List[dict] = []
    current_start: float | None = None
    current_end: float | None = None

    def append_chunk_range(start_time: float, end_time: float) -> None:
        if end_time <= start_time:
            return
        range_start = float(start_time)
        range_end = float(end_time)
        while (range_end - range_start) > chunk_size:
            chunk_end = range_start + chunk_size
            chunks.append({"start_time": range_start, "end_time": chunk_end})
            range_start = chunk_end
        if range_end > range_start:
            chunks.append({"start_time": range_start, "end_time": range_end})

    for segment in timestamps:
        start_time = float(segment["start"]) / float(vad_sampling_rate)
        end_time = float(segment["end"]) / float(vad_sampling_rate)
        if end_time <= start_time:
            continue

        if current_start is None or current_end is None:
            current_start = start_time
            current_end = end_time
            continue

        if end_time - current_start <= chunk_size:
            current_end = max(float(current_end), end_time)
            continue

        append_chunk_range(float(current_start), float(current_end))
        current_start = start_time
        current_end = end_time

    if current_start is not None and current_end is not None:
        append_chunk_range(float(current_start), float(current_end))

    return chunks


def _iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            yield path


def _clip_waveform(audio_tensor: torch.Tensor, sample_rate: int, max_duration: float | None) -> torch.Tensor:
    if max_duration is None:
        return audio_tensor
    if max_duration <= 0:
        raise ValueError("max_duration must be > 0 when provided")
    max_samples = int(max_duration * sample_rate)
    if max_samples <= 0:
        return audio_tensor[:, :0]
    return audio_tensor[:, :max_samples]


def _resample_waveform(audio_tensor: torch.Tensor, sample_rate: int, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    if sample_rate == target_sample_rate:
        return audio_tensor, sample_rate
    audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)  # type: ignore[union-attr]
    return audio_tensor, target_sample_rate


def _format_speaker_prefix(speaker_id: str) -> str:
    normalized = speaker_id.strip()
    if not normalized:
        return ""
    if normalized.lower().startswith("speaker"):
        return f"{normalized}: "
    if normalized.lower().startswith("spk"):
        return f"{normalized}: "
    return f"{normalized}: "


def _canonicalize_speaker_id(speaker_id: str) -> str:
    normalized = speaker_id.strip()
    parts = normalized.split("_", 2)
    if len(parts) == 3 and parts[0] == "chunk" and parts[1].isdigit():
        return parts[2]
    return normalized


def segment_audio(
    input_dir: Path,
    segments_dir: Path,
    *,
    vad_model=None,
    vad_sampling_rate: int = 16_000,
    min_duration: float = 5.0,
    max_duration: float | None = None,
) -> List[Path]:
    """Split incoming audio files into voice segments saved under ``segments_dir``.

    Segments under min_duration seconds are concatenated with following segments
    until they reach at least min_duration seconds.
    """
    _require_torch()
    _require_silero_vad()
    if vad_model is None:
        vad_model = _load_silero_vad_model()
    segments_dir.mkdir(parents=True, exist_ok=True)
    created_segments: List[Path] = []

    for audio_path in tqdm(list(_iter_audio_files(input_dir)), desc="VAD", unit="file"):
        timestamps = _get_speech_timestamps_for_audio(
            audio_path,
            vad_model=vad_model,
            vad_sampling_rate=vad_sampling_rate,
            max_duration=max_duration,
        )
        if not timestamps:
            continue

        audio_tensor, sample_rate = torchaudio.load(str(audio_path))  # type: ignore[union-attr]
        audio_tensor = _clip_waveform(audio_tensor, sample_rate, max_duration)

        # Concatenate segments until they reach min_duration
        merged_segments = []
        current_group = []

        for segment in timestamps:
            current_group.append(segment)
            # Calculate total duration of current group
            group_start = current_group[0]["start"]
            group_end = current_group[-1]["end"]
            duration = (group_end - group_start) / vad_sampling_rate

            # If we've reached min_duration, save this group and start a new one
            if duration >= min_duration:
                merged_segments.append({
                    "start": group_start,
                    "end": group_end
                })
                current_group = []

        # Add any remaining segments
        if current_group:
            merged_segments.append({
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"]
            })

        # Save merged segments
        for idx, segment in enumerate(merged_segments, start=1):
            start = int(segment["start"] * sample_rate / vad_sampling_rate)
            end = int(segment["end"] * sample_rate / vad_sampling_rate)
            if end <= start:
                continue
            if max_duration is not None:
                end = min(end, audio_tensor.shape[1])
                if end <= start:
                    continue
            segment_tensor = audio_tensor[:, start:end]
            segment_tensor, out_sr = _resample_waveform(segment_tensor, sample_rate, _OUTPUT_SAMPLE_RATE)
            output_path = segments_dir / f"{audio_path.stem}_{idx:03d}.wav"
            torchaudio.save(str(output_path), segment_tensor, out_sr)  # type: ignore[union-attr]
            created_segments.append(output_path)

    return created_segments


def transcribe_audio(
    paths: Sequence[Path],
    model_id: str = "openai/whisper-base",
    device: str | None = None,
    batch_size: int = 8,
    language: str | None = None,
) -> Dict[Path, str]:
    from databuilder.transcribe import transcribe_audio as _transcribe_audio

    return _transcribe_audio(
        paths,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        language=language,
    )


def diarize_audio(
    audio_path: Path,
    work_dir: Path,
    *,
    chunk_size: float = _DEFAULT_DIARIZATION_CHUNK_SIZE,
    vad_model=None,
    diarization_state: Dict[str, object] | None = None,
) -> List[dict]:
    """Return diarization segments as dicts with keys: start_time, end_time, speaker_id."""
    _require_torch()
    try:
        from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
        from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "NeMo diarization dependencies are not available. "
            "Install nemo_toolkit[asr] and omegaconf to enable diarization."
        ) from exc

    if vad_model is None:
        vad_model = _load_silero_vad_model()
    if diarization_state is None:
        diarization_state = {}

    diarization_dir = work_dir / "diarization" / audio_path.stem
    diarization_dir.mkdir(parents=True, exist_ok=True)

    audio_tensor, sample_rate = torchaudio.load(str(audio_path))  # type: ignore[union-attr]
    if audio_tensor.ndim != 2:
        raise RuntimeError(f"Expected audio tensor with shape (channels, samples), got {tuple(audio_tensor.shape)}")
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)  # type: ignore[union-attr]
        sample_rate = 16000

    timestamps = _get_speech_timestamps_for_audio(
        audio_path,
        vad_model=vad_model,
        vad_sampling_rate=16_000,
    )
    if not timestamps:
        return []

    diarization_chunks = _build_vad_chunks(
        timestamps,
        chunk_size=chunk_size,
        vad_sampling_rate=16_000,
    )
    if not diarization_chunks:
        return []

    diarization_model = diarization_state.get("model")
    diarization_device = str(diarization_state.get("device") or _resolve_diarization_device(None))
    if diarization_model is None:
        diarization_model = _load_sortformer_diarization_model(device=diarization_device)
        diarization_state["model"] = diarization_model
        diarization_state["device"] = diarization_device

    diarize_cfg = DiarizeConfig(
        session_len_sec=_SORTFORMER_SESSION_LEN_SEC,
        batch_size=1,
        num_workers=0,
        verbose=False,
        postprocessing_params=load_postprocessing_from_yaml(None),
        # max_num_of_spks=4,
    )

    def run_sortformer_diarization(chunk_audio_path: Path) -> List[str]:
        nonlocal diarization_model
        nonlocal diarization_device

        try:
            predicted = diarization_model.diarize(
                audio=str(chunk_audio_path),
                override_config=diarize_cfg,
            )
        except torch.OutOfMemoryError:
            if diarization_device != "cuda":
                raise
            diarization_model = diarization_model.to("cpu")
            diarization_model.eval()
            diarization_device = "cpu"
            diarization_state["model"] = diarization_model
            diarization_state["device"] = diarization_device
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predicted = diarization_model.diarize(
                audio=str(chunk_audio_path),
                override_config=diarize_cfg,
            )

        diarization_lines = predicted
        if isinstance(predicted, list) and predicted and isinstance(predicted[0], list):
            diarization_lines = predicted[0]
        if not isinstance(diarization_lines, list):
            return []
        return [line for line in diarization_lines if isinstance(line, str)]

    segments: List[dict] = []
    for chunk_index, chunk in enumerate(diarization_chunks):
        chunk_start_time = float(chunk["start_time"])
        chunk_end_time = float(chunk["end_time"])
        start_sample = int(chunk_start_time * sample_rate)
        end_sample = min(int(chunk_end_time * sample_rate), audio_tensor.shape[1])
        if end_sample <= start_sample:
            continue

        chunk_tensor = audio_tensor[:, start_sample:end_sample]
        chunk_audio_path = diarization_dir / f"chunk_{chunk_index:05d}.wav"
        torchaudio.save(str(chunk_audio_path), chunk_tensor, sample_rate)  # type: ignore[union-attr]

        diarization_lines = run_sortformer_diarization(chunk_audio_path)
        for line in diarization_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            start_time = float(parts[0]) + chunk_start_time
            end_time = min(float(parts[1]) + chunk_start_time, chunk_end_time)
            if end_time <= start_time:
                continue
            speaker_id = f"chunk_{chunk_index:05d}_{parts[2]}"
            segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id,
                    "chunk_index": chunk_index,
                }
            )

    return sorted(segments, key=lambda s: (float(s["start_time"]), float(s["end_time"])))


def _merge_adjacent_speaker_segments(
    segments: List[dict],
    *,
    max_duration: float | None,
    merge_gap: float = 2.0,
) -> List[dict]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    merged: List[dict] = []

    for seg in ordered:
        start_time = float(seg["start_time"])
        end_time = float(seg["end_time"])
        raw_speaker_id = str(seg["speaker_id"])
        speaker_id = _canonicalize_speaker_id(raw_speaker_id)
        chunk_index = seg.get("chunk_index")
        if end_time <= start_time:
            continue

        if not merged:
            merged.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id,
                    "chunk_index": chunk_index,
                    "source_speaker_ids": [raw_speaker_id],
                    "source_chunk_indices": [chunk_index] if chunk_index is not None else [],
                }
            )
            continue

        prev = merged[-1]
        if prev["speaker_id"] != speaker_id:
            merged.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id,
                    "chunk_index": chunk_index,
                    "source_speaker_ids": [raw_speaker_id],
                    "source_chunk_indices": [chunk_index] if chunk_index is not None else [],
                }
            )
            continue

        gap = start_time - float(prev["end_time"])
        if gap > merge_gap:
            merged.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id,
                    "chunk_index": chunk_index,
                    "source_speaker_ids": [raw_speaker_id],
                    "source_chunk_indices": [chunk_index] if chunk_index is not None else [],
                }
            )
            continue

        candidate_end = max(float(prev["end_time"]), end_time)
        if max_duration is not None and (candidate_end - float(prev["start_time"])) > float(max_duration):
            merged.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_id": speaker_id,
                    "chunk_index": chunk_index,
                    "source_speaker_ids": [raw_speaker_id],
                    "source_chunk_indices": [chunk_index] if chunk_index is not None else [],
                }
            )
            continue

        prev["end_time"] = candidate_end
        previous_source_speaker_ids = list(prev.get("source_speaker_ids", []))
        previous_source_speaker_ids.append(raw_speaker_id)
        prev["source_speaker_ids"] = previous_source_speaker_ids
        previous_source_chunk_indices = list(prev.get("source_chunk_indices", []))
        if chunk_index is not None:
            previous_source_chunk_indices.append(chunk_index)
        prev["source_chunk_indices"] = previous_source_chunk_indices
        if prev.get("chunk_index") != chunk_index:
            prev["chunk_index"] = None

    if max_duration is not None:
        max_d = float(max_duration)
        capped: List[dict] = []
        for seg in merged:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            if end <= start:
                continue
            dur = end - start
            if dur <= max_d:
                capped.append(seg)
                continue
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + max_d, end)
                if chunk_end > chunk_start:
                    capped.append(
                        {
                            "start_time": chunk_start,
                            "end_time": chunk_end,
                            "speaker_id": seg["speaker_id"],
                            "chunk_index": seg.get("chunk_index"),
                            "source_speaker_ids": list(seg.get("source_speaker_ids", [])),
                            "source_chunk_indices": list(seg.get("source_chunk_indices", [])),
                        }
                    )
                chunk_start = chunk_end
        merged = capped

    return merged


def _smooth_speaker_turns(
    segments: List[dict],
    *,
    min_turn_duration: float = 0.7,
    max_gap: float = 0.5,
) -> List[dict]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (float(s["start_time"]), float(s["end_time"])))
    cleaned: List[dict] = []
    for seg in ordered:
        start_time = float(seg["start_time"])
        end_time = float(seg["end_time"])
        if end_time <= start_time:
            continue
        cleaned.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "speaker_id": str(seg["speaker_id"]),
                "chunk_index": seg.get("chunk_index"),
            }
        )

    i = 0
    while i < len(cleaned):
        seg = cleaned[i]
        dur = float(seg["end_time"]) - float(seg["start_time"])
        if dur >= min_turn_duration:
            i += 1
            continue

        prev = cleaned[i - 1] if i - 1 >= 0 else None
        nxt = cleaned[i + 1] if i + 1 < len(cleaned) else None

        if prev is not None and nxt is not None and prev["speaker_id"] == nxt["speaker_id"]:
            gap1 = float(seg["start_time"]) - float(prev["end_time"])
            gap2 = float(nxt["start_time"]) - float(seg["end_time"])
            if gap1 <= max_gap and gap2 <= max_gap:
                prev["end_time"] = max(float(prev["end_time"]), float(nxt["end_time"]))
                del cleaned[i : i + 2]
                i = max(i - 1, 0)
                continue

        if prev is not None:
            gap = float(seg["start_time"]) - float(prev["end_time"])
            if gap <= max_gap:
                prev["end_time"] = max(float(prev["end_time"]), float(seg["end_time"]))
                del cleaned[i]
                i = max(i - 1, 0)
                continue

        if nxt is not None:
            gap = float(nxt["start_time"]) - float(seg["end_time"])
            if gap <= max_gap:
                nxt["start_time"] = min(float(nxt["start_time"]), float(seg["start_time"]))
                del cleaned[i]
                continue

        i += 1

    return cleaned


def _ends_with_sentence_punctuation(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_END_PUNCTUATION)


def _merge_transcribed_utterances(
    utterances: List[dict],
    transcripts_by_utt_idx: Dict[int, str],
) -> List[dict]:
    merged: List[dict] = []

    for utterance in utterances:
        utterance_index = int(utterance["utterance_index"])
        text = transcripts_by_utt_idx.get(utterance_index, "").strip()
        if not text:
            continue

        current = dict(utterance)
        current["text"] = text
        current["source_utterance_indices"] = [utterance_index]

        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        same_speaker = _canonicalize_speaker_id(str(previous["speaker_id"])) == _canonicalize_speaker_id(str(current["speaker_id"]))

        if same_speaker and not _ends_with_sentence_punctuation(str(previous.get("text", ""))):
            previous["end_time"] = max(float(previous["end_time"]), float(current["end_time"]))
            previous["text"] = f"{str(previous.get('text', '')).rstrip()} {text}".strip()
            previous_indices = list(previous.get("source_utterance_indices", []))
            previous_indices.extend(current["source_utterance_indices"])
            previous["source_utterance_indices"] = previous_indices
            previous_source_speaker_ids = list(previous.get("source_speaker_ids", []))
            previous_source_speaker_ids.extend(list(current.get("source_speaker_ids", [])))
            previous["source_speaker_ids"] = previous_source_speaker_ids
            previous_source_chunk_indices = list(previous.get("source_chunk_indices", []))
            previous_source_chunk_indices.extend(list(current.get("source_chunk_indices", [])))
            previous["source_chunk_indices"] = previous_source_chunk_indices
            continue

        merged.append(current)

    return merged


def _build_voice_prompt_candidates(
    utterances: List[dict],
    voice_prompt_paths_by_idx: Dict[int, Path],
    *,
    min_duration: float = 3.0,
    max_duration: float = 15.0,
) -> Dict[int, Dict[str, List[Path]]]:
    prompt_candidates: Dict[int, Dict[str, List[Path]]] = {}

    for utterance in utterances:
        utterance_index = int(utterance["utterance_index"])
        prompt_path = voice_prompt_paths_by_idx.get(utterance_index)
        if prompt_path is None:
            continue
        duration = float(utterance["end_time"]) - float(utterance["start_time"])
        if duration < min_duration or duration > max_duration:
            continue
        source_speaker_ids = {str(speaker_id) for speaker_id in utterance.get("source_speaker_ids", [])}
        source_chunk_indices = {int(chunk_index) for chunk_index in utterance.get("source_chunk_indices", []) if chunk_index is not None}
        if len(source_speaker_ids) != 1 or len(source_chunk_indices) != 1:
            continue
        chunk_index = next(iter(source_chunk_indices))
        speaker_id = next(iter(source_speaker_ids))
        prompt_candidates.setdefault(chunk_index, {}).setdefault(speaker_id, []).append(prompt_path)

    return prompt_candidates


def _select_voice_prompts_for_sample(
    utterances: List[dict],
    prompt_candidates: Dict[int, Dict[str, List[Path]]],
) -> List[str] | None:
    selected_prompts: List[str] = []

    for utterance in utterances:
        source_speaker_ids = {str(speaker_id) for speaker_id in utterance.get("source_speaker_ids", [])}
        source_chunk_indices = {int(chunk_index) for chunk_index in utterance.get("source_chunk_indices", []) if chunk_index is not None}
        if len(source_speaker_ids) != 1 or len(source_chunk_indices) != 1:
            return None
        chunk_index = next(iter(source_chunk_indices))
        speaker_id = next(iter(source_speaker_ids))
        speaker_prompt_candidates = prompt_candidates.get(chunk_index, {}).get(speaker_id)
        if not speaker_prompt_candidates:
            return None
        selected_prompts.append(str(random.choice(speaker_prompt_candidates).resolve()))

    return selected_prompts


def _format_sample_text(*, utterances: List[dict], speaker_texts: Dict[int, str]) -> str:
    local_speaker_map: Dict[str, int] = {}
    next_local_idx = 0
    parts: List[str] = []
    last_local_idx: int | None = None

    for utt in utterances:
        speaker_id = _canonicalize_speaker_id(str(utt["speaker_id"]))
        local_idx = local_speaker_map.get(speaker_id)
        if local_idx is None:
            local_idx = next_local_idx
            local_speaker_map[speaker_id] = local_idx
            next_local_idx += 1

        text = str(utt.get("text", "")).strip()
        if not text:
            text = speaker_texts.get(int(utt["utterance_index"]), "").strip()
        if not text:
            continue
        if last_local_idx == local_idx and parts:
            parts[-1] = parts[-1].rstrip() + " " + text
        else:
            prefix = f"Speaker {local_idx}: "
            if not parts:
                parts.append(prefix + text)
            else:
                parts.append("\n" + prefix + text)
        last_local_idx = local_idx

    return "".join(parts).strip()


def _pack_utterances_into_samples(
    utterances: List[dict],
    *,
    max_duration: float | None,
    max_num_speakers: int | None,
) -> List[List[dict]]:
    if not utterances:
        return []

    samples: List[List[dict]] = []
    current: List[dict] = []
    current_dur = 0.0
    max_d = float(max_duration) if max_duration is not None else None

    current_speakers: set[str] = set()
    max_spk = int(max_num_speakers) if max_num_speakers is not None else None

    for utt in utterances:
        dur = float(utt["end_time"]) - float(utt["start_time"])
        if dur <= 0:
            continue
        spk = _canonicalize_speaker_id(str(utt["speaker_id"]))
        if not current:
            current = [utt]
            current_dur = dur
            current_speakers = {spk}
            continue

        would_exceed_duration = max_d is not None and (current_dur + dur) > max_d
        would_add_speaker = spk not in current_speakers
        would_exceed_speakers = False
        if max_spk is not None and max_spk > 0:
            would_exceed_speakers = would_add_speaker and (len(current_speakers) + 1) > max_spk

        if would_exceed_duration or would_exceed_speakers:
            samples.append(current)
            current = [utt]
            current_dur = dur
            current_speakers = {spk}
            continue
        current.append(utt)
        current_dur += dur
        current_speakers.add(spk)

    if current:
        samples.append(current)

    return samples


def create_diarized_samples(
    *,
    audio_path: Path,
    samples_dir: Path,
    work_dir: Path,
    model_id: str,
    device: str | None,
    batch_size: int,
    max_duration: float | None = None,
    max_num_speakers: int | None = None,
    chunk_size: float = _DEFAULT_DIARIZATION_CHUNK_SIZE,
    vad_model=None,
    diarization_state: Dict[str, object] | None = None,
) -> Dict[Path, Dict[str, object]]:
    _require_torch()
    samples_dir.mkdir(parents=True, exist_ok=True)

    diarization_segments = diarize_audio(
        audio_path,
        work_dir,
        chunk_size=chunk_size,
        vad_model=vad_model,
        diarization_state=diarization_state,
    )
    if not diarization_segments:
        return {}

    diarization_segments = _smooth_speaker_turns(diarization_segments)
    utterances = _merge_adjacent_speaker_segments(diarization_segments, max_duration=max_duration)
    if not utterances:
        return {}

    audio_tensor, sample_rate = torchaudio.load(str(audio_path))  # type: ignore[union-attr]
    if audio_tensor.ndim != 2:
        raise RuntimeError(f"Expected audio tensor with shape (channels, samples), got {tuple(audio_tensor.shape)}")
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    utterance_paths: List[Path] = []
    utterance_paths_by_idx: Dict[int, Path] = {}
    voice_prompt_paths_by_idx: Dict[int, Path] = {}
    utterance_meta: List[dict] = []
    utterances_dir = work_dir / "utterances" / audio_path.stem
    voice_prompts_dir = work_dir / "voice_prompts" / audio_path.stem
    utterances_dir.mkdir(parents=True, exist_ok=True)
    voice_prompts_dir.mkdir(parents=True, exist_ok=True)

    for utt_idx, utt in enumerate(utterances):
        start = int(float(utt["start_time"]) * sample_rate)
        end = int(float(utt["end_time"]) * sample_rate)
        end = min(end, audio_tensor.shape[1])
        if end <= start:
            continue
        utt_tensor = audio_tensor[:, start:end]
        utt_tensor, out_sr = _resample_waveform(utt_tensor, sample_rate, _OUTPUT_SAMPLE_RATE)
        utt_path = utterances_dir / f"{audio_path.stem}_utt_{utt_idx:05d}{_SPEAKER_DELIMITER}{utt['speaker_id']}.wav"
        torchaudio.save(str(utt_path), utt_tensor, out_sr)  # type: ignore[union-attr]
        utterance_paths.append(utt_path)
        utterance_paths_by_idx[utt_idx] = utt_path
        utterance_duration = float(utt["end_time"]) - float(utt["start_time"])
        if 3.0 <= utterance_duration <= 15.0:
            voice_prompt_path = voice_prompts_dir / f"{audio_path.stem}_prompt_{utt_idx:05d}{_SPEAKER_DELIMITER}{utt['speaker_id']}.wav"
            torchaudio.save(str(voice_prompt_path), utt_tensor, out_sr)  # type: ignore[union-attr]
            voice_prompt_paths_by_idx[utt_idx] = voice_prompt_path
        utterance_meta.append({
            "start_time": float(utt["start_time"]),
            "end_time": float(utt["end_time"]),
            "speaker_id": str(utt["speaker_id"]),
            "utterance_index": utt_idx,
            "chunk_index": utt.get("chunk_index"),
            "source_speaker_ids": list(utt.get("source_speaker_ids", [])),
            "source_chunk_indices": list(utt.get("source_chunk_indices", [])),
        })

    if not utterance_paths:
        return {}

    from databuilder.transcribe import transcribe_audio as _transcribe_audio

    utt_transcripts_by_path = _transcribe_audio(
        utterance_paths,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
    )

    transcripts_by_utt_idx: Dict[int, str] = {}
    for path, text in utt_transcripts_by_path.items():
        stem = path.stem
        if "_utt_" not in stem:
            continue
        try:
            idx_part = stem.split("_utt_", 1)[1].split(_SPEAKER_DELIMITER, 1)[0]
            utt_idx = int(idx_part)
        except Exception:
            continue
        transcripts_by_utt_idx[utt_idx] = text

    utterance_meta = [u for u in utterance_meta if transcripts_by_utt_idx.get(int(u["utterance_index"]))]
    voice_prompt_candidates = _build_voice_prompt_candidates(utterance_meta, voice_prompt_paths_by_idx)
    utterance_meta = _merge_transcribed_utterances(utterance_meta, transcripts_by_utt_idx)
    if not utterance_meta:
        return {}

    packed = _pack_utterances_into_samples(
        utterance_meta,
        max_duration=max_duration,
        max_num_speakers=max_num_speakers,
    )
    if not packed:
        return {}

    output: Dict[Path, Dict[str, object]] = {}
    for sample_idx, sample_utts in enumerate(packed):
        sample_audio_chunks: List[torch.Tensor] = []
        speaker_texts: Dict[int, str] = {}
        for utt in sample_utts:
            utt_i = int(utt["utterance_index"])
            speaker_texts[utt_i] = str(utt.get("text", "")).strip() or transcripts_by_utt_idx.get(utt_i, "")

            source_utterance_indices = utt.get("source_utterance_indices") or [utt_i]
            for source_utt_i in source_utterance_indices:
                utt_path = utterances_dir / f"{audio_path.stem}_utt_{int(source_utt_i):05d}{_SPEAKER_DELIMITER}{utt['speaker_id']}.wav"
                if not utt_path.exists():
                    continue
                chunk, chunk_sr = torchaudio.load(str(utt_path))  # type: ignore[union-attr]
                if chunk.ndim != 2:
                    continue
                if chunk_sr != _OUTPUT_SAMPLE_RATE:
                    chunk, _ = _resample_waveform(chunk, chunk_sr, _OUTPUT_SAMPLE_RATE)
                if chunk.shape[0] > 1:
                    chunk = chunk.mean(dim=0, keepdim=True)
                sample_audio_chunks.append(chunk)

        if not sample_audio_chunks:
            continue

        sample_audio = torch.cat(sample_audio_chunks, dim=1)
        sample_path = samples_dir / f"{audio_path.stem}_sample_{sample_idx:05d}.wav"
        torchaudio.save(str(sample_path), sample_audio, _OUTPUT_SAMPLE_RATE)  # type: ignore[union-attr]

        sample_text = _format_sample_text(utterances=sample_utts, speaker_texts=speaker_texts)
        if not sample_text:
            continue
        sample_start_time = min(float(utt["start_time"]) for utt in sample_utts)
        sample_end_time = max(float(utt["end_time"]) for utt in sample_utts)
        output[sample_path] = {
            "text": sample_text,
            "duration": max(0.0, sample_end_time - sample_start_time),
        }
        voice_prompts = _select_voice_prompts_for_sample(sample_utts, voice_prompt_candidates)
        if voice_prompts is not None:
            output[sample_path]["voice_prompts"] = voice_prompts

    return output


def build_dataset(transcripts: Dict[Path, str | Dict[str, object]], speaker_prefix: str = "Speaker 0: ") -> Dataset:
    """Create a Hugging Face dataset where audio references local files."""
    if not transcripts:
        return Dataset.from_dict({"audio": [], "text": [], "duration": [], "voice_prompts": []})

    items = []
    for path, value in sorted(transcripts.items(), key=lambda item: str(item[0])):
        if isinstance(value, dict):
            text = str(value.get("text", ""))
            duration = value.get("duration")
            voice_prompts = value.get("voice_prompts")
            speaker_id = value.get("speaker_id")
        else:
            text = value
            duration = None
            voice_prompts = []
            speaker_id = None
        if isinstance(voice_prompts, list):
            normalized_voice_prompts = [str(prompt_path) for prompt_path in voice_prompts]
        else:
            normalized_voice_prompts = []
        stripped = text.strip()
        if stripped.startswith("Speaker "):
            formatted_text = stripped
        elif isinstance(speaker_id, str) and speaker_id.strip():
            formatted_text = f"{_format_speaker_prefix(speaker_id)}{stripped}".strip()
        else:
            formatted_text = f"{speaker_prefix}{stripped}".strip()
        items.append({"audio": str(path), "text": formatted_text, "duration": duration, "voice_prompts": normalized_voice_prompts})
    dataset = Dataset.from_list(items)
    dataset = dataset.cast_column("audio", Audio())
    dataset = dataset.cast_column("voice_prompts", HFSequence(Audio()))
    return dataset


def push_dataset(dataset: Dataset, repo_id: str | Path, token: str | None = None) -> None:
    """Upload dataset to the specified Hugging Face Hub repo."""
    if isinstance(repo_id, Path):
        dataset.save_to_disk(dataset_path=repo_id)
    else:
        dataset.push_to_hub(repo_id, token=token)


def run_pipeline(
    input_dir: Path,
    repo_id: str | Path,
    work_dir: Path,
    model_id: str = "openai/whisper-base",
    device: str | None = None,
    token: str | None = None,
    speaker_prefix: str = "Speaker 0: ",
    enable_diarization: bool = True,
    max_duration: float | None = None,
    max_num_speakers: int | None = None,
    batch_size: int = 8,
    chunk_size: float = _DEFAULT_DIARIZATION_CHUNK_SIZE,
) -> Dataset:
    """Execute segmentation, transcription, dataset creation, and upload."""
    work_dir.mkdir(parents=True, exist_ok=True)
    vad_model = _load_silero_vad_model()
    diarization_state: Dict[str, object] = {"device": _resolve_diarization_device(device)}
    segments_dir = work_dir / "segments"
    transcripts_dir = work_dir / "transcripts"

    segments: List[Path] = []
    audio_files = list(_iter_audio_files(input_dir))
    diarized_samples_dir = work_dir / "samples"
    if enable_diarization:
        diarized_transcripts: Dict[Path, Dict[str, object]] = {}
        for audio_path in tqdm(audio_files, desc="Diarize", unit="file"):
            diarized_transcripts.update(
                create_diarized_samples(
                    audio_path=audio_path,
                    samples_dir=diarized_samples_dir,
                    work_dir=work_dir,
                    model_id=model_id,
                    device=device,
                    batch_size=batch_size,
                    max_duration=max_duration,
                    max_num_speakers=max_num_speakers,
                    chunk_size=chunk_size,
                    vad_model=vad_model,
                    diarization_state=diarization_state,
                )
            )
        if diarized_transcripts:
            transcripts = diarized_transcripts
        else:
            segments = segment_audio(input_dir, segments_dir, vad_model=vad_model, max_duration=max_duration)
            from databuilder.transcribe import transcribe_audio

            transcripts = transcribe_audio(segments, model_id=model_id, device=device, batch_size=batch_size)
    else:
        segments = segment_audio(input_dir, segments_dir, vad_model=vad_model, max_duration=max_duration)
        from databuilder.transcribe import transcribe_audio

        transcripts = transcribe_audio(segments, model_id=model_id, device=device, batch_size=batch_size)

    if transcripts:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        for path, value in transcripts.items():
            text = value if isinstance(value, str) else str(value.get("text", ""))
            transcript_path = transcripts_dir / f"{path.stem}.txt"
            transcript_path.write_text(text + "\n", encoding="utf-8")

    dataset = build_dataset(transcripts, speaker_prefix=speaker_prefix)
    push_dataset(dataset, repo_id, token=token)
    return dataset
