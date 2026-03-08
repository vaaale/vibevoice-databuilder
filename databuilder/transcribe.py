from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, List

try:
    import librosa
except ModuleNotFoundError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]
try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover
    sf = None  # type: ignore[assignment]
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForSpeechSeq2Seq = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]


def _require_transcription_deps() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if sf is None:
        missing.append("soundfile")
    if torch is None:
        missing.append("torch")
    if tqdm is None:
        missing.append("tqdm")
    if AutoModelForSpeechSeq2Seq is None or AutoProcessor is None:
        missing.append("transformers")
    if librosa is None:
        missing.append("librosa")
    if missing:
        raise RuntimeError(
            "Transcription dependencies are missing: "
            + ", ".join(missing)
            + ". Install them to use transcribe_audio()."
        )

def _read_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path))

    # Convert stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sr,
            res_type="kaiser_best",  # high quality
        )

    return audio


def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _resolve_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_audio(
    paths: Sequence[Path],
    model_id: str = "openai/whisper-base",
    device: str | None = None,
    batch_size: int = 8,
    language: str | None = None,
) -> Dict[Path, str]:
    _require_transcription_deps()
    if not paths:
        return {}

    resolved_device = device or _resolve_device()
    torch_device = torch.device(resolved_device)
    torch_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(torch_device)
    model.eval()

    path_list = list(paths)
    transcripts: Dict[Path, str] = {}

    generate_kwargs = {}
    if language is not None:
        generate_kwargs["language"] = language
        generate_kwargs["task"] = "transcribe"

    for batch_paths in tqdm(list(_batched(path_list, batch_size)), desc="Transcribe", unit="batch"):
        audios: List[np.ndarray] = [_read_audio(path) for path in batch_paths]

        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",        # critical fix
            truncation=True,             # critical fix
            return_attention_mask=True,  # recommended for batched inference
        )

        input_features = inputs.input_features.to(torch_device, dtype=torch_dtype)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=128,
                **generate_kwargs,
            )

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for path, text in zip(batch_paths, texts, strict=True):
            transcripts[path] = text.strip()

    return transcripts
