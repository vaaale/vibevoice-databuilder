# VibeVoice Dataset Preparer

A tool to prepare datasets for training VibeVoice.

## Installation

```
uv pip install git+https://github.com/vibevoice-community/databuilder
```

## Quick start

```
databuilder input/ username/huggingface_dataset_name
```

`input/` is a folder that should have raw MP3/WAV files. Any length.

## What's new in the `multispeaker` branch

The original pipeline processes single-speaker audio only:

**Silero VAD → Whisper Transcription → Hugging Face Dataset**

The `multispeaker` branch extends the tool with full **multi-speaker diarization** support and several other improvements:

### Speaker diarization
- Integrates **NVIDIA NeMo SortformerEncLabelModel** (`nvidia/diar_sortformer_4spk-v1`) for speaker diarization, supporting up to 4 speakers per segment.
- VAD timestamps are grouped into configurable chunks (`--chunk-size`) before being passed to the diarization model.
- Short speaker turns are smoothed and adjacent same-speaker segments are merged to produce clean utterances.

### Multi-speaker transcript formatting
- Each utterance is prefixed with a locally-numbered speaker label (e.g. `Speaker 0:`, `Speaker 1:`).
- Consecutive utterances from the same speaker are merged at sentence boundaries for natural text flow.

### Voice prompt extraction
- Utterances between 3 and 15 seconds are automatically saved as **voice prompt** candidates.
- The resulting dataset includes a `voice_prompts` column (a sequence of `Audio` features) that can be used for voice-conditioned TTS training.

### Improved transcription
- Whisper transcription has been moved to a dedicated module (`transcribe.py`) and now uses **batched inference** with `AutoModelForSpeechSeq2Seq` and `AutoProcessor` for significantly faster throughput.
- Supports `librosa`-based resampling and proper attention masking for batched decoding.

### Sample packing
- Diarized utterances are packed into samples respecting a maximum duration (`--max-duration`) and a maximum number of speakers (`--max-num-speakers`), producing dataset entries that are well-suited for multi-speaker TTS training.

### Audio resampling
- All output audio is resampled to **24 kHz** for consistency.

### New CLI options
| Option | Default | Description |
|---|---|---|
| `--max-duration` | `25` | Maximum duration (seconds) per sample |
| `--max-num-speakers` | `4` | Upper bound for speakers per sample |
| `--batch-size` | `8` | Whisper batch size |
| `--chunk-size` | `300` | Maximum duration (seconds) for VAD-built diarization chunks |

### Other changes
- **Local dataset saving** — passing a filesystem path as `repo_id` saves the dataset to disk instead of pushing to the Hub.
- **Graceful dependency handling** — missing optional packages (`torch`, `torchaudio`, `nemo`, etc.) raise clear errors instead of import-time crashes.
- **OOM fallback** — if the diarization model runs out of GPU memory it automatically falls back to CPU.