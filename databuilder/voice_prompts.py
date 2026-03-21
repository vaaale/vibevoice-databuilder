"""Shared voice prompt selection utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

VOICE_PROMPT_MIN_DURATION = 3.0
VOICE_PROMPT_MAX_DURATION = 15.0


def build_speaker_prompt_index(
    candidates: List[dict],
    *,
    min_duration: float = VOICE_PROMPT_MIN_DURATION,
    max_duration: float = VOICE_PROMPT_MAX_DURATION,
) -> Dict[str, List[Path]]:
    """Group voice prompt candidates by ``speaker_id``.

    Only candidates whose ``duration`` falls within
    [*min_duration*, *max_duration*] are kept.

    Each element of *candidates* must contain the keys
    ``speaker_id``, ``path``, and ``duration``.
    """
    index: Dict[str, List[Path]] = {}
    for c in candidates:
        duration = float(c["duration"])
        if not (min_duration <= duration <= max_duration):
            continue
        speaker_id = str(c["speaker_id"])
        path = Path(c["path"]) if not isinstance(c["path"], Path) else c["path"]
        index.setdefault(speaker_id, []).append(path)
    return index


def select_voice_prompts(
    speaker_ids: List[str],
    prompt_index: Dict[str, List[Path]],
    *,
    exclude: set[str] | None = None,
) -> List[str] | None:
    """Pick one random voice prompt per speaker from *prompt_index*.

    Returns a list of resolved path strings (one per speaker in the same
    order as *speaker_ids*), or ``None`` if any speaker has no available
    candidate.

    *exclude* is an optional set of path strings to skip (e.g. the
    sample's own audio file).
    """
    exclude_set = exclude or set()
    result: List[str] = []
    for speaker_id in speaker_ids:
        candidates = prompt_index.get(speaker_id, [])
        available = [p for p in candidates if str(p) not in exclude_set]
        if not available:
            return None
        result.append(str(random.choice(available).resolve()))
    return result
