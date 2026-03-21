"""Shared voice prompt selection utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

VOICE_PROMPT_MIN_DURATION = 3.0
VOICE_PROMPT_MAX_DURATION = 15.0


class SpeakerPromptCache:
    """Rolling cache of the last *max_size* voice prompt paths per speaker.

    Call :meth:`add` as you process records so that later records can
    draw from prompts seen earlier.  :meth:`select` picks a random
    candidate while respecting an optional exclusion set.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._cache: Dict[str, List[Path]] = {}

    def add(self, speaker_id: str, path: str | Path) -> None:
        """Append *path* to the cache for *speaker_id*."""
        p = Path(path) if isinstance(path, str) else path
        bucket = self._cache.setdefault(speaker_id, [])
        bucket.append(p)
        if len(bucket) > self._max_size:
            del bucket[:-self._max_size]

    def select(
        self,
        speaker_id: str,
        *,
        exclude: set[str] | None = None,
    ) -> str | None:
        """Pick a random cached prompt for *speaker_id*.

        Returns a resolved path string, or ``None`` when no candidate
        is available after filtering out *exclude* paths.
        """
        exclude_set = exclude or set()
        candidates = self._cache.get(speaker_id, [])
        available = [p for p in candidates if str(p) not in exclude_set]
        if not available:
            return None
        return str(random.choice(available).resolve())


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
