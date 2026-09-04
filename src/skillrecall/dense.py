"""Optional meaning-level scorer backed by a small local embedding model.

Install with `pip install "skillrecall[dense]"`. The model file is downloaded
once and cached by the embedding library; after that everything runs
offline and deterministically. Embeddings are memoised per text so the edit
loop, which rescores the same neighbours many times, pays for them once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedScorer:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("dense scoring needs the 'dense' extra: pip install 'skillrecall[dense]'") from e
        self._model = TextEmbedding(model_name=model)
        self._cache: dict[str, list[float]] = {}
        self.label = model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, vec in zip(missing, self._model.embed(missing), strict=False):
                v = [float(x) for x in vec]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                self._cache[t] = [x / n for x in v]
        return [self._cache[t] for t in texts]

    def similarities(self, query: str, texts: Sequence[str]) -> list[float]:
        q = self._embed([query])[0]
        return [sum(a * b for a, b in zip(q, v, strict=False)) for v in self._embed(texts)]


def load_dense(model: str | None = None):
    return FastEmbedScorer(model or DEFAULT_MODEL)
