"""Text primitives: tokenization, n-grams, sentence splitting, and token counting.

Everything here is pure Python with precompiled regular expressions. The
functions are called thousands of times per assessment, so they avoid
allocations where it matters and never import optional packages eagerly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
_PIECE = re.compile(r"[A-Za-z]+|\d+|\n+|[^\sA-Za-z\d]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_CODE = re.compile(r"`([^`]*)`")
_MD_EMPH = re.compile(r"[*_]{1,3}")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")

# Function words and generic verbs that carry no routing signal. Domain words
# never appear here; the ranking is stable with or without the list, but the
# term suggestions shown to authors must not contain filler.
STOP = frozenset(
    """
    a an the and or but of to in on for with by from as at is are be been was
    were this that these those it its into if then than not no when use used
    using your you their which any all can will may should also each every
    our ours does did has have had what who how why where some someone something
    please help need want make get let just one two about over after before
    would could like via per within without between across through them they
    there here than more most much many very own same other such only both
    add set put show give take keep look turn run running
        """.split()
)

MIN_TOKEN_LEN = 3


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of at least three characters, minus stop words."""
    return [t for t in _WORD.findall(text.lower()) if len(t) >= MIN_TOKEN_LEN and t not in STOP]


def bigrams(toks: list[str]) -> list[str]:
    return [f"{toks[i]}_{toks[i + 1]}" for i in range(len(toks) - 1)]


def terms(text: str) -> list[str]:
    """Unigrams plus adjacent bigrams; the unit the lexical scorer indexes."""
    toks = tokens(text)
    return toks + bigrams(toks)


def term_counts(text: str) -> Counter[str]:
    return Counter(terms(text))


def sentences(text: str) -> list[str]:
    """Split prose into sentences, dropping empties. Newline pairs also split."""
    out: list[str] = []
    for part in _SENTENCE.split(text):
        part = _WS.sub(" ", part).strip()
        if part:
            out.append(part)
    return out


def strip_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _MD_EMPH.sub("", text)
    return _WS.sub(" ", text).strip()


def name_words(name: str) -> list[str]:
    """Split a skill name like `landing-page-audit` into words."""
    return [w for w in re.split(r"[-_\s./]+", name.lower()) if w]


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = f" {s.lower()} "
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def name_similarity(a: str, b: str) -> float:
    """Jaccard over character trigrams of two names. 1.0 means identical."""
    ga, gb = char_ngrams(a), char_ngrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


class TokenCounter:
    """Count tokens exactly when a tokenizer is installed, else estimate.

    The estimate is deliberately simple and deterministic: one token per short
    word or punctuation mark, more for long words and digit runs. It tracks
    real tokenizers to within about ten percent on English prose, which is
    enough for relative comparisons. `exact` tells callers which one ran.
    """

    __slots__ = ("_encode", "exact", "label")

    def __init__(self, prefer_exact: bool = True) -> None:
        self._encode = None
        self.exact = False
        self.label = "estimated"
        if prefer_exact:
            try:
                import tiktoken  # type: ignore

                self._encode = tiktoken.get_encoding("o200k_base").encode
                self.exact = True
                self.label = "o200k_base"
            except Exception:  # pragma: no cover - optional dependency
                self._encode = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encode is not None:
            return len(self._encode(text))
        return _estimate_tokens(text)


@lru_cache(maxsize=4096)
def _estimate_tokens(text: str) -> int:
    total = 0
    for piece in _PIECE.findall(text):
        c = piece[0]
        if c.isalpha():
            n = len(piece)
            total += 1 if n <= 7 else math.ceil(n / 6)
        elif c.isdigit():
            total += math.ceil(len(piece) / 3)
        elif c == "\n":
            total += 1
        else:
            total += 1
    return total


def top_terms(counter: Counter[str], k: int, exclude: Iterable[str] = ()) -> list[str]:
    ex = set(exclude)
    return [t for t, _ in counter.most_common() if t not in ex][:k]
