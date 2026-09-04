"""Optional reference router: ask a model to pick a skill the way a host would.

Install with `pip install "skillrecall[router]"` and provide credentials the
Anthropic SDK understands. The reference run is sampled (a few dozen tasks)
and used two ways: to report how the model itself picks, and to state how
well the fast local scorer agrees with it. It is never required.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

from .scoring import Doc, Router
from .tasks import Task

DEFAULT_MODEL = "claude-opus-5"
_INT = re.compile(r"\d+")

SYSTEM = (
    "You are the routing layer of a coding agent. You will be shown a user request and a numbered list of "
    "installed skills, each with its name and description. Reply with the number of the single skill that "
    "should handle the request, or 0 if none applies. Reply with the number only."
)


@dataclass(slots=True)
class ReferenceResult:
    n: int
    recall: float
    agreement: float
    model: str

    def as_dict(self) -> dict:
        return {"n": self.n, "recall": round(self.recall, 4), "agreement": round(self.agreement, 4), "model": self.model}


class ReferenceRouter:
    def __init__(self, model: str = DEFAULT_MODEL, seed: int = 7) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("the reference router needs the 'router' extra: pip install 'skillrecall[router]'") from e
        self._client = anthropic.Anthropic()
        self.model = model
        self._rng = random.Random(seed)

    def choose(self, task: str, docs: Sequence[Doc]) -> int:
        order = list(range(len(docs)))
        self._rng.shuffle(order)
        listing = "\n".join(f"{k + 1}. {docs[i].text}" for k, i in enumerate(order))
        prompt = f"Request: {task}\n\nInstalled skills:\n{listing}\n\nAnswer with the number only."
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return -1
        text = "".join(b.text for b in response.content if b.type == "text")
        m = _INT.search(text)
        if not m:
            return -1
        k = int(m.group(0))
        if k <= 0 or k > len(order):
            return -1
        return order[k - 1]


def reference_check(router: Router, self_idx: int, own: Sequence[Task], n: int = 30, model: str = DEFAULT_MODEL, seed: int = 7) -> ReferenceResult:
    ref = ReferenceRouter(model, seed)
    rng = random.Random(seed)
    sample = own if len(own) <= n else rng.sample(list(own), n)
    hits = 0
    agree = 0
    for t in sample:
        picked = ref.choose(t.text, router.docs)
        local = router.winner(router.score(t.text))
        hits += picked == self_idx
        agree += picked == local
    m = max(1, len(sample))
    return ReferenceResult(len(sample), hits / m, agree / m, model)
