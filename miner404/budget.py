"""Per-prompt call and credit accounting with the round's rate file.

The validator charges every dispatched call: input tokens at the full input weight (cached included) and output
tokens (reasoning included) at the output weight, switching the whole request to long-context weights above
272,000 input tokens. Limits: 60 calls and 5,000,000 credits per prompt, checked on the calls actually consumed at
replay (miner-runtime accounting.py): a prompt over either limit is rejected outright. So every call reserves its
worst case before it is dispatched (input estimate + its full output cap, which the API enforces), and the prompt can
never pass the limits even if every call used its whole cap.

v9: the worst case is estimated per image detail instead of a flat 3,000 tokens per image. Requests carry up to 14
images, so the flat figure tripled the reservation of a selection call and refused the last affordable rewrite steps
of a prompt. Every earlier call is charged at its actual usage, so the margin only has to absorb an input
underestimate on the stages still running when the last one was admitted (a rewrite, its repair, two selections and
the pose selection: about 50 images and 100,000 text tokens, every one already overestimated). 2% (100,000 credits)
covers an image costing 1,000 tokens more than estimated in all of them.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text())
CREDIT_LIMIT = PRICING["credits_per_prompt"]
CALL_LIMIT = PRICING["calls_per_prompt"]
LONG_CONTEXT = PRICING["long_context_above_input_tokens"]
SAFETY = 0.02  # keep 2% of the credit limit unspent
TOKENS_PER_IMAGE = 1800  # conservative per-image input estimate at detail "high" (a 1024 px image is ~800-1,100)
TOKENS_PER_LOW_IMAGE = 300  # detail "low" is a fixed small thumbnail (~85 tokens on current models)
CHARS_PER_TOKEN = 3.0  # conservative (code and JSON tokenize densely)


def call_credits(model: str, input_tokens: int, output_tokens: int) -> float:
    weights = PRICING["models"][model]
    long = input_tokens > LONG_CONTEXT
    return (input_tokens * weights["long_input_credits_per_token" if long else "input_credits_per_token"]
            + output_tokens * weights["long_output_credits_per_token" if long else "output_credits_per_token"])


def worst_case(model: str, text_chars: int, images: int, max_output_tokens: int, low_images: int = 0) -> float:
    """Credits of a call that uses its whole output cap. `images` are high/auto detail, `low_images` detail "low"."""
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE + low_images * TOKENS_PER_LOW_IMAGE
    return call_credits(model, input_tokens, max_output_tokens)


def content_worst_case(model: str, instructions: str, content: list[dict], max_output_tokens: int) -> float:
    """worst_case() of a request from its instructions and input items."""
    text_chars = len(instructions) + sum(len(item.get("text", "")) for item in content)
    low = sum(1 for item in content if item.get("type") == "input_image" and item.get("detail") == "low")
    high = sum(1 for item in content if item.get("type") == "input_image") - low
    return worst_case(model, text_chars, high, max_output_tokens, low)


class Budget:
    """Thread-safe ledger of one prompt's calls. Reservations make concurrent stages safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.credits = 0.0
        self._reserved_calls = 0
        self._reserved_credits = 0.0

    def allows(self, estimates: list[float], keep_calls: int = 0, keep_credits: float = 0.0) -> bool:
        with self._lock:
            calls = self.calls + self._reserved_calls + len(estimates) + keep_calls
            credits = self.credits + self._reserved_credits + sum(estimates) + keep_credits
            return calls <= CALL_LIMIT and credits <= CREDIT_LIMIT * (1 - SAFETY)

    def reserve(self, estimate: float, keep_calls: int = 0, keep_credits: float = 0.0) -> bool:
        """Reserve one call, but only if `keep_calls` / `keep_credits` stay affordable after it (v9: a rewrite or
        repair is admitted only if the selection that judges its result can still be paid for)."""
        with self._lock:
            if (self.calls + self._reserved_calls + 1 + keep_calls > CALL_LIMIT
                    or self.credits + self._reserved_credits + estimate + keep_credits > CREDIT_LIMIT * (1 - SAFETY)):
                return False
            self._reserved_calls += 1
            self._reserved_credits += estimate
            return True

    def settle(self, estimate: float, model: str, usage: object | None) -> None:
        """Release a reservation and charge the actual usage (or the estimate when usage is unknown)."""
        if usage is not None:
            charged = call_credits(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        else:
            charged = estimate
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_credits -= estimate
            self.calls += 1
            self.credits += charged
