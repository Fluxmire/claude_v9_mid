"""The plan this submission runs (Claude v9).

One deep, gated lineage. In round 3, on the same 128 prompts against the same boss, the margin over the boss rose with
the number of gated rewrites on the final candidate: about +0.20 with ~4 steps, +0.31 with ~6 steps over two lineages,
+0.43 to +0.47 with one lineage whose 10 planned steps the budget cut to about 6 (taohunter251/lark-ridge, the round's
winner). So: the best of six drafts, then gated rewrites.

- Every step is admitted only if its rewrite, a repair, the two selection calls that judge the child and the pose stage
  at the end still fit the budget at their worst case; a step that does not fit is skipped without spending anything,
  so a cheaper step after it may still run.
- One Astra rewrite early, for a second model's eyes; Sol high and xhigh after it, because one Astra rewrite costs about
  as much as three Sol rewrites. The focus rotates so successive rewrites work on different things.

Three plans, the same code (ACTIVE_PLAN at the bottom picks one; estimates from an offline model of measured usage):
- DEEP_PLAN: up to 16 steps, as many as the 5M credits allow (about 11), the tail in Sol medium (16k output cap) to use
  what the last Sol high step leaves. About 75-90 minutes per prompt; 128 prompts in 24 h need >= 8 prompts in flight.
- MID_PLAN: the first 8 steps. About 65-75 minutes per prompt, >= 7 prompts in flight.
- FAST_PLAN: the first 6 steps. About 50-60 minutes per prompt, >= 6 prompts in flight.
Almost all wall time is serial model time (one call in flight after the drafts), so more prompts in flight
(main.py --parallel-prompts) is what makes 128 prompts fit; 12 leaves a safe margin for every plan.
"""

from __future__ import annotations

from .pipeline import Draft, StagePlan, Step

SOL_MEDIUM = ("gpt-6.1-sol", "medium")
SOL_HIGH = ("gpt-6.1-sol", "high")
SOL_XHIGH = ("gpt-6.1-sol", "xhigh")
ASTRA_HIGH = ("gpt-6-astra", "high")

_DEEP = (
    (SOL_HIGH, ""),
    (ASTRA_HIGH, ""),
    (SOL_XHIGH, "shape"),
    (SOL_HIGH, "details"),
    (SOL_XHIGH, "materials"),
    (SOL_HIGH, ""),
    (SOL_HIGH, "shape"),
    (SOL_XHIGH, "details"),
    (SOL_HIGH, "materials"),
    (SOL_HIGH, ""),
    (SOL_HIGH, "shape"),
    (SOL_HIGH, "details"),
    (SOL_HIGH, "materials"),
    (SOL_HIGH, ""),
    (SOL_MEDIUM, "details"),
    (SOL_MEDIUM, ""),
)


def _plan(steps: tuple[tuple[tuple[str, str], str], ...]) -> StagePlan:
    return StagePlan(
        drafts=tuple(Draft(start=start, model="gpt-6.1-sol", effort="high") for start in ("plain", "proportions") * 3),
        steps=tuple(Step(rewrites=(rewrite,), keep=1, focus=(focus,) if focus else ()) for rewrite, focus in steps),
        draft_keep=1,
        pose=True,
        fill=0.99,
        glow=0.0,
        measured=False,
        outline_fit=False,
        outline_fit_last=True,
    )


DEEP_PLAN = _plan(_DEEP)
MID_PLAN = _plan(_DEEP[:8])
FAST_PLAN = _plan(_DEEP[:6])
ACTIVE_PLAN = MID_PLAN
