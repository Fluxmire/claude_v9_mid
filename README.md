# Image to Three.js miner (Claude v9, MID_PLAN: a bit faster, a bit weaker)

This folder runs **MID_PLAN**.

Generates a Three.js module for a prompt image through the competition witness: six drafts, the best one kept, then
one lineage rewritten against its own renders in as many gated steps as the 5M credits allow, a pose check and the
outline fit. Entry point: `main.py` -> `async generate(image, seed, witness) -> str | None`. The plan is in
`miner404/versions.py`; what changed from the base is at the top of `miner404/pipeline.py` and in `provenance.json`.

## Per prompt

| Stage | Calls | Notes |
|---|---|---|
| Drafts | 6 Sol high | plain / proportions starts; a rejected module gets one repair |
| Draft selection | 2 Sol medium | mirrored panel on the judge's four front views and its neutral-light grid; keeps 1 |
| Gated steps | 3 each (rewrite + 2 judges) | 1 Sol high, 1 Astra high, 3 Sol xhigh, Sol high after; Sol medium tail |
| Pose | 0-2 | quarter turn / stand-up offered to the selector, wins only by a clear margin |
| Outline fit | 0 | applied to the final module (also after a turn) only if its close-ups still match |

- **Gated step:** a rewrite is sent only if it still fits the budget at its worst case, together with a repair, the two
  selection calls that judge it and the pose stage. The child replaces its parent unless the parent leads by 0.25
  penalty points. A step that does not fit is skipped and costs nothing.
- **Fixes:** the selector lists up to six concrete fixes for each candidate. The kept module's list goes into its next
  rewrite, which checks each fix against the renders before applying it.
- **Plans** (same code; `ACTIVE_PLAN` in `miner404/versions.py` picks one). Figures are from an offline model of
  measured usage; times are estimates, not live measurements.

  | Plan | Gated steps run | Calls | Credits | Minutes per prompt | 128 prompts at 12 in flight |
  |---|---|---|---|---|---|
  | DEEP_PLAN (strongest) | about 11 of 16 (budget-limited) | 41-45 | 4.4-4.6M | 75-90 | about 15-16 h |
  | MID_PLAN | 8 | 32-34 | 3.8-3.9M | 65-75 | about 12-13 h |
  | FAST_PLAN | 6 | 26-28 | 3.2-3.3M | 50-60 | about 10 h |

  - The limits cannot be exceeded: with every output at its cap DEEP spends 4.80M; with images 45% over the
    estimate, 4.90M (limit 5M / 60 calls).
  - All wall time is mostly serial model time, so more prompts in flight is what fits 128 prompts in 24 h: at least
    8 (DEEP), 7 (MID), 6 (FAST). Use `--parallel-prompts 12 --parallel-calls 12` or more.
- **`service_tier` stays `"default"`.** The witness passes `"priority"` through and credits are the same, but a Codex
  subscription may reject it. Test one call before changing it.

## Tests (offline, no model calls)

From the research tree:
- `claude_research/tests/test_v9_offline.py REPO`: fake model, renderer and judge. It checks the budget and call
  limits, unique and replay-safe request ids, that every kept child was judged, the pose stage, the outline fit and
  its guard, repairs, injected failures, the fallback, the async entry point, and determinism.
- `claude_research/tests/test_v9_wrapper.py REPO [BASE]`: runs the wrappers in real three.js 0.183.2 to check that no
  part moves or turns inside out (fits, merge, bake, pose groups).

## Live run

```sh
uv venv --python 3.13 .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
git clone --branch miner-runtime-v0.1.5 --depth 1 https://github.com/404-Repo/404-gen-subnet.git ../404-sdk
```

Create `witness.json` next to `main.py`, with `witness_url` set to `https://witness.subnet404.xyz` and
`proof_parameters` set to the absolute path of `404-sdk/witness-server/verifier/params`. Then:

```sh
.venv-live/bin/python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY \
  --parallel-prompts 12 --parallel-calls 12
```
