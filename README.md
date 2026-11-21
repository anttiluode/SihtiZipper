# SihtiZipper — reversible, local vector switchyard

A working **offline browser experiment** for a mechanical idea from a broken zipper: instead of averaging two vectors into one residual stream, a slider stores a reversible **relation between two original teeth**. When a pair ceases to match, it can slip its alignment by one index, jam and open the seam, or re-lock after compatible evidence returns. The raw vectors never change. Three tracks (CORE reference, MID and FAST auxiliary signals) have independent sliders.

**[Open the demo](index.html)** or visit the repository's GitHub Pages root if Pages is enabled. Double-click `index.html` on a desktop; there is no server, build step, GPU, model download, `npm install`, or network request.

This is **not** an assertion that transformers always average away every distinguishable feature, nor that existing alignment methods are unable to stop, skip or detect mismatches. Known ancestors include Huet's data-structure zipper, dynamic time warping, monotonic alignment, state-space tracking, and biological zippers. The experiment asks a narrower, empirical question: **does explicit reversible seam continuity help online multi-track binding under slips, sensor damage, and noise?**

## What actually happens

At time `t`, the input contains three four-dimensional *key* vectors and separate three-dimensional private payloads. The world precommits to latent tooth identities and presents auxiliary streams with a changing source-index offset of 0, 1 or 2. The zipper sees only the keys; `world.truth` (source indices and fault intervals) is used **after** every step to score the match. No supervised labels, true delays or future observations reach `engine.js`.

For track `s` and current CORE key `c_t`, the reader compares only three causal candidates `s_t`, `s_(t-1)`, `s_(t-2)`. It stores a small state `(offset, lock/open, tension)`. If the previous offset is acceptable, it stays locked unless another offset is better by the specified margin. If no candidate is acceptable, the seam jams and opens rather than forcing a wrong binding. Once a valid candidate returns, it can re-lock. Each seam entry stores **indices, not an averaged or overwritten vector**. The **Unzip last 16 / Rezip last 16** controls operate on a separate seam-link mask. They can release past link relationships and restore them exactly; the original tape and past online score do not change. This is an access bubble, not time travel or retrospective resampling. Original keys and private payloads remain available exactly via `original(t,track)` and `export().tape`.

* `LOCK`: retain current lag when it is compatible.
* `SLIP`: select another lag after a measured mismatch or clearly better compatible alternative.
* `JAM` / `OPEN` / `DETACH`: refuse to bind an incompatible tooth, preserve the earlier archived seam, and retain the raw tape.
* `RELOCK`: resume once a compatible tooth is available.

The displayed **tension** is normalized nearest-candidate distance; it is *not* Predictive Sihti's B (the variance of disagreeing recalled future deltas). Joining these mechanisms is a future experiment.

## Try it

Use **Play** or **Step one tooth**, watch the CORE/MID/FAST tracks and historical seam states. MID and FAST independently change phase and suffer fault intervals. The **Break next 12 MID teeth** and **Slip MID +1 tooth** controls alter future observations of the **environment**, not the stored engine state, so the intervention is causal. The original vector tape remains reconstructable. **Run 10-seed comparison** measures all three readers on the same independent generated trials at the selected noise setting. **Unzip / Rezip** demonstrates reversible access to archived links without editing raw tracks. **Export receipt** downloads per-step seam decisions and scores without raw vector payloads.

## What is compared

Three methods receive identical causal observations and are scored against hidden source-tick identities:

1. **Fixed seam:** always pairs MID `t-1`, FAST `t`, even after phase slips or noise. It has no abstention option.
2. **Greedy local:** compares lags `0,1,2` and joins the nearest candidate below the same distance threshold as the zipper. There is no seam history.
3. **Zipper:** the same local candidates, the same threshold, and a saved offset with a small switching margin. It may open or re-lock without destroying its history.

Scoring: a correctly sourced pair counts `correct`; joining unrelated source ticks counts `wrong`; no link is `open`. We report each *separately*. An optional utility = `(correct - 2×wrong) / all possible pairs` makes wrong joins more costly than abstention, but its penalty is a declared preference, not an objective physical fact.

Run deterministic tests (Node 18+, no dependencies):

```bat
node tests/test.cjs
```

This also rewrites `results/benchmark.json`. It uses seeds 1..40 for each of four noise levels, with 390 observations per seed. World schedule and thresholds are fixed in code. All seed sets are synthetic and use 4D keyed vectors rather than actual video/EEG. The 40 seeds vary noisy vectors, **not** the timing schedule; testing randomized schedules and learned thresholds is an important next step.

| Noise | Zipper correct | Greedy correct | Fixed correct | Zipper false join | Greedy false join |
|---|---:|---:|---:|---:|---:|
| 0.08 | 91.81% | 91.74% | 62.47% | 0.22% | 0.29% |
| 0.13 | 91.66% | 91.31% | 62.47% | 0.36% | 0.71% |
| 0.18 | 90.10% | 89.36% | 62.47% | 1.13% | 1.87% |
| 0.24 | 73.11% | 72.42% | 62.47% | 3.20% | 3.89% |

These are **identity-binding rates**, not future prediction errors and not improvements in multimodal perception. The dramatic win against fixed binding is mostly local slip correction plus an option to refuse damaged data. Against the meaningful greedy baseline, the additional seam-history effect is **small but measurable** under this particular generator. The data does not yet justify claiming a superior general architecture. See the JSON for exact denominators and counts.

## Relation to the Sihti family

[Sihti](https://github.com/anttiluode/Sihti) keeps discarded signed image differences: the original can be reconstructed from its core and residue levels. [PredictiveSihtiV2](https://github.com/anttiluode/PredictiveSihtiV2) builds local next-frame transition memories from those bands and uses disagreement B to recruit history. [LazyRetrospectiveCreditAssignment](https://github.com/anttiluode/LazyRetrospectiveCreditAssignment) retains overlooked evidence for later recall.

Here, the **bindings themselves** are reversible and retained. Current CORE/MID/FAST are deliberately *synthetic keyed tracks*, not the actual spatial blur bands. The proposed next experiment is to feed real Sihti residues as tracks and ask whether a mismatch triggers jam, delayed-history retrieval, or a new address—and whether a saved seam improves **held-out prediction**, not merely match identity on predeclared generated data.

## Implementation limits and falsifiers

- The engine is a small finite-lag local search, not an O(1) arbitrary-distance lookup and not hierarchical representation learning. Storage of the full lossless tape takes O(T); each new step checks O(3 × track count) candidates, excluding memory bookkeeping.
- A wrong but very similar tooth can still be zipped. When tooth features are truly indistinguishable, no local zipper can infer the hidden correct identity.
- The current test shares one underlying key representation across tracks, making cross-stream comparison easy. Real Sihti bands are differently distributed and require learned or calibrated compatibility metrics.
- A history seam only reduces false joins slightly relative to a simple greedy nearest-neighbor reader. A strong independent baseline may be equally good or better.
- Experiments run on synthetic, precommitted tracks; neither biological gene regulation nor real-world sensor fusion is demonstrated.

## Files

`engine.js` has the reversible state machine and two baselines. `world.js` provides independently seeded multitrack worlds and post hoc scoring. `app.js` and `index.html` implement the offline browser visualization. `tests/test.cjs` is the preregistered-style regression and 40-seed benchmark, with receipts under `results/`. `tests/ui_smoke.cjs` checks the browser control wiring in a stub DOM. Everything runs without third-party packages.
