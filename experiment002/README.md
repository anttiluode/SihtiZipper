# Experiment 002 — Occluded-video diffusion vs explicit zipper-style memory

This is an **honest two-task comparison**, not an attempt to establish that a zipper beats modern diffusion models.

**Run / train the model:**

```bat
python -m pip install numpy torch pillow
python experiment002\compare.py --steps 480 --train-scenes 320 --test-scenes 24
python experiment002\make_viewer.py
```

Open [`viewer.html`](viewer.html) directly in Edge; it embeds five recorded evaluation frames and needs no server. The trained checkpoint is [`results/tiny_ddpm.pt`](results/tiny_ddpm.pt). On the original test run, a CPU-only PyTorch 2.10 environment completed the 480-step training in ~21 seconds. CUDA is selected automatically if PyTorch detects a GPU.

## The shared world and measurements

- Synthetic **24 × 24, two-appearance-channel video** with 18 frames per independent scene. Two moving Gaussian discs cross behind a **6-column opaque screen**. About one-third of scenes reverse direction under the occluder. An independent camera noise process sometimes inserts bright *ghost* patches that resemble a real object's color.
- A test scene's latent positions and unmasked image are **never passed to either model**. They are used only for held-out scoring. Training seeds (1000+) and test seeds (50000+) are disjoint. The algorithms see the same observed frames; the diffusion net is conditioned on the *current and previous two observations*, while the tracker retains an explicit online state accumulated from the entire previous stream.
- **Genuine learned diffusion:** an 18-step conditional DDPM in **pixel space**, with a small convolutional network trained by Adam to predict added Gaussian noise at randomly selected diffusion depths. The target is the unmasked clean frame, which is available only for *training scenes*; no masked test target or identity label is supplied to inference. The network has 80,786 learned parameters; it is not a pretrained Stable Diffusion model. A tiny model is intentionally a weak baseline and **not competitive with a full video diffusion model**. A separately held-out 24-video validation split (seeds 40000–40023) chooses one scalar background offset, 0.45, before testing. Raw and calibrated scores are both reported.
- **Zipper port:** a small new Python **zipper-inspired** tracker, not a drop-in invocation of the original JavaScript `Zipper` class. It stores independent object tracks, predicts an occluded track forward, and labels an observation `LOCK`, `PREDICT_ONLY`, `JAM_ABSTAIN` or `RELOCK`. It does **not** treat extrapolation as proof that the hidden object was seen; the observation stream is never overwritten. It is essentially a conventional gated constant-velocity tracker with zipper vocabulary. That limitation is part of the result.
- **Control 1:** the identical constant-velocity tracker without a rejection threshold. This isolates what the gate/abstention does. **Control 2:** hold-last.
- **Two separate tasks:** (a) how well each branch predicts *unseen pixels* in the occluded columns, and (b) how many incoming observations the trackers falsely bind to the real object's history. Diffusion pixels do not carry links; there is no meaningful “diffusion wrong join” without adding a tracker. Actor appearance is preserved as two distinct color channels. Therefore the experiment does not establish tracking of visually indistinguishable objects or fully recover object identity from ambiguous appearance.

## Reproduced result, 24 unseen scenes

| Method | Hidden-strip pixel MSE ↓ | Hidden-object pixel MSE ↓ | Hidden center error, px ↓ | Wrong joins / accepted joins |
| --- | ---: | ---: | ---: | ---: |
| Hold-last | 0.012496 | 0.298107 | — | Not a tracker |
| Ungated constant velocity | 0.007788 | 0.166677 | 5.536 | 40 / 625 |
| Zipper-style gated tracker | **0.006566** | 0.115809 | 1.789 | **2 / 567** |
| **Tiny DDPM, raw** | 0.136207 | **0.055025** | 2.859 | Not a tracker |
| **DDPM with separately validated background correction** | 0.011443 | 0.247984 | 0.758 on just 2 confident cases | Not a tracker |

The **raw diffuser** scores better on foreground-only reconstruction but paints a broad gray stripe in the empty background. A separately held-out 24-video calibration selects a uniform background offset **0.45**, reducing empty-strip hallucinations while also wiping out most true hidden foreground. The gated tracker beats its ungated counterpart in rejecting false observed links but abstains more often (**201 vs 143** decisions). Neither version of the tiny DDPM is a strong modern video model. These are one training run and one deterministic evaluation seed block. No claim of general superiority or statistical significance is made.

The result is **not a fair matched-capacity generative benchmark**: the tracker has an explicit longer lived state and analytical motion prior; the diffusion model has a two-frame history and limited training. This is a diagnostic separation of image reconstruction versus binding reliability. A proper next test must give an equally strong diffusion/flow tracker a persistent state and evaluate on real, identity-ambiguous video, with randomized schedules and calibrated abstention curves.

## Files

- `compare.py`: world generator, pixel DDPM training and sampling, tracker, scoring, receipt.
- `results/receipt.json`: full parameters, per-method aggregates, five recorded frames and per-seed notes (the offline viewer additionally reconstructs two complete 16-frame clips from the saved checkpoint).
- `results/tiny_ddpm.pt`: trained model checkpoint for this exact run (not needed to view recorded results).
- `viewer.html`: offline visual inspector; regenerated by `make_viewer.py`.
- `results/contact_sheet.png`: selected true / observed / diffusion / zipper frames.

**Safety / privacy:** no webcams or external images are recorded. Everything is generated locally. There are no data uploads, API calls or cloud dependencies.
