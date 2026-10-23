# Quickstart

Explore terrain-optimized G1 motion examples without running the optimization
pipeline. Run each command from the repository root.

## 1. Install

Install [uv](installation.md#install-uv), then create the full environment:

```bash
uv sync --locked
```

See [Installation](installation.md) for the Python, CUDA and GPU requirements.

## 2. Replay a motion

The repository includes three motion excerpts with optimized results for four
terrain variants, alongside their flat-ground references. Start with walking on
stairs:

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir resources/g1/motions/pyramid \
  --group pyramid_stairs_lv0 \
  --motion walk/motion.npz \
  --viz-ee-contact
```

Open the printed URL in your browser. Use **Terrain** and **Motion** to switch
references, **Play / Pause** to control playback, and the frame slider to
inspect individual poses. **Contact markers** shows estimated hand and foot
contacts.

See [Motion examples](motion-examples.md) for the other clips and
[Playback](playback.md#reference-motion-replay) for the viewer controls.

## Continue from here

| Your next step | Guide |
| --- | --- |
| Adapt your own motions to terrain | [Data preparation](data-preparation.md) |
| Use adaptation without the training dependencies | [Standalone adaptation](motion-adaptation.md) |
| Train on your own prepared motion dataset | [Training](training.md) |
| Download the ONNX model and inference settings | [Pretrained model](pretrained-model.md) |
