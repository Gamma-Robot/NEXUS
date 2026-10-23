# Data preparation

Adapt your own G1 motions to terrain, inspect the results, and pack them for
training. Use the [full installation](installation.md#install-the-full-pipeline)
and run commands from the repository root.

To explore existing optimization results, start with the
[motion examples](motion-examples.md).

## Prepare flat-ground robot motions

Start with robot motions at **50 FPS** in this layout:

```text
datasets/g1/flat/
├── manifest.json
├── walk/motion.npz
└── turn/motion.npz
```

| File | Required contents |
| --- | --- |
| `motion.npz` | float32 `qpos` and scalar `fps` |
| `manifest.json` | The 29 joint names in the same order as the joint columns |

See [Motion format](../reference/motion-format.md) for array shapes and a file-writing
example. Human motion must already be retargeted to G1. These commands do not
retarget or resample it.

## Generate terrain variants

Run adaptation on a CPU machine. Set `--num-workers` to fit its CPU and memory
allocation; this example uses 32:

```bash
uv run python -m perceptive_bfm.scripts.data_process.adapt_motions \
  --input-dir datasets/g1/flat \
  --base-dir datasets/g1/flat \
  --output-dir datasets/g1/pyramid \
  --num-workers 32
```

Results are saved in four terrain groups:

| Output group | Terrain |
| --- | --- |
| `hf_pyramid_slope_lv0` | Pyramid slope |
| `hf_pyramid_slope_inv_lv0` | Inverse pyramid slope |
| `pyramid_stairs_lv0` | Pyramid stairs |
| `pyramid_stairs_inv_lv0` | Inverse pyramid stairs |

For example, `walk/motion.npz` becomes
`datasets/g1/pyramid/pyramid_stairs_lv0/walk/motion.npz` on stairs.
See [Terrains](../reference/terrains.md) for the scene geometry.

Review failed or skipped clips in the completion summary before continuing.
To resume an interrupted run, rerun the same command; compatible existing
outputs are skipped.

Add the original motions as the paired `flat` group once:

```bash
ln -s ../flat datasets/g1/pyramid/flat
```

This shares the original files and their manifest without copying them.

## Inspect the references

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir datasets/g1/pyramid
```

Switch **Terrain** to compare versions of a motion. Use slow playback and the
frame slider to check contacts, sliding, penetration and sudden pose changes.
See [Replay controls](playback.md#reference-motion-replay).

## Pack the training dataset

```bash
uv run python -m perceptive_bfm.scripts.data_process.pack_motions \
  --input-dir datasets/g1/pyramid \
  --output-dir dataset/g1/pyramid \
  --target-fps 50 \
  --num-workers 32
```

Each terrain motion needs a flat counterpart with matching FPS and frame count.
`--num-workers` controls concurrent discovery, source-path checks, motion
validation and packing reads. These stages show progress separately. Reads
use bounded prefetch; groups are written sequentially in a stable motion order.
All groups pass source/output protection, data validation and flat-pairing
checks before any packed group is written.

Use a fresh output directory when changing the terrain groups, and keep it
separate from the input directory.

| Directory | Use |
| --- | --- |
| `datasets/g1/pyramid` | NPZ motions for reference replay |
| `dataset/g1/pyramid` | Packed data for training and policy playback |

Continue with [Training](training.md).
