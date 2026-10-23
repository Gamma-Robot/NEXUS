# Terrain Motion Adaptation

Standalone terrain-aware adaptation of robot joint trajectories using contact
keyframes, IK and displacement smoothing. The supplied configuration targets
Unitree G1. PyTorch, mjlab and RSL-RL are not required.

## Installation

Run from this directory after cloning the repository:

```bash
uv sync --locked
```

For browser-based motion replay:

```bash
uv sync --locked --extra visualization
```

Full-pipeline users can use the repository's root environment instead. See the
[installation guide](../docs/guides/installation.md).

## Motion format

Each clip is `<relative-clip-path>/motion.npz` with exactly two arrays: float32
`qpos` of shape `(T, 7 + J)` and a finite positive scalar `fps`. Root columns
are XYZ followed by quaternion **wxyz**, then joint positions. A group-level
`manifest.json` records `version: 1` and ordered `joint_names`.

See the [motion format reference](../docs/reference/motion-format.md) for
coordinates, validation, manifests and terrain pairing.

## Batch adaptation

The following commands run from this package directory. Prepare input robot
motions under the repository root's `datasets/g1/flat/`:

```bash
uv run python -m terrain_motion_adaptation.scripts.adapt_motions \
  --model src/terrain_motion_adaptation/assets/robots/unitree_g1/g1.xml \
  --scene src/terrain_motion_adaptation/assets/scenes/*_lv0.xml \
  --input-dir ../datasets/g1/flat \
  --base-dir ../datasets/g1/flat \
  --output-dir ../datasets/g1/pyramid \
  --num-workers 32
```

The example uses 32 workers. Reduce this to fit your job or container's actual
CPU quota and memory allocation, rather than the host CPU count.

Four fixed terrain XMLs are bundled. Omit `--scene` for a plane, or supply
terrain-only MJCFs containing a body named `terrain`. Output groups use scene
filenames and retain input joint order, FPS and relative clip paths.

The [standalone guide](../docs/guides/motion-adaptation.md) covers resuming,
output protection and custom scenes. For training data on the shared terrain
grid, use the main package's [data preparation guide](../docs/guides/data-preparation.md).

## Motion replay

Use the same terrain scene as the output group:

```bash
uv run --extra visualization python -m terrain_motion_adaptation.scripts.visualize_motions \
  --input-dir ../datasets/g1/pyramid/pyramid_stairs_lv0 \
  --base-dir ../datasets/g1/pyramid/pyramid_stairs_lv0 \
  --scene src/terrain_motion_adaptation/assets/scenes/pyramid_stairs_lv0.xml
```

Open the Viser URL printed in the terminal. Playback has a Play/Pause button;
scrubbing pauses playback. Pass the parent directory of terrain groups to switch
matching clips and scenes from the Terrain selector. Bundled scenes are recognized
automatically; supply custom XMLs with `--scene` (filenames identify groups).
**Debug → Contact markers** shows runtime G1 contact estimates, green for contact
and red for no detected contact; `--viz-ee-contact` enables them at startup.
These estimates use FK velocities and terrain-relative height, not stored labels
or measured forces. For browser-free validation, omit `--extra visualization`
and add `--headless`.

## Python API

The pipeline takes a compiled model, explicit `RobotConfig`, qpos names and
clip timestep. It returns float32 qpos in the input joint-column order. See
the [complete API example](../docs/reference/configuration.md#adaptation-python-api).

## Model requirements

Terrain geometry uses group 0; robot geometry uses other groups. Configured
end-effectors need capsule collision geometry directly on their own bodies.
See [model requirements](../docs/reference/configuration.md#adaptation-model-requirements)
before configuring another robot or scene.
