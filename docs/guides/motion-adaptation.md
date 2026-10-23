# Standalone motion adaptation

Use `terrain_motion_adaptation` to adapt robot motions without installing the
RL training environment. The package includes G1 and four terrain-only scene
XMLs. Its CLI selects the G1 robot configuration.

Complete the [standalone installation](installation.md#install-motion-adaptation-only).
Run every command in this guide from `terrain_motion_adaptation/`; the example
data directories are relative to the repository root one level above.

## Inputs

Prepare `../datasets/g1/flat/manifest.json` and clips such as
`../datasets/g1/flat/walk/motion.npz` using the
[motion format](../reference/motion-format.md). Retargeting to the robot is an
upstream step. Contact labels, body states and velocities are computed rather
than supplied in the exchange file.

## Adapt to the bundled terrain scenes

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

Each scene filename supplies an output group name. For example, the stairs
result is `../datasets/g1/pyramid/pyramid_stairs_lv0/walk/motion.npz`.
Outputs retain the input joint-column order, FPS and relative clip path.

To use a plane, omit `--scene`; the output group is named `plane`. To use a
custom terrain, supply a terrain-only MJCF containing a body named `terrain`.
See [model requirements](../reference/configuration.md#adaptation-model-requirements).

The bundled terrain XMLs describe individual patches centered at XY zero and
retain their absolute heights. For training data generation, the
[main-package entry](data-preparation.md#generate-terrain-variants) constructs
the complete shared training grid directly.

## Resume and handle partial runs

Compatible existing outputs are retained. Add `--overwrite` to recompute them.
Changing a joint schema requires a separate output directory; `--overwrite`
does not bypass schema protection. Existing files without their original
manifest cannot safely have their joint order inferred.

Input and output trees must be disjoint, including symlink destinations.
Output manifests describe the schema, not completion: check the batch summary
and any per-motion errors. Successful outputs remain usable after a partial
failure. This entry does not copy the flat input into an output `flat` group.

## Replay on the same scene

Use a scene matching the selected output group:

```bash
uv run --extra visualization python -m terrain_motion_adaptation.scripts.visualize_motions \
  --input-dir ../datasets/g1/pyramid/pyramid_stairs_lv0 \
  --base-dir ../datasets/g1/pyramid/pyramid_stairs_lv0 \
  --scene src/terrain_motion_adaptation/assets/scenes/pyramid_stairs_lv0.xml
```

Open the printed Viser URL. Choose a **Motion**, then use **Play / Pause**,
**Speed** and **Frame** to inspect it. Camera and geometry controls are under
**Scene**.

To compare terrain variants, pass the parent directory instead. The **Terrain**
selector keeps the same motion while switching scenes:

```bash
uv run --extra visualization python -m terrain_motion_adaptation.scripts.visualize_motions \
  --input-dir ../datasets/g1/pyramid \
  --base-dir ../datasets/g1/pyramid
```

Enable **Debug → Contact markers**, or add `--viz-ee-contact`, to see estimated
G1 contacts: green for contact, red for no detected contact.

<details>
<summary>Custom scenes and contact debug</summary>

Bundled terrain names and `plane` / `flat` are recognized automatically. For
custom terrains, pass the same scene XMLs used for generation with `--scene`;
the filenames must have distinct group names. Missing or invalid motion
counterparts leave the current selection intact.

The markers estimate contact from poses and terrain; they are not measured
forces. Calculating them for a long clip can take time. Compatible custom robots
can replay with debug disabled.

</details>

This viewer displays individual terrain patches. To inspect the full training
grid, use the [grouped viewer](playback.md#reference-motion-replay).

For a browser-free check, omit `--extra visualization` and add `--headless`.
This checks loading and pose application; inspect the replay to assess motion
quality.

## How adaptation works

The pipeline preserves the input timeline and adapts the trajectory in four stages:

1. **Detect contact.** Use end-effector height and speed to identify source contact frames.
2. **Adjust pelvis height.** Use terrain heights to place the robot before solving limb targets.
3. **Solve contact keyframes.** Apply limb IK and penetration correction at the selected frames.
4. **Reconstruct the trajectory.** Blend sparse displacement corrections across the clip, then enforce scalar-joint position limits.

The output keeps the input FPS, frame count and joint-column order. Body states
and contact information are computed internally; only the adapted pose sequence
is written to the motion file.

## Use the Python API

The API accepts a compiled robot-and-terrain model, an explicit `RobotConfig`,
qpos names and the clip timestep. See the
[complete example](../reference/configuration.md#adaptation-python-api).
