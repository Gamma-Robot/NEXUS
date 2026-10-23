# Playback and export

Reference replay displays stored robot poses. Policy playback runs a controller
in simulation. They accept different data directories:

| Operation | Input | Checkpoint |
| --- | --- | --- |
| Reference replay | Raw grouped NPZ: `datasets/g1/pyramid` | Not needed |
| Policy playback | Packed dataset: `dataset/g1/pyramid` | Required for a trained policy |

Run the commands below from the repository root with the full installation.

## Reference motion replay

The [G1 motion examples](motion-examples.md) include walking, crawling with a
back handspring, and backward roll across flat ground and four terrains.

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir datasets/g1/pyramid
```

Open the printed Viser URL.

| Control | Use |
| --- | --- |
| **Terrain** | Switch terrain while keeping the same motion |
| **Motion** | Select a clip |
| **Play / Pause**, **Speed**, **Frame** | Replay slowly or inspect individual poses; scrubbing pauses playback |
| **Scene** | Adjust the camera and toggle terrain, robot visuals or collision geometry |
| **Debug → Contact markers** | Show estimated hand and foot contacts: green for contact, red for no contact |

Add `--viz-ee-contact` to enable contact markers at startup.

<details>
<summary>Contact estimates and missing motion counterparts</summary>

Contact markers are calculated from the stored poses and terrain. They are
estimates, not measured forces or contact annotations. Calculating them for a
long clip can take time. Contact debug currently uses G1 end-effectors; other
compatible robots can replay with debug disabled.

If a terrain group has no valid counterpart for the selected motion, the viewer
reports it and keeps the current selection. See
[Terrain pairing and coordinates](../reference/motion-format.md#terrain-pairing-and-coordinates)
for how stored poses are placed on the scene.

</details>

To choose an initial clip and a port:

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir datasets/g1/pyramid \
  --group pyramid_stairs_lv0 \
  --motion walk/motion.npz \
  --port 8080
```

For a browser-free pass over every available group and frame:

```bash
uv run python -m perceptive_bfm.scripts.data_process.visualize_motions \
  --input-dir datasets/g1/pyramid \
  --headless
```

Headless replay checks loading and pose application. Use visual inspection to
assess contact, sliding, penetration and continuity.

## Policy playback

Replace `<student_run>` and `<iteration>` with an existing student checkpoint:

```bash
uv run python -m perceptive_bfm.scripts.play \
  Perceptive-PyramidTerrain-Unitree-G1-BC-Symmetry \
  --motion-file dataset/g1/pyramid \
  --checkpoint-file "logs/rsl_rl/g1_perceptive/<student_run>/model_<iteration>.pt" \
  --num-envs 1 \
  --viewer viser
```

Use the task matching the checkpoint: `-Teacher[-Symmetry]` for a teacher,
`-BC[-Symmetry]` for a distilled student, or the base/`-Symmetry` task for
one-stage PPO. `--motion-file` refers to the packed group root, despite its name.

`--viewer native` opens a desktop MuJoCo viewer. `--viewer auto` selects native
when a display is available and Viser otherwise. On a remote machine, forward
the port reported by Viser and open the forwarded URL on your own computer.

| Option | Use |
| --- | --- |
| `--video True` | Record trained-policy playback |
| `--camera robot/tracking` | Select the recording camera; requires `--video True` |
| `--fixed-terrain 3 0` | Start on stairs at level 0 |

`--camera` accepts a model camera name or ID and affects the recording, not the
interactive orbit camera. Omit it to use the configured recording viewpoint.
Terrain indices are 0 flat, 1 slope, 2 inverse slope, 3 stairs and 4 inverse
stairs; the supplied configuration has level 0.

Pass the task before `--help` to inspect options:

```bash
uv run python -m perceptive_bfm.scripts.play \
  Perceptive-PyramidTerrain-Unitree-G1-BC-Symmetry \
  --help
```

## Export ONNX during playback

Add `--export-onnx True` to the trained-policy command. It writes `policy.onnx`
beside the loaded checkpoint, then continues playback. This option is not an
export-only command. Training also attempts ONNX export on checkpoint saves;
check the log for export failures rather than assuming a checkpoint implies
an ONNX file was written.

## Export a scene

Export the task's scene without training or starting a viewer:

```bash
uv run python -m perceptive_bfm.scripts.export_scene \
  Perceptive-PyramidTerrain-Unitree-G1 \
  --output-dir logs/scenes/pyramid
```

Without `--subterrain`, the export retains the full terrain grid. To export a
single selected patch, append `--subterrain 3 0`. The output includes the robot
and scene; it is not a terrain-only input for the standalone adaptation CLI.
Use a dedicated output directory that does not exist or is empty. The exporter
rejects nonempty directories and directory symlinks. With `--zip`, it also
rejects an existing destination archive; choose a new output path to retain
previous exports.
