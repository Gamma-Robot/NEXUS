# Configuration and API

Use the task ID to select a learning method and its observation configuration.
Use the Python API when integrating motion adaptation into another data pipeline.

## Registered tasks

The base ID is `Perceptive-PyramidTerrain-Unitree-G1`.

| Suffix added to the base ID | Learning method | Actor input | Symmetry |
| --- | --- | --- | --- |
| `-Teacher` | Privileged teacher PPO | Privileged observations + elevation | Disabled |
| `-Teacher-Symmetry` | Privileged teacher PPO | Privileged observations + elevation | Mirror loss |
| `-BC` | Student distillation | Flat motion command, proprioception + depth | Disabled |
| `-BC-Symmetry` | Student distillation | Flat motion command, proprioception + depth | Augmented teacher supervision |
| None | One-stage perceptive PPO | Flat motion command, proprioception + depth | Disabled |
| `-Symmetry` | One-stage perceptive PPO | Flat motion command, proprioception + depth | Mirror loss |

The teacher and PPO critics additionally receive privileged critic observations.
Actor, critic, teacher and student observation groups are configured separately.
See [Symmetry](symmetry.md) for the exact spatial transforms and losses.

## Configuration entry points

Paths below are relative to `src/perceptive_bfm/`:

| Area | File |
| --- | --- |
| Shared terrain generator | `terrains/config.py` |
| Task registration | `tasks/perceptive/config/g1/__init__.py` |
| G1 environment and play overrides | `tasks/perceptive/config/g1/env_cfgs.py` |
| Observations, rewards, events and simulation | `tasks/perceptive/perceptive_env_cfg.py` |
| Teacher and one-stage PPO | `tasks/perceptive/config/g1/rl_cfg.py` |
| Student distillation | `tasks/perceptive/config/g1/distillation_cfg.py` |
| Depth camera profiles and processing | `tasks/perceptive/sensors/` |

See [terrain configuration](terrains.md) for rendered terrain views, parameter
values and how to keep motion generation and training geometry consistent.

The default depth observation uses the processed `(1, 18, 32)` image, rather than the
camera's raw resolution. The elevation observation is a 187-value height scan
on a `(Ny, Nx) = (11, 17)` grid. These dimensions matter for CNN inputs and
the symmetry transforms.

`DEPTH_BACKEND` in `perceptive_env_cfg.py` selects the depth implementation.
The optional `PINHOLE_RAYCAST` baseline converts upstream ray hits to optical
depth, maps misses to the maximum depth, normalizes over 0–2 m, and crops and
resizes to `(1, 18, 32)`. It retains the upstream body-frame camera geometry;
it does not reproduce the default head-camera calibration or noise pipeline.
Camera intrinsic and extrinsic randomization events are registered only for
`GROUPED_RAYCASTER`, whose sensor exposes the required setters. The upstream
`RENDER` baseline retains its native `(1, 27, 48)` observation. These alternative
backends have not been validated as replacements for pretrained policy inputs.

Runner defaults are 30,000 iterations / 24 rollout steps per environment for
PPO and 10,000 iterations / 16 steps for distillation. All save every 500
iterations by default. CLI overrides are recorded in the saved `params/` configurations.

## Adaptation Python API

Run this example from the repository root in the full environment. It reads an
existing `walk` input; model paths refer to bundled resources:

```python
from pathlib import Path
import json

from terrain_motion_adaptation.optimization.pipeline import (
    MotionAdaptationConfig,
    terrain_adapt_qpos,
)
from terrain_motion_adaptation.robots.g1 import G1_CONFIG
from terrain_motion_adaptation.utils.model import combine_model_and_scene
from terrain_motion_adaptation.utils.motion import load_motion, write_motion

assets = Path("terrain_motion_adaptation/src/terrain_motion_adaptation/assets")
model = combine_model_and_scene(
    assets / "robots/unitree_g1/g1.xml",
    assets / "scenes/pyramid_stairs_lv0.xml",
)
motion = load_motion("datasets/g1/flat/walk/motion.npz", base_dir="datasets/g1/flat")
adapted = terrain_adapt_qpos(
    model=model,
    robot=G1_CONFIG,
    qpos=motion.qpos,
    qpos_names=motion.qpos_names,
    timestep=motion.timestep,
    config=MotionAdaptationConfig(),
)
output = Path("datasets/g1/adapted_walk")
output.mkdir(parents=True, exist_ok=False)
(output / "manifest.json").write_text(
    json.dumps({"version": 1, "joint_names": list(motion.joint_names)}),
    encoding="utf-8",
)
write_motion(output / "walk/motion.npz", adapted, fps=motion.fps)
```

`RobotConfig` supplies the root body, contact end-effectors and kinematic chains.
`MotionAdaptationConfig` supplies contact thresholds and keyframe IK settings.
Its `contact` field accepts one `EEContactCfg` or a dictionary keyed exactly by
the end-effector names. Its `keyframe_ik` field is a `KeyframeIKCfg`.

The array result is float32 and retains the input column order and frame count.
When no source contact keyframe is active, `NoActiveKeyframesError` is raised.
Final limited scalar joints are clipped inside the compiled model's position
limits, including float32 rounding. Position limits alone do not establish
contact quality or smooth velocities.

For a patch placed elsewhere in a larger scene, supply `dx` and `dy`. Output
removes that XY placement and keeps absolute Z; no Z-offset argument is needed.
The bundled standalone scenes are already centered and use the zero defaults.

## Adaptation model requirements

The current implementation accepts one free-root robot with scalar hinge/slide
joints. G1's configured root is `pelvis`, and its contact end-effectors are the
two ankle-roll links and two wrist-yaw links. Other robots need an explicit
`RobotConfig` and must satisfy the solver's kinematic assumptions; they have
not been validated by the supplied configuration.

Terrain raycasts use geometry group 0. Robot geometry, including visual-only
geometry, must use other groups. The model must contain terrain geometry in
group 0. Each configured contact end-effector requires capsule collision
geometry directly on its own body, not only on a child body.

The standalone scene-composition helper expects a terrain body named `terrain`
and rejects scene XMLs with joints, actuators or sensors. The Python pipeline
checks the solver requirements before FK; batch adaptation checks scenes before
creating output files.
