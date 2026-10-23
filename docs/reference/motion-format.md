# Motion format

A clip stores the pose trajectory needed to reconstruct the robot: root
position, root orientation and joint positions. A separate manifest names the
joint columns, and the caller supplies the robot model for FK.

## Manifest

```json
{"version": 1, "joint_names": ["joint_a", "joint_b"]}
```

The short example shows the schema. For G1, list all 29 scalar joint names
in the same order as the stored qpos joint columns. Names must be unique,
nonempty strings. Missing or extra model joints are rejected; different column
orders are supported through name mapping.

Loaders search for the closest manifest along the clip's logical ancestors,
stopping at `base_dir`. Place a manifest in each group root for the simplest
layout. A nested manifest can describe a subtree with a different column order.

## NPZ fields

Each file contains **exactly** these two arrays:

| Field | Storage | Meaning |
| --- | --- | --- |
| `qpos` | float32, shape `(T, 7 + J)` | At least two finite poses |
| `fps` | Finite positive numeric scalar, shape `()` | Sampling rate in frames per second |

For the supplied G1 model, `J = 29` and each qpos row has 36 entries.

| Columns | Meaning |
| --- | --- |
| `0:3` | Root position XYZ |
| `3:7` | Root quaternion **wxyz** |
| `7:` | Scalar joint positions in `joint_names` order |

Coordinates are right-handed with Z up. Lengths use meters and angles use
radians. The quaternion maps root-local coordinates into world coordinates.
Its norm must be within `1e-3` of one. Readers reject invalid values without
normalizing or repairing them. FPS is a scalar, not a one-element array.

Body poses, velocities and contact labels are derived at runtime and are not
additional NPZ fields. The manifest does not embed an XML path or bind a clip
to a particular model file.

## Write and read a clip

Given root and joint arrays already expressed for the target robot:

```python
import json
from pathlib import Path
import numpy as np

group = Path("datasets/g1/flat")
clip = group / "walk" / "motion.npz"
clip.parent.mkdir(parents=True, exist_ok=True)
(group / "manifest.json").write_text(
    json.dumps({"version": 1, "joint_names": joint_names}), encoding="utf-8"
)
qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos], axis=-1).astype(np.float32)
np.savez(clip, qpos=qpos, fps=np.asarray(50.0))
```

Supply `joint_names`, `root_pos`, `root_quat_wxyz` and `joint_pos` from your
retargeted motion. Use `np.savez` for uncompressed storage.
Validate through the shared standalone reader:

```python
from terrain_motion_adaptation.utils.motion import load_motion

motion = load_motion("datasets/g1/flat/walk/motion.npz", base_dir="datasets/g1/flat")
root_pos = motion.qpos[:, :3]
root_quat_wxyz = motion.qpos[:, 3:7]
joint_pos = motion.qpos[:, 7:]
dt = motion.timestep  # 1 / motion.fps
```

## Terrain pairing and coordinates

```text
datasets/g1/pyramid/
├── flat/walk/motion.npz
├── hf_pyramid_slope_lv0/walk/motion.npz
└── pyramid_stairs_lv0/walk/motion.npz
```

Here `walk/motion.npz` identifies one motion across groups. Each non-flat clip
must have a flat counterpart with the same FPS and frame count for packing.
Different terrain groups may have different subsets. The flat group can contain
extra motions; it must be nonempty.

Adapted files store patch-local XY and absolute terrain Z. Main-package
adaptation removes its temporary patch XY offset, and reference replay restores
it. Do not subtract the terrain origin's Z when assembling a dataset.

## FPS and the training pack

Individual exchange files may have different or fractional FPS. A training
pack requires **one common FPS across all groups**. The packer checks FPS and
paired frame counts; it does not resample or prove semantic time alignment.
Inputs must already be aligned in time.

Packed data are a separate runtime layout under `dataset/<robot>/`. Each group
contains `meta_motion.json` and `_tensordict/`, with model-ordered root positions,
wxyz quaternions and joint positions. FK and derived velocities are reconstructed
by the training loader. Repack when the model's ordered joint schema changes.

## Logical paths and links

File and directory symlinks retain their logical dataset-relative identities.
Two aliases remain separate clips; directory ancestor cycles are skipped.
Use consistent logical paths for the input and `base_dir`. Output protection
also checks physical destinations so a link cannot write back into source data.
