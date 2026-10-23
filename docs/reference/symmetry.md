# Left-right symmetry

A mirrored sample must remain physically consistent: depth, terrain heights,
motion commands, robot state and actions all describe the same reflection.
The `-Symmetry` tasks apply these transforms together during learning.

## Depth: reflect the camera image

```{figure} ../_static/media/symmetry-depth.svg
:alt: MuJoCo-rendered stair depth, its left-right reflection, and the corresponding original and mirrored 32 by 18 policy inputs.
:figclass: doc-figure

Simulated head-camera depth and its left-right reflection. The upper views show the scene at higher resolution; the lower views show the 32 × 18 policy input. Bright pixels are nearer. Both columns use the same scale.
```

The default head-camera tensor has shape `(B, 1, 18, 32)`. Reflection reverses its width
axis while preserving row order and depth values:

```python
mirrored_depth = depth.flip(-1)
```

Equivalently, $D^{\mathrm{mirror}}_{v,u}=D_{v,W-1-u}$. The transform acts after
normalization and crop/resize. The high-resolution view above is for reading
the scene; it does not change the policy's input resolution.

The optional upstream pinhole and rendered baselines retain body-aligned cameras
looking along local `-Z`. For these configurations, lateral body `y` is the image
height axis, so their sagittal transform and debug preview use `depth.flip(-2)`.
This differs from the default head-camera calibration. Arbitrary camera mounting
rotations require a corresponding review of the image symmetry transform.

## Elevation: reflect lateral position

```{figure} ../_static/media/symmetry-elevation.svg
:alt: Original and mirrored terrain elevation from downward MuJoCo raycasts, with a shared color scale and the actual 17 by 11 height-scan samples.
:figclass: doc-figure

Terrain heights from the same pose. Color encodes sensor-relative z, and the triangle marks the pelvis and forward direction. The upper maps sample the geometry densely for display; the lower maps show the 187 samples used by the observation.
```

The height scan uses a yaw-aligned frame: **x** points forward, **y** is lateral,
and **z** points up. Its 1.6 m × 1.0 m footprint at 0.1 m spacing gives
`Nx = 17` and `Ny = 11`. Values are flattened from a `(Ny, Nx)` grid.

Restore that layout, reverse lateral y, then flatten again:

```python
grid = elevation.reshape(-1, 11, 17)  # batch, lateral y, forward x
mirrored_elevation = grid.flip(-2).flatten(start_dim=1)
```

This is $H^{\mathrm{mirror}}_{i_y,i_x}=H_{N_y-1-i_y,i_x}$: forward position and
height remain unchanged. Current height observations are clipped to
`[-1.5, 0.0]` m relative to the sensor.

The plots display `grid.T` so forward x points upward. Their horizontal mirror
is therefore a reversal of the **stored lateral axis**, not the stored last
axis. Flipping the flattened vector end-to-end would also reverse forward x
and is a different operation.

## Reflect the full observation

Image mirroring alone would leave the rest of a training sample inconsistent.
The same reflection also acts on commands, state and joint actions:

| Quantity | Reflection |
| --- | --- |
| Position and linear velocity | `(x, y, z) → (x, −y, z)` |
| Angular velocity | `(wx, wy, wz) → (−wx, wy, −wz)` |
| Joints and joint actions | Exchange left/right joints and apply their joint-axis signs |
| Bodies and contact channels | Exchange left/right counterparts |

The group transforms check observation widths and processed camera shape against
the configured layout. No mirrored motion files need to be stored on disk.

## PPO and distillation

**PPO uses mirror consistency.** The additional loss encourages the action on a
mirrored observation to match the mirrored action on the original observation.
The supplied PPO configurations enable that loss and leave rollout data
augmentation disabled.

**Distillation uses teacher supervision on augmented observations.** The frozen
teacher is queried again on the original-plus-mirrored observations; its targets
are not produced by simply reflecting an earlier action. This does not assume
the teacher is already perfectly symmetric. `symmetry_bc_coef` weights the
additional supervised loss and defaults to 1.0.

Remove `-Symmetry` from a task ID to select its configuration without these
terms. The transforms live in `tasks/perceptive/mdp/symmetry_terms.py` and
`utils/symmetry.py` under `src/perceptive_bfm/`.
