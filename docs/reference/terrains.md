# Terrain configuration

Motion adaptation and policy training share the terrain generator in
`src/perceptive_bfm/terrains/config.py`. It builds five terrain types at a
single difficulty level: flat ground, pyramid slopes and pyramid stairs,
including their inverse variants.

## Terrain shapes

```{figure} ../_static/media/terrain-types.png
:alt: MuJoCo views of the pyramid slope, inverse slope, pyramid stairs and inverse stairs, with gradient and step dimensions.
:figclass: doc-figure

MuJoCo views of the four non-flat patches. Each patch is 20 × 20 m.
The ordinary variants rise toward the center; the inverse variants descend.
```

| Terrain | Configuration class | Dataset group | Shape parameters |
| --- | --- | --- | --- |
| Flat ground | `BoxFlatTerrainCfg` | `flat` | Level reference surface |
| Pyramid slope | `HfPyramidSlopedTerrainCfg` | `hf_pyramid_slope_lv0` | `slope_range=(0.3, 0.3)` |
| Inverse pyramid slope | `HfPyramidSlopedTerrainCfg` | `hf_pyramid_slope_inv_lv0` | Same range, `inverted=True` |
| Pyramid stairs | `BoxPyramidStairsTerrainCfg` | `pyramid_stairs_lv0` | `step_height_range=(0.1, 0.1)`, `step_width=0.3` |
| Inverse pyramid stairs | `BoxInvertedPyramidStairsTerrainCfg` | `pyramid_stairs_inv_lv0` | Same step height and width |

**Slope is a gradient, not an angle.** `0.3` specifies rise/run; a larger
magnitude produces a steeper surface. The inverse slope uses the same positive
range and reverses the direction through `inverted=True`.

**Stair height and width are in meters.** `step_height_range` sets the vertical
change per step; `step_width` sets the horizontal tread depth. The inverse
stair class creates descending steps with the same positive dimensions.

## The shared scene

```{figure} ../_static/media/terrain-grid.png
:alt: Overhead view of the complete five-column terrain scene, ordered flat, slope, inverse slope, stairs and inverse stairs.
:figclass: doc-figure

One generated scene contains all five patches. Columns identify terrain types;
rows identify difficulty levels. The supplied scene has one row.
```

| Generator parameter | Current value | Meaning |
| --- | --- | --- |
| `size` | `(20.0, 20.0)` m | Dimensions of each patch, not the whole scene |
| `num_rows` | `1` | One difficulty level, indexed as `0` |
| `curriculum` | `True` | Allocate one column per named terrain type |
| `proportion` | `0.2` for each type | Relative weight for assigning environments to terrain columns |
| `seed` | `42` | Seed for terrain generation |

In this allocation mode, the five entries in `sub_terrains` determine the
five columns; `num_cols` does not override that count. The resulting grid
covers 20 × 100 m. Many training environments share these patches, so changing
`--env.scene.num-envs` does not create a new patch for every environment.

Although the allocation mode is named `curriculum`, this configuration has
**fixed difficulty**: there is only one row and both ends of each parameter
range are equal. Increasing the number of training environments does not make
the terrain harder.

## Parameters and where to change them

The fixed difficulty is selected by `_build_pyramid_terrain()`:

```python
return _build_pyramid_terrain_grid(
    num_rows=1,
    slope_range=(0.3, 0.3),
    step_height_range=(0.1, 0.1),
)
```

This is an excerpt from `terrains/config.py`. Edit the two range values there
to change the fixed slope or stair height. Keep each pair equal when selecting
one fixed difficulty. Patch size, tread width and the terrain-type dictionary
are configured in `_build_pyramid_terrain_grid()` in the same file.

The terrain classes also supply these defaults:

| Parameter | Applies to | Default | Effect |
| --- | --- | --- | --- |
| `platform_width` | Slopes and stairs | `1.0` m | Requested central platform width; stair discretization determines the final remaining width |
| `horizontal_scale` | Slopes | `0.1` m | Heightfield grid spacing |
| `vertical_scale` | Slopes | `0.005` m | Height quantization |
| `border_width` | Individual slopes and stairs | `0.0` m | Flat border around a patch |
| `holes` | Stairs | `False` | Use complete steps rather than a cross with corner gaps |

Set these fields on the corresponding terrain configuration in
`_build_pyramid_terrain_grid()` when customizing them. A smaller heightfield
spacing resolves finer geometry but increases the heightfield size.

### Keep reference motions and training geometry together

After changing geometry, regenerate adapted motions and repack them into a
new dataset directory using [data preparation](../guides/data-preparation.md).
Existing adapted motions retain the contacts and heights of the terrain used
to generate them. Keeping the same group name does not make old motions match
a new slope or staircase, and compatible existing output files are skipped by
default.

The main adaptation entry builds the complete scene and uses each patch's XY
origin to place a motion during optimization. Saved motions have that XY
offset removed while retaining absolute Z. Training and grouped replay restore
the XY placement. This is why reference motions and their matching terrain
must use the same geometry and height convention.

### Standalone terrain scenes

The standalone package bundles four terrain-only XMLs under
`terrain_motion_adaptation/src/terrain_motion_adaptation/assets/scenes/`.
Each contains one non-flat patch centered at XY zero. These files are static:
editing the main Python terrain configuration does not update them.

For standalone adaptation with custom terrain, supply a terrain-only MJCF
with a body named `terrain`, following the
[model requirements](configuration.md#adaptation-model-requirements). Use the
same scene for adaptation and [motion replay](../guides/motion-adaptation.md#replay-on-the-same-scene).

## Inspect a scene

Export the full task scene from the repository root:

```bash
uv run python -m perceptive_bfm.scripts.export_scene \
  Perceptive-PyramidTerrain-Unitree-G1-Teacher-Symmetry \
  --output-dir logs/terrain_scene
```

To export only the ordinary stairs patch, select terrain column `3` and level
`0`:

```bash
uv run python -m perceptive_bfm.scripts.export_scene \
  Perceptive-PyramidTerrain-Unitree-G1-Teacher-Symmetry \
  --subterrain 3 0 \
  --output-dir logs/terrain_stairs
```

Column indices follow the table above, starting with flat ground at `0`.
The selected patch is centered for inspection. These exports include the task's
robot and sensors; standalone adaptation takes a terrain-only scene.
Use a dedicated output directory that does not exist or is empty. The exporter
rejects nonempty directories and directory symlinks. With `--zip`, it also
rejects an existing destination archive; choose a new output path to retain
previous exports.
