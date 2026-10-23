"""Shared terrain configuration for motion generation and perceptive training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from perceptive_bfm.asset_zoo.unitree_g1_description import G1_XML

if TYPE_CHECKING:
    from mjlab.terrains import TerrainEntityCfg


def _build_pyramid_terrain_grid(
    *,
    num_rows: int,
    slope_range: tuple[float, float],
    step_height_range: tuple[float, float],
) -> TerrainEntityCfg:
    """Build five terrain columns with explicit difficulty ranges."""
    import mjlab.terrains as terrain_gen
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(20.0, 20.0),
            num_rows=num_rows,
            curriculum=True,
            sub_terrains={
                "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    slope_range=slope_range,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    slope_range=slope_range,
                    inverted=True,
                ),
                "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=step_height_range,
                    step_width=0.3,
                ),
                "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=step_height_range,
                    step_width=0.3,
                ),
            },
            add_lights=True,
        ),
    )


def _build_pyramid_terrain() -> TerrainEntityCfg:
    """Build the fixed single-level terrain for data generation and training."""
    return _build_pyramid_terrain_grid(
        num_rows=1,
        slope_range=(0.3, 0.3),
        step_height_range=(0.1, 0.1),
    )


def build_terrain_model(terrain: str) -> tuple[mujoco.MjModel, np.ndarray, list[str]]:
    """Compile bare G1 and the full grid, retaining training patch origins."""
    if terrain != "pyramid":
        raise ValueError(f"only the formal pyramid terrain is supported, got {terrain!r}")
    from mjlab.terrains.terrain_generator import TerrainGenerator

    generator_cfg = _build_pyramid_terrain().terrain_generator
    assert generator_cfg is not None
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    generator = TerrainGenerator(generator_cfg)
    generator.compile(spec)
    return spec.compile(), generator.terrain_origins, list(generator_cfg.sub_terrains)
