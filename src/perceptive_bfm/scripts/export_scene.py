"""Export a perceptive control task scene to a directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
import tyro.conf
from mjlab.scene import Scene

import perceptive_bfm.tasks  # noqa: F401
from perceptive_bfm.tasks.registry import load_env_cfg


@dataclass
class ExportSceneCfg:
    target: tyro.conf.Positional[str]
    """Registered perceptive control task ID."""

    output_dir: str = "export"
    """Output directory; must be absent or empty. Existing archives are not overwritten."""

    zip: bool = False
    """Compress the exported scene into a zip archive."""

    subterrain: tuple[int, int] | None = None
    """Optional ``(terrain_type, terrain_level)`` patch selector."""


def build_scene(task_id: str, subterrain: tuple[int, int] | None = None) -> Scene:
    """Build a complete task scene or one centered terrain patch."""
    env_cfg = load_env_cfg(task_id)
    if subterrain is not None:
        terrain_type, terrain_level = subterrain
        terrain_cfg = env_cfg.scene.terrain
        if (
            terrain_cfg is None
            or terrain_cfg.terrain_type != "generator"
            or terrain_cfg.terrain_generator is None
        ):
            raise ValueError("--subterrain requires a generated terrain scene")
        generator_cfg = terrain_cfg.terrain_generator

        subterrain_names = list(generator_cfg.sub_terrains)
        if not 0 <= terrain_type < len(subterrain_names):
            raise ValueError(
                f"terrain TYPE must be in [0, {len(subterrain_names) - 1}], " f"got {terrain_type}"
            )
        if not 0 <= terrain_level < generator_cfg.num_rows:
            raise ValueError(
                f"terrain LEVEL must be in [0, {generator_cfg.num_rows - 1}], "
                f"got {terrain_level}"
            )
        selected_name = subterrain_names[terrain_type]
        lower, upper = generator_cfg.difficulty_range
        difficulty = lower + (upper - lower) * terrain_level / max(generator_cfg.num_rows - 1, 1)

        generator_cfg.sub_terrains = {selected_name: generator_cfg.sub_terrains[selected_name]}
        generator_cfg.num_rows = 1
        generator_cfg.num_cols = 1
        generator_cfg.difficulty_range = (difficulty, difficulty)
    return Scene(env_cfg.scene, device="cpu")


def export_scene(cfg: ExportSceneCfg) -> None:
    """Export a registered task's complete scene or one selected terrain patch."""
    output_dir = Path(cfg.output_dir)
    exported_path = output_dir.with_suffix(".zip") if cfg.zip else output_dir
    if output_dir.is_symlink():
        raise FileExistsError(f"Refusing to export through a directory symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Export directory must be absent or empty: {output_dir}")
    if cfg.zip:
        if exported_path == output_dir:
            raise ValueError("ZIP output directory must not itself have a .zip suffix")
        if exported_path.exists() or exported_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing archive: {exported_path}")

    scene = build_scene(cfg.target, subterrain=cfg.subterrain)
    scene.write(output_dir, zip=cfg.zip)
    print(f"Exported to {exported_path}")


def parse_args(args: list[str] | None = None) -> ExportSceneCfg:
    """Parse CLI arguments using two tokens for fixed-size tuples."""
    return tyro.cli(
        ExportSceneCfg,
        args=args,
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )


def main() -> None:
    export_scene(parse_args())


if __name__ == "__main__":
    main()
