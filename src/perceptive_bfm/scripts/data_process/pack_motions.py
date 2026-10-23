r"""Pack grouped qpos/fps motions into minimal per-terrain memmap datasets.

Example (from the repository root)::

    uv run python -m perceptive_bfm.scripts.data_process.pack_motions \
        --input-dir datasets/g1/pyramid \
        --output-dir dataset/g1/pyramid
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from perceptive_bfm.utils.motion import (
    _collect_pack_source_protection,
    _validate_pack_destination,
    create_motion_dataset_from_files,
)
from perceptive_bfm.utils.motion_tree import discover_grouped_motion_files


def _resolve_joint_names() -> list[str]:
    """Read the formal G1 joint order without constructing a training environment."""
    import mujoco

    from perceptive_bfm.asset_zoo.unitree_g1_description import G1_XML

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    return [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]


def _reject_unexpected_group_dirs(output_root: str, expected_groups: set[str]) -> None:
    root = Path(output_root)
    if not root.is_dir():
        return
    unexpected = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name not in expected_groups
    )
    if unexpected:
        raise ValueError(
            f"Output root {output_root} contains unexpected packed group directories: {unexpected}. Use a fresh output directory."
        )


def pack_perceptive_dataset(
    input_dir: str,
    output_dir: str,
    *,
    joint_names: Sequence[str],
    target_fps: float | None = None,
    num_workers: int = 1,
) -> list[str]:
    """Read and validate motions once, publishing only after all groups pass.

    Empty terrain groups produce no output directory. All groups must share one
    FPS; target_fps asserts that rate and never resamples a clip.
    """
    root = Path(os.path.abspath(Path(input_dir).expanduser()))
    if not root.is_dir():
        raise ValueError(f"input_dir must be a directory, got: {input_dir}")
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if target_fps is not None and (not math.isfinite(target_fps) or target_fps <= 0):
        raise ValueError("--target-fps must be finite and positive")
    discovered_groups = discover_grouped_motion_files(
        root, num_workers=num_workers, include_empty_groups=True
    )
    if not discovered_groups:
        raise ValueError(f"No group subdirectories found under {input_dir}")
    group_files = {}
    for group_name, source_files in discovered_groups.items():
        if not source_files:
            print(f"Skipping group '{group_name}': no motion.npz clips found.")
            continue
        group_files[group_name] = source_files
    if not group_files:
        raise ValueError(f"No motion.npz clips found under group directories in {input_dir}")
    if "flat" not in group_files:
        raise ValueError("A non-empty 'flat' group is required")
    _reject_unexpected_group_dirs(output_dir, set(group_files))
    all_files = {
        f"{group}/{name}": path
        for group, files in group_files.items()
        for name, path in files.items()
    }
    destinations = {}
    destination_ids = {}
    source_protection = _collect_pack_source_protection(
        root, all_files, num_workers=num_workers
    )
    for group_name in tqdm(
        group_files, desc="checking output paths", unit="group", dynamic_ncols=True
    ):
        _validate_pack_destination(
            Path(output_dir) / group_name,
            base_dir=root,
            source_files=all_files,
            destinations=destinations,
            destination_ids=destination_ids,
            _source_protection=source_protection,
        )
    # IDs are already known from discovery; numeric checks share the packing read.
    flat_files = group_files["flat"]
    with tqdm(
        total=len(all_files), desc="checking flat motion IDs", unit="clip", dynamic_ncols=True
    ) as progress:
        for group_name, files in group_files.items():
            for name in files:
                if name not in flat_files:
                    raise ValueError(
                        f"Group '{group_name}' motion '{name}' is not present on the flat group"
                    )
                progress.update(1)
    output_root = Path(os.path.abspath(Path(output_dir).expanduser()))
    existing_output = output_root.exists() or output_root.is_symlink()
    if existing_output and not output_root.is_dir():
        raise ValueError(f"output_dir must be a directory: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    flat_lengths: dict[str, int] = {}
    fps = None
    schema_cache = {}
    pack_order = ["flat"] + [group for group in group_files if group != "flat"]
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-", dir=output_root.parent
    ) as temporary:
        staging = Path(temporary)
        for group_name in pack_order:

            def validate_clip(name: str, length: int, rate: float) -> None:
                nonlocal fps
                if fps is None:
                    fps = rate
                if rate != fps:
                    raise ValueError(
                        f"fps mismatch in {group_files[group_name][name]}: dataset fps={fps}, clip fps={rate}"
                    )
                if target_fps is not None and rate != target_fps:
                    raise ValueError(
                        f"Group '{group_name}' fps={rate} does not match --target-fps={target_fps}"
                    )
                if group_name == "flat":
                    flat_lengths[name] = length
                elif length != flat_lengths[name]:
                    raise ValueError(
                        f"Paired frame count mismatch for {group_name}/{name}: {length} vs flat {flat_lengths[name]}"
                    )

            create_motion_dataset_from_files(
                group_files[group_name],
                str(staging / group_name),
                base_dir=root / group_name,
                joint_names=joint_names,
                num_workers=num_workers,
                _source_protection=source_protection,
                _schema_cache=schema_cache,
                _validate_clip=validate_clip,
            )

        # Recheck final destinations after staging, before touching existing packs.
        _reject_unexpected_group_dirs(output_dir, set(group_files))
        destinations, destination_ids = {}, {}
        for group_name in group_files:
            _validate_pack_destination(
                output_root / group_name,
                base_dir=root,
                source_files=all_files,
                destinations=destinations,
                destination_ids=destination_ids,
                _source_protection=source_protection,
            )
        if not existing_output:
            if output_root.exists() or output_root.is_symlink():
                raise FileExistsError(f"Output appeared during packing: {output_root}")
            with tqdm(total=1, desc="publishing packed dataset", unit="dataset") as progress:
                staging.rename(output_root)
                progress.update(1)
        else:
            for group_name in tqdm(group_files, desc="publishing packed groups", unit="group"):
                destination = output_root / group_name
                if destination.exists() or destination.is_symlink():
                    shutil.copytree(staging / group_name, destination, dirs_exist_ok=True)
                else:
                    (staging / group_name).rename(destination)
    return list(group_files)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory of <group>/<motion-relative-path>/motion.npz clips",
    )
    ap.add_argument("--output-dir", required=True, help="Output packed-dataset root directory")
    ap.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="If set, assert every group's fps equals this value",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Threads for motion discovery, source checks, validation and packing reads (default: 1)",
    )
    args = ap.parse_args()
    group_names = pack_perceptive_dataset(
        args.input_dir,
        args.output_dir,
        joint_names=_resolve_joint_names(),
        target_fps=args.target_fps,
        num_workers=args.num_workers,
    )
    print(
        f"Packed '{args.input_dir}' -> '{args.output_dir}' ({len(group_names)} groups: {', '.join(group_names)})."
    )


if __name__ == "__main__":
    main()
