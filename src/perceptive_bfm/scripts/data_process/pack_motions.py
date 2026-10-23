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
import os.path as osp
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from perceptive_bfm.utils.motion import (
    _collect_pack_source_protection,
    _load_motion,
    _pack_map,
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
    """Pack nonempty terrain groups after validating every source and flat pairing.

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
    # Validate every group before writes; a late bad clip cannot leave earlier packs.
    lengths = {group: {} for group in group_files}
    schema_cache = {}

    def validate(item):
        group_name, name, path = item
        qpos, rate = _load_motion(
            path,
            base_dir=root / group_name,
            joint_names=joint_names,
            _schema_cache=schema_cache,
        )
        return group_name, name, len(qpos), rate

    items = (
        (group_name, name, path)
        for group_name, files in group_files.items()
        for name, path in files.items()
    )
    fps = None
    for group_name, name, length, rate in _pack_map(
        validate,
        items,
        total=len(all_files),
        num_workers=num_workers,
        desc="validating motions",
    ):
        lengths[group_name][name] = length
        if fps is None:
            fps = rate
        if rate != fps:
            path = group_files[group_name][name]
            raise ValueError(f"fps mismatch in {path}: dataset fps={fps}, clip fps={rate}")
        if target_fps is not None and rate != target_fps:
            raise ValueError(
                f"Group '{group_name}' fps={rate} does not match --target-fps={target_fps}"
            )
    flat = lengths["flat"]
    with tqdm(
        total=len(all_files), desc="checking flat pairing", unit="clip", dynamic_ncols=True
    ) as progress:
        for group_name, group_lengths in lengths.items():
            for name, length in group_lengths.items():
                if name not in flat:
                    raise ValueError(
                        f"Group '{group_name}' motion '{name}' is not present on the flat group"
                    )
                if length != flat[name]:
                    raise ValueError(
                        f"Paired frame count mismatch for {group_name}/{name}: {length} vs flat {flat[name]}"
                    )
                progress.update(1)
    for group_name in group_files:
        create_motion_dataset_from_files(
            group_files[group_name],
            osp.join(output_dir, group_name),
            base_dir=root / group_name,
            joint_names=joint_names,
            num_workers=num_workers,
            _source_protection=source_protection,
            _schema_cache=schema_cache,
        )
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
