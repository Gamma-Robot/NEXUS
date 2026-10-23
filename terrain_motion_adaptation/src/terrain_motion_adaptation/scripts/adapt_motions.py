#!/usr/bin/env python3
"""Batch terrain-adapt robot motions across terrain scenes."""

from __future__ import annotations

# ruff: noqa: E402 -- thread limits must precede NumPy/SciPy imports.

import os

# Avoid multiplying BLAS/OpenMP pools inside each motion worker.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import multiprocessing as mp
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from terrain_motion_adaptation.optimization.pipeline import terrain_adapt_qpos
from terrain_motion_adaptation.robots.config import RobotConfig
from terrain_motion_adaptation.robots.g1 import G1_CONFIG

from terrain_motion_adaptation.utils.model import (
    combine_model_and_scene,
    validate_qpos_schema,
    validate_adaptation_model,
)
from terrain_motion_adaptation.utils.motion import (
    OutputManifestValidator,
    MotionMetadata,
    load_motion_metadata,
    load_motion_metadata_batch,
    load_motion,
    write_motion,
    validate_qpos,
    _validate_fps,
)
from terrain_motion_adaptation.utils.paths import (
    DiscoveredMotion,
    discover_motions,
    output_motion_path,
    resolve_scenes,
)


@dataclass(frozen=True)
class MotionJob:
    source: Path
    destination: Path
    base_dir: Path


@dataclass(frozen=True)
class JobResult:
    source: Path
    destination: Path
    error: str | None


_WORKER_MODEL: mujoco.MjModel | None = None
_WORKER_ROBOT: RobotConfig | None = None


def _init_worker(model_path: Path, scene_path: Path | None, robot: RobotConfig) -> None:
    global _WORKER_MODEL, _WORKER_ROBOT
    if mp.parent_process() is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_MODEL = combine_model_and_scene(model_path, scene_path)
    _WORKER_ROBOT = robot


def _run_job(job: MotionJob) -> JobResult:
    try:
        if _WORKER_MODEL is None or _WORKER_ROBOT is None:
            raise RuntimeError("motion worker has not loaded its model and robot configuration")
        motion = load_motion(job.source, base_dir=job.base_dir)
        terrain_qpos = terrain_adapt_qpos(
            model=_WORKER_MODEL,
            robot=_WORKER_ROBOT,
            qpos=motion.qpos,
            qpos_names=motion.qpos_names,
            timestep=motion.timestep,
        )
        write_motion(job.destination, terrain_qpos, fps=motion.fps)
        return JobResult(job.source, job.destination, None)
    except Exception:  # noqa: BLE001 - preserve every per-motion traceback for the batch report
        return JobResult(job.source, job.destination, traceback.format_exc())


def _parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, type=Path, help="Robot MJCF XML.")
    parser.add_argument(
        "--scene",
        nargs="+",
        type=Path,
        help="One or more terrain-only MJCF XML files; defaults to a plane.",
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path, help="One NPZ file or a recursive NPZ tree."
    )
    parser.add_argument(
        "--base-dir", required=True, type=Path, help="Base used for source-relative paths."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes per scene.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    return parser.parse_args()


def _validate_roots(input_path: Path, output_dir: Path) -> None:
    source = input_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    source_tree = source if source.is_dir() else source.parent
    if output == source_tree or source_tree in output.parents or output in source_tree.parents:
        raise ValueError("input_dir and output_dir must be disjoint directory trees")


def _load_metadata(motions: list[DiscoveredMotion], base_dir: Path) -> list[MotionMetadata]:
    return load_motion_metadata_batch([motion.path for motion in motions], base_dir=base_dir)


def _existing_output_is_compatible(path: Path, source: MotionMetadata, base_dir: Path) -> bool:
    source_motion = load_motion(source.path, base_dir=base_dir)
    relative = source.path.relative_to(base_dir)
    scene_root = path.parents[len(relative.parts) - 1]
    try:
        output_metadata = load_motion_metadata(path, base_dir=scene_root)
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return False
    else:
        if output_metadata.joint_names != source.joint_names:
            return False
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != 2 or set(loaded.files) != {"qpos", "fps"}:
                return False
            qpos = loaded["qpos"]
            validate_qpos(qpos, source_motion.qpos.shape[1])
            return (
                qpos.shape == source_motion.qpos.shape
                and _validate_fps(loaded["fps"]) == source.fps
            )
    except (OSError, ValueError):
        return False


def _run_scene_jobs(
    *,
    model_path: Path,
    scene_path: Path | None,
    scene_name: str,
    motions: list[DiscoveredMotion],
    metadata: list[MotionMetadata],
    base_dir: Path,
    output_dir: Path,
    num_workers: int,
    overwrite: bool,
    robot: RobotConfig,
) -> tuple[int, int, list[JobResult]]:
    jobs: list[MotionJob] = []
    kept = 0
    for discovered, info in zip(motions, metadata, strict=True):
        destination = output_motion_path(output_dir, scene_name, discovered.relative_path)
        if destination.is_file() and not overwrite:
            if not _existing_output_is_compatible(destination, info, base_dir):
                raise ValueError(
                    f"existing output is not compatible; use --overwrite: {destination}"
                )
            kept += 1
            continue
        jobs.append(MotionJob(discovered.path, destination, base_dir))

    if not jobs:
        return 0, kept, []
    if num_workers == 1:
        _init_worker(model_path, scene_path, robot)
        results = [_run_job(job) for job in tqdm(jobs, desc=scene_name)]
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=min(num_workers, len(jobs)),
            initializer=_init_worker,
            initargs=(model_path, scene_path, robot),
        ) as pool:
            results = list(
                tqdm(pool.imap_unordered(_run_job, jobs), total=len(jobs), desc=scene_name)
            )
    return len(jobs), kept, [result for result in results if result.error is not None]


def run_adaptation(*, robot: RobotConfig, description: str, completion_label: str) -> None:
    """Run the shared multi-scene robot batch pipeline."""
    args = _parse_args(description)
    if args.num_workers < 1:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"robot model XML not found: {model_path}")
    _validate_roots(args.input_dir, args.output_dir)
    scenes = resolve_scenes(args.scene)
    motions = discover_motions(args.input_dir, args.base_dir)
    base_dir = Path(os.path.abspath(args.base_dir.expanduser()))
    output_dir = args.output_dir.expanduser().resolve()
    metadata = _load_metadata(motions, base_dir)

    # Compile every combination before creating output, then validate the shared robot schema.
    models = [combine_model_and_scene(model_path, scene.path) for scene in scenes]
    for model in models:
        validate_adaptation_model(model, robot)
    for names in dict.fromkeys(info.qpos_names for info in metadata):
        validate_qpos_schema(models[0], names)
    output_validator = OutputManifestValidator(base_dir=base_dir, motions=metadata)
    for scene in scenes:
        output_validator.validate(output_dir / scene.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_kept = 0
    all_failures: list[JobResult] = []
    for scene in scenes:
        # Establish the schema before producing clips, including interrupted/failed batches.
        output_validator.write(output_dir / scene.name)
        written, kept, failures = _run_scene_jobs(
            model_path=model_path,
            scene_path=scene.path,
            scene_name=scene.name,
            motions=motions,
            metadata=metadata,
            base_dir=base_dir,
            output_dir=output_dir,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            robot=robot,
        )
        total_written += written
        total_kept += kept
        all_failures.extend(failures)

    if all_failures:
        details = "\n\n".join(
            f"{failure.source} -> {failure.destination}\n{failure.error}"
            for failure in all_failures
        )
        raise RuntimeError(f"{len(all_failures)} terrain-retarget job(s) failed:\n\n{details}")
    total = len(scenes) * len(motions)
    print(
        f"{completion_label}: total={total}, written={total_written}, kept={total_kept}, "
        f"scenes={len(scenes)}, motions={len(motions)}, output={output_dir}"
    )


def main() -> None:
    run_adaptation(
        robot=G1_CONFIG,
        description="Adapt flat robot G1 qpos motions to one or more terrain scenes.",
        completion_label="NEXUS complete",
    )


if __name__ == "__main__":
    main()
