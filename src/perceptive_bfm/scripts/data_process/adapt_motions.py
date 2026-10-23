"""Adapt flat G1 motions to the formal PyramidTerrain patches.

Inputs use the shared qpos/fps motion protocol and retain logical paths through
symlinks. Each worker builds and reuses the full terrain model in memory; only
patch XY offsets are passed to the adaptation pipeline. Outputs retain absolute
terrain Z and are grouped as <type>_lv0/<relative-path>/motion.npz.
"""

from __future__ import annotations

# ruff: noqa: E402 -- thread limits must precede numerical imports.

import os
import shutil

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import dataclasses
import json
import multiprocessing as mp
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal, NamedTuple

import tyro
from tqdm import tqdm

from perceptive_bfm.terrains.config import _build_pyramid_terrain, build_terrain_model
from terrain_motion_adaptation.optimization.contact import EEContactCfg
from terrain_motion_adaptation.optimization.keyframe_ik import KeyframeIKCfg
from terrain_motion_adaptation.optimization.pipeline import (
    MotionAdaptationConfig,
    NoActiveKeyframesError,
    terrain_adapt_qpos,
)
from terrain_motion_adaptation.robots.g1 import G1_CONFIG
from terrain_motion_adaptation.utils.model import validate_adaptation_model, validate_qpos_schema
from terrain_motion_adaptation.utils.motion import (
    OutputManifestValidator,
    Motion,
    MotionMetadata,
    load_motion,
    load_motion_metadata_batch,
    load_motion_schemas_batch,
    write_json_atomic,
    write_motion,
)
from terrain_motion_adaptation.utils.paths import discover_motions

DEFAULT_INPUT_DIR = "datasets/g1/flat"
MANIFEST_NAME = "flat_to_terrain_manifest.json"
Terrain = Literal["pyramid"]


class Job(NamedTuple):
    rel_path: str  # source clip path relative to base_dir (ends in .../motion.npz)
    col: int  # terrain-type index (grid column)
    row: int  # terrain level index (grid row)
    terrain_type: str


class JobResult(NamedTuple):
    job: Job
    status: str  # "done" | "kept" | "skipped" | "failed"
    error: str | None


@dataclass(frozen=True)
class WorkerConfig:
    """Constant context shared by every job; broadcast once via the initializer."""

    terrain: str
    base_dir: str
    output_dir: str
    contact_cfg: EEContactCfg
    ik_cfg: KeyframeIKCfg
    skip_existing: bool = False


def build_jobs(
    rel_paths: list[str],
    subterrain_names: list[str],
    num_levels: int,
    selected_types: list[str] | None,
    selected_levels: list[int] | None,
    output_dir: Path,
    skip_existing: bool,
    *,
    existing_outputs: set[Path] | None = None,
) -> list[Job]:
    """Every (clip, non-flat sub-terrain, level) still needing an output.

    ``'flat'`` is never a target: it is the *source* group, and the perceptive
    dataset takes it straight from the untouched flat clips.
    """
    jobs: list[Job] = []
    for col, terrain_type in enumerate(subterrain_names):
        if terrain_type == "flat":
            continue
        if selected_types is not None and terrain_type not in selected_types:
            continue
        for row in range(num_levels):
            if selected_levels is not None and row not in selected_levels:
                continue
            for rel_path in rel_paths:
                out = output_dir / f"{terrain_type}_lv{row}" / rel_path
                if skip_existing and (
                    out in existing_outputs if existing_outputs is not None else out.exists()
                ):
                    continue
                jobs.append(Job(rel_path=rel_path, col=col, row=row, terrain_type=terrain_type))
    return jobs


def _find_existing_outputs(
    metadata: list[MotionMetadata], base_dir: Path, group_root: Path, *, num_workers: int
) -> set[Path]:
    """Validate existing clips with bounded concurrency and retain skip decisions."""
    def check(info):
        path = group_root / info.path.relative_to(base_dir)
        if not path.exists():
            return None
        source = load_motion(info.path, base_dir=base_dir)
        _check_existing_output(path, source, group_root)
        return path

    existing: set[Path] = set()
    with (
        ThreadPoolExecutor(max_workers=num_workers) as pool,
        tqdm(
            total=len(metadata), desc=f"checking existing {group_root.name}", unit="clip",
            ncols=max(1, shutil.get_terminal_size().columns - 1),
        ) as progress,
    ):
        iterator = iter(metadata)
        while batch := list(islice(iterator, 4 * num_workers)):
            for path in pool.map(check, batch):
                if path is not None:
                    existing.add(path)
                progress.update(1)
    return existing


def _check_existing_output(path: Path, source: Motion, group_root: Path) -> None:
    """Reject incompatible existing data without overwriting it."""
    try:
        old = load_motion(path, base_dir=group_root)
        if (
            old.joint_names != source.joint_names
            or old.fps != source.fps
            or old.qpos.shape != source.qpos.shape
        ):
            raise ValueError("schema, FPS or frame shape differs")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"existing output is not compatible; use --no-skip-existing: {path}"
        ) from exc


_WCFG: WorkerConfig | None = None
_STATE: dict | None = None


def _get_state() -> dict:
    """Build the full terrain model once per worker and reuse it across jobs."""
    global _STATE
    if _STATE is not None:
        return _STATE
    assert _WCFG is not None
    model, origins, _names = build_terrain_model(_WCFG.terrain)
    _STATE = {"model": model, "origins": origins}
    return _STATE


def _process_job(job: Job) -> JobResult:
    """Report each motion independently, preserving successful outputs."""
    assert _WCFG is not None
    try:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            status = _run_single(job)
        return JobResult(job, status, None)
    except NoActiveKeyframesError as exc:
        return JobResult(job, "skipped", str(exc))
    except Exception:
        return JobResult(job, "failed", traceback.format_exc())


def _run_single(job: Job) -> Literal["done", "kept"]:
    assert _WCFG is not None
    source = load_motion(Path(_WCFG.base_dir) / job.rel_path, base_dir=_WCFG.base_dir)
    group_root = Path(_WCFG.output_dir) / f"{job.terrain_type}_lv{job.row}"
    destination = group_root / job.rel_path
    if _WCFG.skip_existing and destination.exists():
        _check_existing_output(destination, source, group_root)
        return "kept"
    state = _get_state()
    origin = state["origins"][job.row, job.col]
    qpos = terrain_adapt_qpos(
        model=state["model"],
        robot=G1_CONFIG,
        qpos=source.qpos,
        qpos_names=source.qpos_names,
        timestep=source.timestep,
        config=MotionAdaptationConfig(contact=_WCFG.contact_cfg, keyframe_ik=_WCFG.ik_cfg),
        dx=float(origin[0]),
        dy=float(origin[1]),
    )
    write_motion(destination, qpos, fps=source.fps)
    return "done"


def _resolve_inputs(
    base_dir: str, input_dir: str | None, *, num_workers: int = 1
) -> tuple[list[Path], Path]:
    """Discover protocol motions while retaining logical relative identities."""
    base = Path(os.path.abspath(Path(base_dir).expanduser()))
    discovered = discover_motions(
        input_dir or base, base, num_workers=num_workers, show_progress=True
    )
    return [motion.path for motion in discovered], base


def _init_worker(wcfg: WorkerConfig) -> None:
    """Pool initializer: stash config and make Ctrl-C the parent's job only."""
    global _WCFG, _STATE
    _WCFG = wcfg
    _STATE = None
    # Ignore SIGINT in workers so a Ctrl-C in the parent triggers an orderly
    # pool.terminate() instead of a traceback storm from every child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _default_num_workers() -> int:
    """Cores actually allocated to this process (respects SLURM/cgroup limits)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on all platforms
        return max(1, os.cpu_count() or 1)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def _mjlab_version_info() -> dict[str, Any]:
    """Best-effort mjlab version identity, for the drift-diagnosis manifest.

    Records the PyPI version string and, if mjlab was installed from git (not a
    released wheel), the exact commit -- this is what actually determines terrain
    generation behavior.
    """
    import importlib.metadata as md

    info: dict[str, Any] = {"version": None, "commit": None}
    try:
        info["version"] = md.version("mjlab")
    except md.PackageNotFoundError:
        pass
    try:
        raw = md.distribution("mjlab").read_text("direct_url.json")
        if raw:
            vcs_info = json.loads(raw).get("vcs_info")
            if vcs_info:
                info["commit"] = vcs_info.get("commit_id")
    except Exception:  # noqa: BLE001 - best-effort only
        pass
    return info


def _write_manifest(
    output_dir: str,
    terrain: str,
    subterrain_names: list[str],
    num_levels: int,
    contact_cfg: EEContactCfg,
    ik_cfg: KeyframeIKCfg,
) -> None:
    """Retain the existing generation report, separate from motion manifests."""
    generator_cfg = _build_pyramid_terrain().terrain_generator
    assert generator_cfg is not None
    manifest = {
        "pipeline": "smoothed_keyframe_ik",
        "mjlab": _mjlab_version_info(),
        "terrain": terrain,
        "terrain_generator": {
            "seed": generator_cfg.seed,
            "num_rows": generator_cfg.num_rows,
            "sub_terrains": subterrain_names,
        },
        "num_levels": num_levels,
        "contact_cfg": dataclasses.asdict(contact_cfg),
        "ik_cfg": dataclasses.asdict(ik_cfg),
    }
    path = Path(output_dir) / MANIFEST_NAME
    write_json_atomic(path, manifest)
    print(f"[INFO] Wrote {path}", flush=True)


def main(
    output_dir: str,
    input_dir: str = DEFAULT_INPUT_DIR,
    base_dir: str | None = None,
    terrain: Terrain = "pyramid",
    subterrain_types: list[str] | None = None,
    subterrain_levels: list[int] | None = None,
    num_workers: int | None = None,
    skip_existing: bool = True,
    preflight: Literal["light", "full"] = "light",
    max_tasks_per_child: int = 0,
    contact_cfg: EEContactCfg = EEContactCfg(),
    ik_cfg: KeyframeIKCfg = KeyframeIKCfg(),
) -> None:
    """Generate selected non-flat patches from minimal flat motion clips.

    base_dir defines logical relative paths and the manifest search boundary.
    Supply it explicitly when selecting a single motion file. Existing outputs
    are skipped only when their schema, FPS and frame shape remain compatible.
    Light preflight defers per-file checks to workers; full checks them up front.
    """
    if preflight not in ("light", "full"):
        raise ValueError(f"unknown preflight mode: {preflight}")
    output_dir = str(Path(output_dir).expanduser().resolve())
    if base_dir is None:
        base_dir = input_dir
    avail = _default_num_workers()
    if num_workers is None:
        num_workers = 1
        print(
            "[INFO] num_workers not set -> running SINGLE-THREADED (safe default). "
            f"Pass --num_workers N to parallelize (this host has ~{avail} allocated cores)."
        )
    num_workers = max(1, num_workers)
    if num_workers > avail:
        print(
            f"[WARN] num_workers={num_workers} exceeds the {avail} cores allocated "
            f"to this job; capping to {avail} to avoid oversubscription."
        )
        num_workers = avail
    print(f"[INFO] Discovering input motions with up to {num_workers} threads ...", flush=True)
    input_files, base_root = _resolve_inputs(base_dir, input_dir, num_workers=num_workers)
    base_dir = str(base_root)
    rel_paths = [str(path.relative_to(base_root)) for path in input_files]
    physical_base = base_root.resolve()
    physical_output = Path(output_dir)
    if physical_base.is_relative_to(physical_output) or physical_output.is_relative_to(
        physical_base
    ):
        raise ValueError("input base_dir and output_dir must be disjoint directory trees")
    print(f"[INFO] Relative-path root (base_dir): {base_dir}")
    print(f"[INFO] Selected clips: {len(rel_paths)}", flush=True)
    print(f"[INFO] Building '{terrain}' terrain grid to enumerate jobs ...", flush=True)
    model, origins, subterrain_names = build_terrain_model(terrain)
    num_levels = int(origins.shape[0])
    non_flat = [name for name in subterrain_names if name != "flat"]
    if subterrain_types is not None:
        unknown = [name for name in subterrain_types if name not in subterrain_names]
        if unknown:
            raise ValueError(f"unknown --subterrain_types {unknown}; available: {non_flat}")
        if "flat" in subterrain_types:
            print("[WARN] 'flat' in --subterrain_types is ignored (it is the source group).")
        subterrain_types = [name for name in subterrain_types if name != "flat"]
        if not subterrain_types:
            raise ValueError(f"no sub-terrain types selected; available: {non_flat}")
    if subterrain_levels is not None and -1 in subterrain_levels:
        subterrain_levels = None
    if subterrain_levels is not None:
        bad = [level for level in subterrain_levels if level < 0 or level >= num_levels]
        if bad:
            raise ValueError(f"--subterrain_levels {bad} out of range [0, {num_levels - 1}]")
    selected_types = subterrain_types if subterrain_types is not None else non_flat
    selected_levels = (
        subterrain_levels if subterrain_levels is not None else list(range(num_levels))
    )
    groups = [f"{name}_lv{level}" for name in selected_types for level in selected_levels]
    loader = load_motion_metadata_batch if preflight == "full" else load_motion_schemas_batch
    metadata = loader(
        input_files, base_dir=base_root, num_workers=num_workers, show_progress=True
    )
    validate_adaptation_model(model, G1_CONFIG)
    for names in dict.fromkeys(info.qpos_names for info in metadata):
        validate_qpos_schema(model, names)
    output_validator = OutputManifestValidator(
        base_dir=base_root,
        motions=metadata,
        num_workers=num_workers,
        show_progress=True,
    )
    existing_outputs: set[Path] = set()
    for group in groups:
        group_root = Path(output_dir) / group
        print(f"[INFO] Validating output paths and schemas: {group}", flush=True)
        output_validator.validate(group_root)
        if skip_existing and preflight == "full":
            existing_outputs.update(_find_existing_outputs(
                metadata, base_root, group_root, num_workers=num_workers
            ))
    output_validator.validate_file_destination(Path(output_dir) / MANIFEST_NAME)
    for group in groups:
        print(f"[INFO] Writing validated joint manifests: {group}", flush=True)
        output_validator.write(Path(output_dir) / group)
    _write_manifest(output_dir, terrain, subterrain_names, num_levels, contact_cfg, ik_cfg)
    jobs = build_jobs(
        rel_paths,
        subterrain_names,
        num_levels,
        subterrain_types,
        subterrain_levels,
        Path(output_dir),
        skip_existing and preflight == "full",
        existing_outputs=existing_outputs,
    )
    total_targets = len(rel_paths) * len(selected_types) * len(selected_levels)
    print(
        f"[INFO] Targets: {total_targets} | queued: {len(jobs)} | "
        f"already checked/skipped: {total_targets - len(jobs)} | workers: {num_workers} | "
        f"preflight: {preflight}",
        flush=True,
    )
    if not jobs:
        print("[INFO] Nothing to do.")
        return

    # 4) Run the pool with a live progress bar + clean teardown.
    wcfg = WorkerConfig(
        terrain=terrain,
        base_dir=base_dir,
        output_dir=output_dir,
        contact_cfg=contact_cfg,
        ik_cfg=ik_cfg,
        skip_existing=skip_existing and preflight == "light",
    )
    ctx = mp.get_context("spawn")  # MuJoCo-safe; never fork
    maxtasks = max_tasks_per_child if max_tasks_per_child > 0 else None

    n_done = n_skipped = n_failed = 0
    n_kept = total_targets - len(jobs)
    failures: list[JobResult] = []
    start = time.time()

    pool = ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(wcfg,),
        maxtasksperchild=maxtasks,
    )
    try:
        with tqdm(
            total=len(jobs), unit="clip", ncols=max(1, shutil.get_terminal_size().columns - 1)
        ) as pbar:
            for res in pool.imap_unordered(_process_job, jobs):
                if res.status == "failed":
                    n_failed += 1
                    failures.append(res)
                    reason = res.error.strip().splitlines()[-1] if res.error else "unknown error"
                    tqdm.write(
                        f"[FAIL] {res.job.terrain_type}_lv{res.job.row}/{res.job.rel_path}: {reason}",
                        file=pbar.fp,
                    )
                elif res.status == "kept":
                    n_kept += 1
                elif res.status == "skipped":
                    n_skipped += 1
                    tqdm.write(
                        f"[SKIP] {res.job.terrain_type}_lv{res.job.row}/{res.job.rel_path}: "
                        f"{res.error}",
                        file=pbar.fp,
                    )
                else:
                    n_done += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_done, kept=n_kept, skip=n_skipped, fail=n_failed, refresh=False)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted — terminating workers ...")
        pool.terminate()
        pool.join()
        print("[WARN] Workers stopped.")
        sys.exit(130)
    finally:
        pool.terminate()
        pool.join()

    # 5) Summary.
    elapsed = time.time() - start
    print("\n" + "=" * 40)
    print("Batch terrain adaptation complete!")
    print(f"Succeeded       : {n_done}")
    print(f"Already present : {n_kept}")
    print(f"Skipped         : {n_skipped}")
    print(f"Failed          : {n_failed}")
    print(f"Total processed : {len(jobs)}")
    print(f"Total time      : {_fmt_duration(elapsed)}")
    if n_done > 0 and elapsed > 0:
        print(f"Average speed   : {len(jobs) / elapsed:.2f} clips/sec")
    if failures:
        print("-" * 40)
        print("Failed jobs (first 10 tracebacks):")
        for res in failures[:10]:
            print(f"\n### {res.job.terrain_type}_lv{res.job.row}/{res.job.rel_path}\n{res.error}")
    print("=" * 40)
    if n_kept and not (n_done or n_skipped or n_failed):
        print("[INFO] Nothing to do.")


if __name__ == "__main__":
    tyro.cli(main)
