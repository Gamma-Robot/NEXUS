"""Input discovery and deterministic output-path construction."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveredMotion:
    path: Path
    relative_path: Path


@dataclass(frozen=True)
class SceneFile:
    path: Path | None
    name: str


def iter_motion_paths(directory: Path) -> Iterator[Path]:
    """Follow directory links, retaining aliases and pruning ancestor cycles."""
    if not directory.is_dir():
        return
    pending: list[tuple[Path, frozenset[tuple[int, int]]]] = [(directory, frozenset())]
    while pending:
        current, ancestors = pending.pop()
        status = current.stat()
        identity = (status.st_dev, status.st_ino)
        if identity in ancestors:
            continue
        branch = ancestors | {identity}
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir():
                    pending.append((Path(entry.path), branch))
                elif entry.name == "motion.npz" and entry.is_file():
                    yield Path(entry.path)


def _scan_motion_directory(
    directory: Path, ancestors: frozenset[tuple[int, int]]
) -> tuple[list[Path], list[tuple[Path, frozenset[tuple[int, int]]]]]:
    status = directory.stat()
    identity = (status.st_dev, status.st_ino)
    if identity in ancestors:
        return [], []
    branch = ancestors | {identity}
    files = []
    children = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                children.append((Path(entry.path), branch))
            elif entry.name == "motion.npz" and entry.is_file():
                files.append(Path(entry.path))
    return files, children


def _iter_motion_paths_parallel(
    directory: Path, num_workers: int, show_progress: bool
) -> Iterator[Path]:
    from tqdm import tqdm

    queued = [(directory, frozenset())]
    pending = set()
    found = 0
    with (
        ThreadPoolExecutor(max_workers=num_workers) as pool,
        tqdm(
            desc="discovering motions",
            unit="dir",
            ncols=max(1, shutil.get_terminal_size().columns - 1),
            disable=not show_progress,
        ) as progress,
    ):
        while queued or pending:
            # Split work at every directory, including branches within one large source.
            while queued and len(pending) < 2 * num_workers:
                path, ancestors = queued.pop()
                pending.add(pool.submit(_scan_motion_directory, path, ancestors))
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                files, children = future.result()
                queued.extend(children)
                found += len(files)
                progress.update(1)
                progress.set_postfix(motions=found, pending=len(queued) + len(pending), refresh=False)
                yield from files


def discover_motions(
    input_path: str | Path,
    base_dir: str | Path,
    *,
    num_workers: int = 1,
    show_progress: bool = False,
) -> list[DiscoveredMotion]:
    """Discover motions with optional concurrent directory scans and progress.

    Results retain sorted logical paths; aliases remain distinct and ancestor
    cycles terminate independently for each traversal branch.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    source = Path(os.path.abspath(Path(input_path).expanduser()))
    base = Path(os.path.abspath(Path(base_dir).expanduser()))
    if not base.is_dir():
        raise FileNotFoundError(f"base_dir not found: {base}")
    if source.is_file():
        if source.name != "motion.npz":
            raise ValueError(f"input file must be named motion.npz: {source}")
        paths = [source]
    elif source.is_dir():
        paths = list(
            _iter_motion_paths_parallel(source, num_workers, show_progress)
            if num_workers > 1 or show_progress
            else iter_motion_paths(source)
        )
        if not paths:
            raise ValueError(f"no motion.npz files found under input_dir: {source}")
    else:
        raise FileNotFoundError(f"input_dir not found: {source}")

    discovered: list[DiscoveredMotion] = []
    for path in paths:
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"input motion must be inside base_dir: {path} (base_dir={base})"
            ) from exc
        discovered.append(DiscoveredMotion(path=path, relative_path=relative))
    return sorted(discovered, key=lambda motion: motion.relative_path.as_posix())


def resolve_scenes(scene_paths: Iterable[str | Path] | None) -> list[SceneFile]:
    """Resolve terrain-only scene XMLs, defaulting to the built-in plane."""
    if scene_paths is None:
        return [SceneFile(path=None, name="plane")]

    scenes: list[SceneFile] = []
    for value in scene_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene XML not found: {path}")
        if path.suffix.lower() != ".xml":
            raise ValueError(f"scene must be an XML file: {path}")
        scenes.append(SceneFile(path=path, name=path.stem))
    if not scenes:
        raise ValueError("at least one scene XML is required")
    names = [scene.name for scene in scenes]
    if len(names) != len(set(names)):
        raise ValueError(f"scene names must be unique, got: {names}")
    return scenes


def output_motion_path(
    output_dir: str | Path, scene_name: str, relative_motion_path: str | Path
) -> Path:
    """Map one source identity to its scene-specific output path."""
    relative = Path(relative_motion_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"relative motion path is unsafe: {relative}")
    return Path(output_dir).expanduser().resolve() / scene_name / relative
