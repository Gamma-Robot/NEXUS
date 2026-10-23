"""Concurrent discovery of grouped ``motion.npz`` dataset trees."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _absolute_lexical_path(path: Path) -> Path:
    """Make ``path`` absolute without resolving symlinks."""
    return Path(os.path.abspath(path.expanduser()))


def _validated_motion_path(entry: os.DirEntry[str]) -> Path:
    path = Path(entry.path)
    if not entry.is_file(follow_symlinks=True):
        raise ValueError(f"{path} is named motion.npz but is not a regular file")
    return path


def available_cpus() -> int:
    """Return the process' usable CPU count, respecting affinity when available."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _scan_source_root(
    dataset_root: Path, motion_root: Path, *, ancestor_ids: frozenset[tuple[int, int]] = frozenset()
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    stat = dataset_root.stat()
    pending = [(motion_root, ancestor_ids | {(stat.st_dev, stat.st_ino)})]
    while pending:
        directory, ancestors = pending.pop()
        stat = directory.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in ancestors:
            continue
        branch = ancestors | {identity}
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name == "motion.npz":
                    path = _validated_motion_path(entry)
                    files[str(path.relative_to(dataset_root))] = path
                elif entry.is_dir(follow_symlinks=True):
                    pending.append((Path(entry.path), branch))
    return files


def _scan_motion_root(group_dir: Path, motion_root: Path) -> tuple[str, dict[str, Path]]:
    stat = group_dir.parent.stat()
    files = _scan_source_root(
        group_dir, motion_root, ancestor_ids=frozenset({(stat.st_dev, stat.st_ino)})
    )
    clips = {str(Path(source_path).parent): path for source_path, path in files.items()}
    return group_dir.name, clips


def discover_motion_files_in_root(
    dataset_root: Path,
    num_workers: int = 1,
    *,
    available_cpu_count: int | None = None,
) -> dict[str, Path]:
    """Return sorted ``{relative/motion.npz: path}`` for one dataset root."""
    from tqdm import tqdm

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    dataset_root = _absolute_lexical_path(dataset_root)
    files: dict[str, Path] = {}
    motion_roots: list[Path] = []
    discovered_motions = 0

    with tqdm(desc="enumerating motion roots", unit="root", dynamic_ncols=True) as progress:
        progress.set_postfix(motions=0, refresh=False)
        with os.scandir(dataset_root) as entries:
            for entry in entries:
                if entry.name == "motion.npz":
                    path = _validated_motion_path(entry)
                    files["motion.npz"] = path
                    discovered_motions += 1
                    progress.update(1)
                elif entry.is_dir(follow_symlinks=True):
                    motion_roots.append(Path(entry.path))
                    progress.update(1)
                progress.set_postfix(motions=discovered_motions, refresh=False)

        motion_roots.sort(key=lambda path: path.name)
        cpu_count = available_cpu_count if available_cpu_count is not None else available_cpus()
        scan_workers = min(num_workers, cpu_count, max(1, len(motion_roots)))
        progress.set_description("scanning motion roots", refresh=False)
        progress.reset(total=len(motion_roots))

        def record(discovered: dict[str, Path]) -> None:
            nonlocal discovered_motions
            files.update(discovered)
            discovered_motions += len(discovered)
            progress.update(1)
            progress.set_postfix(motions=discovered_motions, refresh=False)

        if scan_workers > 1:
            with ThreadPoolExecutor(max_workers=scan_workers) as executor:
                futures = [
                    executor.submit(_scan_source_root, dataset_root, motion_root)
                    for motion_root in motion_roots
                ]
                for future in as_completed(futures):
                    record(future.result())
        else:
            for motion_root in motion_roots:
                record(_scan_source_root(dataset_root, motion_root))

    return {source_path: files[source_path] for source_path in sorted(files)}


def discover_motion_files(
    input_dir: Path,
    num_workers: int = 1,
    *,
    include_empty_groups: bool = False,
    available_cpu_count: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Return ``{group: {relative_motion_path: motion_file}}`` deterministically."""
    from tqdm import tqdm

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    input_dir = _absolute_lexical_path(input_dir)
    with os.scandir(input_dir) as entries:
        group_dirs = sorted(
            (Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=True)),
            key=lambda path: path.name,
        )
    stat = input_dir.stat()
    root_identity = (stat.st_dev, stat.st_ino)
    group_dirs = [
        path for path in group_dirs if (path.stat().st_dev, path.stat().st_ino) != root_identity
    ]
    groups: dict[str, dict[str, Path]] = (
        {group_dir.name: {} for group_dir in group_dirs} if include_empty_groups else {}
    )
    motion_roots: list[tuple[Path, Path]] = []
    discovered_motions = 0

    with tqdm(
        desc="enumerating motion roots",
        unit="root",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(motions=0, refresh=False)
        for group_dir in group_dirs:
            with os.scandir(group_dir) as entries:
                for entry in entries:
                    if entry.name == "motion.npz":
                        groups.setdefault(group_dir.name, {})["."] = _validated_motion_path(entry)
                        discovered_motions += 1
                        progress.update(1)
                    elif entry.is_dir(follow_symlinks=True):
                        motion_roots.append((group_dir, Path(entry.path)))
                        progress.update(1)
                    progress.set_postfix(motions=discovered_motions, refresh=False)

        motion_roots.sort(key=lambda item: (item[0].name, item[1].name))
        cpu_count = available_cpu_count if available_cpu_count is not None else available_cpus()
        scan_workers = min(num_workers, cpu_count, max(1, len(motion_roots)))
        progress.set_description("scanning motion roots", refresh=False)
        progress.reset(total=len(motion_roots))

        def record(result: tuple[str, dict[str, Path]]) -> None:
            nonlocal discovered_motions
            group, clips = result
            if clips:
                groups.setdefault(group, {}).update(clips)
            discovered_motions += len(clips)
            progress.update(1)
            progress.set_postfix(motions=discovered_motions, refresh=False)

        if scan_workers > 1:
            with ThreadPoolExecutor(max_workers=scan_workers) as executor:
                futures = [
                    executor.submit(_scan_motion_root, group_dir, motion_root)
                    for group_dir, motion_root in motion_roots
                ]
                for future in as_completed(futures):
                    record(future.result())
        else:
            for group_dir, motion_root in motion_roots:
                record(_scan_motion_root(group_dir, motion_root))

    return {
        group: {clip: groups[group][clip] for clip in sorted(groups[group])}
        for group in sorted(groups)
    }


def discover_grouped_motion_files(
    input_dir: Path,
    num_workers: int = 1,
    *,
    include_empty_groups: bool = False,
    available_cpu_count: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Return grouped motions keyed by complete relative ``motion.npz`` paths."""
    clips_by_group = discover_motion_files(
        input_dir,
        num_workers=num_workers,
        include_empty_groups=include_empty_groups,
        available_cpu_count=available_cpu_count,
    )
    return {
        group: {
            "motion.npz" if clip == "." else f"{clip}/motion.npz": path
            for clip, path in clips.items()
        }
        for group, clips in clips_by_group.items()
    }
