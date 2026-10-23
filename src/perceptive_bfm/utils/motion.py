"""Minimal motion protocol, memmap packing and resident tensor loading.

Only root position, wxyz quaternion and joint positions are packed. Body state
and velocities are reconstructed by the runtime FK helper.
"""

from __future__ import annotations

import json
import os
import math
import os.path as osp
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from tensordict import TensorClass
from tqdm import tqdm

META_FILENAME = "meta_motion.json"
TENSORDICT_DIRNAME = "_tensordict"

_T = TypeVar("_T")
_U = TypeVar("_U")
_SourceProtection = tuple[set[Path], set[Path], set[tuple[int, int]]]
_SchemaCache = dict[tuple[Path, Path, tuple[str, ...]], tuple[list[str], list[int]]]


def _pack_map(
    function: Callable[[_T], _U],
    items: Iterable[_T],
    *,
    total: int,
    num_workers: int,
    desc: str,
) -> Iterator[_U]:
    """Read concurrently in input order with bounded prefetch and visible waits."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    with tqdm(total=total, desc=desc, unit="clip", dynamic_ncols=True) as progress:
        if num_workers == 1:
            for item in items:
                result = function(item)
                progress.update(1)
                yield result
            return
        iterator = iter(items)
        with ThreadPoolExecutor(max_workers=min(num_workers, max(1, total))) as executor:
            pending = deque(
                executor.submit(function, item) for item in islice(iterator, 2 * num_workers)
            )
            while pending:
                future = pending.popleft()
                while True:
                    try:
                        result = future.result(timeout=2.0)
                        break
                    except TimeoutError:
                        if future.done():
                            raise
                        progress.refresh()
                progress.update(1)
                yield result
                for item in islice(iterator, 1):
                    pending.append(executor.submit(function, item))


class MotionMinimalData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor


class MotionData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_quat_w: torch.Tensor
    body_quat_b: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_lin_vel_b: torch.Tensor
    body_ang_vel_w: torch.Tensor
    body_ang_vel_b: torch.Tensor


def is_packed_dir(path: str) -> bool:
    """Return True if ``path`` was produced by :func:`create_motion_dataset_from_path`."""
    return osp.isdir(path) and osp.isfile(osp.join(path, META_FILENAME))


def _collect_motion_files(dataset_root: str) -> list[Path]:
    """Discover only motion.npz, preserving logical paths and following links."""
    from perceptive_bfm.utils.motion_tree import discover_motion_files_in_root

    root = Path(os.path.abspath(Path(dataset_root).expanduser()))
    if not root.is_dir():
        raise ValueError(f"dataset_root must be a directory, got: {dataset_root}")
    files = list(discover_motion_files_in_root(root).values())
    if not files:
        raise ValueError(f"No motion.npz clips found under {dataset_root}")
    return files


def _validate_joint_names(joint_names: Sequence[str]) -> list[str]:
    if isinstance(joint_names, (str, bytes)):
        raise ValueError("joint_names must be a sequence of unique nonempty strings")
    names = list(joint_names)
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("joint_names must be unique nonempty strings")
    return names


def _load_motion(
    path: Path,
    *,
    base_dir: str | Path,
    joint_names: Sequence[str],
    _schema_cache: _SchemaCache | None = None,
) -> tuple[np.ndarray, float]:
    """Validate one clip and map joint columns to the requested model order."""
    path = Path(os.path.abspath(path.expanduser()))
    base = Path(os.path.abspath(Path(base_dir).expanduser()))
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f"motion must be inside base_dir: {path} (base_dir={base})")
    expected = _validate_joint_names(joint_names)
    order = tuple(expected)
    cache = {} if _schema_cache is None else _schema_cache
    directory = path.parent
    visited = []
    while (base, directory, order) not in cache:
        if not visited and not base.is_dir():
            raise ValueError(f"motion must be inside base_dir: {path} (base_dir={base})")
        visited.append(directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"failed to read manifest {manifest_path}: {exc}") from exc
            if (
                not isinstance(manifest, dict)
                or type(manifest.get("version")) is not int
                or manifest["version"] != 1
            ):
                raise ValueError(f"manifest version must be 1: {manifest_path}")
            if not isinstance(manifest.get("joint_names"), list):
                raise ValueError(f"manifest joint_names must be a list: {manifest_path}")
            names = _validate_joint_names(manifest["joint_names"])
            if set(names) != set(expected):
                raise ValueError(
                    f"joint_names mismatch in {manifest_path}: expected {expected}, got {names}"
                )
            columns = list(range(7)) + [7 + names.index(name) for name in expected]
            cache[(base, directory, order)] = (names, columns)
            break
        if directory == base:
            raise FileNotFoundError(f"manifest.json not found between {path.parent} and {base}")
        directory = directory.parent
    names, columns = cache[(base, directory, order)]
    for ancestor in visited:
        cache[(base, ancestor, order)] = (names, columns)
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != 2 or set(data.files) != {"qpos", "fps"}:
            raise ValueError(f"motion must contain only qpos and fps: {path}")
        qpos = data["qpos"]
        rate = data["fps"]
    if rate.ndim != 0 or rate.dtype.kind not in "ifu" or not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"fps must be a finite positive numeric scalar: {path}")
    width = 7 + len(names)
    if qpos.dtype != np.float32 or qpos.ndim != 2 or qpos.shape[1] != width:
        raise ValueError(f"qpos must be float32 with shape (T, {width}): {path}")
    if len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError(f"qpos must contain at least two finite frames: {path}")
    norms = np.linalg.norm(qpos[:, 3:7].astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError(f"root quaternion must be unit length within 1e-3 (wxyz): {path}")
    return qpos[:, columns], float(rate)


def _collect_pack_source_protection(
    base_dir: str | Path, source_files: Mapping[str, Path], *, num_workers: int = 1
) -> _SourceProtection:
    """Inspect source paths once for all groups, caching shared directory metadata."""
    base = Path(os.path.abspath(Path(base_dir).expanduser()))

    @lru_cache(maxsize=None)
    def directory_info(directory: Path) -> tuple[Path, frozenset[Path]]:
        if directory == directory.parent:
            return directory.resolve(), frozenset()
        parent, links = directory_info(directory.parent)
        resolved = parent / directory.name
        if resolved.is_symlink():
            resolved = resolved.resolve()
            if directory.is_relative_to(base):
                links = links | {resolved}
        return resolved, links

    def resolve_file(path: Path) -> Path:
        resolved = directory_info(path.parent)[0] / path.name
        return resolved.resolve() if resolved.is_symlink() else resolved

    @lru_cache(maxsize=None)
    def identity(path: Path) -> tuple[int, int]:
        status = path.stat()
        return status.st_dev, status.st_ino

    @lru_cache(maxsize=None)
    def manifest_at(directory: Path) -> Path | None:
        if not directory.is_relative_to(base):
            return None
        candidate = directory / "manifest.json"
        if candidate.is_file():
            return resolve_file(candidate)
        return None if directory == base else manifest_at(directory.parent)

    def inspect(path: Path):
        path = Path(os.path.abspath(Path(path).expanduser()))
        resolved = resolve_file(path)
        manifest = manifest_at(path.parent)
        files = {resolved} if manifest is None else {resolved, manifest}
        return resolved.parent, directory_info(path.parent)[1], files, {identity(p) for p in files}

    protected = {directory_info(base)[0]}
    source_ancestors: set[Path] = set()
    protected_ids: set[tuple[int, int]] = set()

    def index_ancestors(path: Path) -> None:
        while path not in source_ancestors:
            source_ancestors.add(path)
            if path == path.parent:
                break
            path = path.parent

    index_ancestors(directory_info(base)[0])
    for parent, links, files, identities in _pack_map(
        inspect,
        source_files.values(),
        total=len(source_files),
        num_workers=num_workers,
        desc="checking source paths",
    ):
        protected.add(parent)
        protected.update(links)
        for path in files | links | {parent}:
            index_ancestors(path)
        protected_ids.update(identities)
    return protected, source_ancestors, protected_ids


def _validate_pack_destination(
    mem_path: str | Path,
    *,
    base_dir: str | Path,
    source_files: Mapping[str, Path],
    destinations: dict[Path, Path] | None = None,
    destination_ids: dict[tuple[int, int], Path] | None = None,
    _source_protection: _SourceProtection | None = None,
) -> None:
    """Protect inputs and distinct output fields; share registries across groups."""
    if _source_protection is None:
        _source_protection = _collect_pack_source_protection(base_dir, source_files)
    protected, source_ancestors, protected_ids = _source_protection
    destination = Path(mem_path).expanduser().resolve()
    candidates = [destination]
    if destination.is_dir():
        visited = set()
        for directory, dirs, files in os.walk(destination, followlinks=True):
            stat = Path(directory).stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                dirs[:] = []
                continue
            visited.add(identity)
            candidates.extend((Path(directory) / name).resolve() for name in dirs + files)
    for target in candidates:
        # Check both containment directions without walking the source index.
        if target in source_ancestors or target in protected or any(
            ancestor in protected for ancestor in target.parents
        ):
            raise ValueError(
                f"input and output paths overlap through a source or symlink: {target}"
            )

    destinations = {} if destinations is None else destinations
    destination_ids = {} if destination_ids is None else destination_ids
    relative_targets = [".", META_FILENAME, TENSORDICT_DIRNAME] + [
        f"{TENSORDICT_DIRNAME}/{name}"
        for name in ("meta.json", "root_pos_w.memmap", "root_quat_w.memmap", "joint_pos.memmap")
    ]
    logical_root = Path(os.path.abspath(Path(mem_path).expanduser()))
    for relative in relative_targets:
        owner = logical_root / relative
        target = owner.resolve()
        previous = destinations.get(target)
        if previous is not None and previous != owner:
            raise ValueError(
                f"different packed outputs share the same destination: {previous}, {owner}"
            )
        destinations[target] = owner
        try:
            status = target.stat()
        except FileNotFoundError:
            continue
        identity = (status.st_dev, status.st_ino)
        if identity in protected_ids:
            raise ValueError(f"input and output paths overlap through a hard link: {owner}")
        previous = destination_ids.get(identity)
        if previous is not None and previous != owner:
            raise ValueError(f"different packed outputs share the same file: {previous}, {owner}")
        destination_ids[identity] = owner


def collect_motion_source_paths(dataset_root: str) -> list[str]:
    """Return sorted `.npz` paths relative to ``dataset_root``."""
    root = Path(os.path.abspath(Path(dataset_root).expanduser()))
    return [str(path.relative_to(root)) for path in _collect_motion_files(str(root))]


def create_motion_dataset_from_path(
    dataset_root: str,
    mem_path: str,
    *,
    joint_names: Sequence[str],
    storage_dtype: torch.dtype = torch.float32,
    source_paths: Sequence[str] | None = None,
) -> None:
    """Pack selected clips into root/joint memmaps in the requested joint order.

    An explicit source_paths sequence selects unique logical relative paths.
    Every frame is retained; FPS and manifest validation precede output writes.
    """
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    paths = _collect_motion_files(dataset_root)
    if source_paths is not None:
        requested = list(source_paths)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("source_paths must be non-empty and unique")
        path_by_source = {str(path.relative_to(dataset_root)): path for path in paths}
        missing = sorted(set(requested) - set(path_by_source))
        if missing:
            raise ValueError(
                f"source_paths contains {len(missing)} path(s) not found under "
                f"{dataset_root}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        paths = [path_by_source[source_path] for source_path in sorted(requested)]

    source_files = {str(path.relative_to(dataset_root)): path for path in paths}
    create_motion_dataset_from_files(
        source_files,
        mem_path,
        base_dir=dataset_root,
        joint_names=joint_names,
        storage_dtype=storage_dtype,
    )


def create_motion_dataset_from_files(
    source_files: Mapping[str, Path],
    mem_path: str,
    *,
    base_dir: str | Path,
    joint_names: Sequence[str],
    storage_dtype: torch.dtype = torch.float32,
    num_workers: int = 1,
    _source_protection: _SourceProtection | None = None,
    _schema_cache: _SchemaCache | None = None,
) -> None:
    """Pack discovered files with bounded parallel reads in sorted source order."""
    if not source_files:
        raise ValueError("source_files must contain at least one motion")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if _source_protection is None:
        _source_protection = _collect_pack_source_protection(
            base_dir, source_files, num_workers=num_workers
        )
    _validate_pack_destination(
        mem_path,
        base_dir=base_dir,
        source_files=source_files,
        _source_protection=_source_protection,
    )
    ordered_files = [
        (source_path, Path(source_files[source_path])) for source_path in sorted(source_files)
    ]

    names = _validate_joint_names(joint_names)
    joint_dim = len(names)
    fps: float | None = None
    packed_source_paths: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    root_pos_chunks: list[torch.Tensor] = []
    root_quat_chunks: list[torch.Tensor] = []
    joint_pos_chunks: list[torch.Tensor] = []
    schema_cache = {} if _schema_cache is None else _schema_cache

    def load(item: tuple[str, Path]):
        source_path, path = item
        qpos, clip_fps = _load_motion(
            path, base_dir=base_dir, joint_names=names, _schema_cache=schema_cache
        )
        return source_path, path, qpos, clip_fps

    cursor = 0
    for source_path, path, qpos, clip_fps in _pack_map(
        load,
        ordered_files,
        total=len(ordered_files),
        num_workers=num_workers,
        desc=f"packing {Path(mem_path).name}",
    ):
        if fps is None:
            fps = clip_fps
        elif clip_fps != fps:
            raise ValueError(
                f"fps mismatch in {path}: dataset fps={fps}, clip fps={clip_fps}. "
                "Mixed frame rates are not supported (velocity reconstruction needs "
                "a single dt for the whole packed dataset)."
            )

        length = len(qpos)
        root_pos_w = qpos[:, :3]
        root_quat_w = qpos[:, 3:7]
        clip_joint_pos = qpos[:, 7:]

        packed_source_paths.append(source_path)
        starts.append(cursor)
        ends.append(cursor + length)
        cursor += length
        root_pos_chunks.append(torch.from_numpy(root_pos_w).to(storage_dtype))
        root_quat_chunks.append(torch.from_numpy(root_quat_w).to(storage_dtype))
        joint_pos_chunks.append(torch.from_numpy(clip_joint_pos).to(storage_dtype))

    assert fps is not None  # source_files is non-empty

    with tqdm(
        total=4, desc=f"writing {Path(mem_path).name}", unit="step", dynamic_ncols=True
    ) as progress:
        root_pos_w = torch.cat(root_pos_chunks, dim=0)
        progress.update(1)
        root_quat_w = torch.cat(root_quat_chunks, dim=0)
        progress.update(1)
        joint_pos = torch.cat(joint_pos_chunks, dim=0)
        progress.update(1)
        data = MotionMinimalData(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            joint_pos=joint_pos,
            batch_size=[cursor],
        )
        os.makedirs(mem_path, exist_ok=True)
        data.memmap(osp.join(mem_path, TENSORDICT_DIRNAME))
        progress.update(1)

    meta = {
        "num_motions": len(ordered_files),
        "joint_dim": joint_dim,
        "joint_names": names,
        "fps": fps,
        "total_frames": cursor,
        "starts": starts,
        "ends": ends,
        "source_paths": packed_source_paths,
    }
    with open(osp.join(mem_path, META_FILENAME), "w") as f:
        json.dump(meta, f)


class GlobalMotionDataset:
    """GPU/CPU-resident minimal motion dataset, loaded from a packed directory.

    Minimal fields are small compared to full per-body state, so
    the entire rank shard is loaded fully resident onto
    ``device`` at init -- no per-window disk streaming is needed. The on-disk memmap
    is only used to avoid duplicated host RAM while reading the packed file.
    """

    def __init__(
        self,
        mem_path: str,
        *,
        joint_names: Sequence[str],
        device: str = "cpu",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size and world_size >= 1")
        if not is_packed_dir(mem_path):
            raise ValueError(
                f"`mem_path` must be a directory produced by "
                f"create_motion_dataset_from_path (missing '{META_FILENAME}'), got: "
                f"{mem_path}"
            )
        with open(osp.join(mem_path, META_FILENAME), "r") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("packed metadata must be an object")
        self.joint_names = _validate_joint_names(joint_names)
        if meta.get("joint_names") != self.joint_names:
            raise ValueError(
                "packed joint_names differ from model order; re-pack using the model joint_names"
            )
        self.joint_dim = len(self.joint_names)
        rate = meta.get("fps")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("packed fps must be finite and positive")
        self.fps: float = float(rate)
        self.device = device
        starts_full = meta.get("starts")
        ends_full = meta.get("ends")
        source_paths_full = meta.get("source_paths")
        if not all(
            isinstance(value, list) for value in (starts_full, ends_full, source_paths_full)
        ):
            raise ValueError("packed starts, ends and source_paths must be lists")
        num_motions_total = len(starts_full)
        if (
            not num_motions_total
            or len(ends_full) != num_motions_total
            or len(source_paths_full) != num_motions_total
            or meta.get("num_motions") != num_motions_total
            or meta.get("joint_dim") != self.joint_dim
        ):
            raise ValueError("packed metadata dimensions disagree")
        if (
            any(not isinstance(path, str) or not path for path in source_paths_full)
            or len(set(source_paths_full)) != num_motions_total
        ):
            raise ValueError("packed source_paths must be unique nonempty strings")
        cursor = 0
        for start, end in zip(starts_full, ends_full):
            if type(start) is not int or type(end) is not int or start != cursor or end - start < 2:
                raise ValueError(
                    "packed frame ranges must be contiguous and contain at least two frames"
                )
            cursor = end
        if meta.get("total_frames") != cursor:
            raise ValueError("packed total_frames disagrees with frame ranges")

        shard_motion_ids = list(range(rank, num_motions_total, world_size))
        if not shard_motion_ids:
            raise ValueError(
                f"Rank {rank} has no motions assigned (num_motions={num_motions_total}, "
                f"world_size={world_size})."
            )

        tensor_dir = Path(mem_path) / TENSORDICT_DIRNAME
        try:
            tensor_meta = json.loads((tensor_dir / "meta.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read packed tensor metadata in {tensor_dir}: {exc}") from exc
        if not isinstance(tensor_meta, dict) or tensor_meta.get("shape") != [cursor]:
            raise ValueError("packed tensor shapes disagree with metadata")
        # Writable memory maps can extend truncated files with zeros. Check every
        # field before opening any mapping, including the shapes used by TensorDict.
        for field, width in (("root_pos_w", 3), ("root_quat_w", 4), ("joint_pos", self.joint_dim)):
            entry = tensor_meta.get(field)
            if (
                not isinstance(entry, dict)
                or entry.get("shape") != [cursor, width]
                or entry.get("is_nested", False)
            ):
                raise ValueError(f"packed tensor {field} shape disagrees with metadata")
            dtype_name = entry.get("dtype")
            dtype = (
                getattr(torch, dtype_name.removeprefix("torch."), None)
                if isinstance(dtype_name, str) and dtype_name.startswith("torch.")
                else None
            )
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid packed tensor dtype for {field}: {dtype_name!r}")
            expected_bytes = cursor * width * torch.empty((), dtype=dtype, device="meta").element_size()
            path = tensor_dir / f"{field}.memmap"
            if not path.is_file():
                raise ValueError(f"Missing packed tensor file: {path}")
            actual_bytes = path.stat().st_size
            # Repacking fewer frames can leave unused trailing bytes in an existing file.
            if actual_bytes < expected_bytes:
                raise ValueError(
                    f"Truncated packed tensor file {path}: expected at least {expected_bytes} "
                    f"bytes, found {actual_bytes}; copy or re-pack the complete dataset"
                )

        depot = MotionMinimalData.load_memmap(tensor_dir)
        if (
            tuple(depot.batch_size) != (cursor,)
            or tuple(depot.root_pos_w.shape) != (cursor, 3)
            or tuple(depot.root_quat_w.shape) != (cursor, 4)
            or tuple(depot.joint_pos.shape) != (cursor, self.joint_dim)
        ):
            raise ValueError("packed tensor shapes disagree with metadata")

        local_starts: list[int] = []
        local_ends: list[int] = []
        chunks: list[MotionMinimalData] = []
        cursor = 0
        for m in shard_motion_ids:
            s, e = int(starts_full[m]), int(ends_full[m])
            chunks.append(depot[s:e])
            local_starts.append(cursor)
            cursor += e - s
            local_ends.append(cursor)

        data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        self.data: MotionMinimalData = data.to(device=device, dtype=torch.float32)
        self.starts = torch.tensor(local_starts, dtype=torch.long, device=device)
        self.ends = torch.tensor(local_ends, dtype=torch.long, device=device)
        self.lengths = self.ends - self.starts
        self.source_paths = [source_paths_full[m] for m in shard_motion_ids]

    @property
    def num_motions(self) -> int:
        return int(self.starts.numel())

    @property
    def motion_lengths(self) -> torch.Tensor:
        return self.lengths

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> MotionMinimalData:
        """Fetch a ``(len(motion_ids), frame_offsets.shape[-1])`` window per query.

        ``frame_offsets`` is broadcast against ``motion_ids`` (typically
        ``starts[:, None] + arange(window)[None, :]``, already relative to each
        motion's own frame 0). Out-of-range indices are clamped to the clip's first/
        last valid frame.
        """
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        frame_offsets = frame_offsets.to(device=self.device, dtype=torch.long)
        clip_starts = self.starts[motion_ids]
        clip_ends = self.ends[motion_ids]
        if frame_offsets.ndim == 1:
            idx = clip_starts.unsqueeze(1) + frame_offsets.unsqueeze(0)
        else:
            idx = clip_starts.unsqueeze(-1) + frame_offsets
        idx = torch.max(idx, clip_starts.unsqueeze(-1))
        idx = torch.min(idx, (clip_ends - 1).unsqueeze(-1))
        return self.data[idx]
