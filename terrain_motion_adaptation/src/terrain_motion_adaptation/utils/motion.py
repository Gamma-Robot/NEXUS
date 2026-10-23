"""Read and write minimal robot motions: float32 qpos and scalar fps."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from terrain_motion_adaptation.utils.paths import iter_motion_paths

ROOT_QPOS_NAMES = ("root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz")


@dataclass(frozen=True)
class MotionSchema:
    path: Path
    joint_names: tuple[str, ...]
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def qpos_names(self) -> tuple[str, ...]:
        return ROOT_QPOS_NAMES + self.joint_names


@dataclass(frozen=True)
class MotionMetadata:
    path: Path
    joint_names: tuple[str, ...]
    fps: float
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def qpos_names(self) -> tuple[str, ...]:
        return ROOT_QPOS_NAMES + self.joint_names

    @property
    def timestep(self) -> float:
        return 1.0 / self.fps


@dataclass(frozen=True)
class Motion(MotionMetadata):
    qpos: np.ndarray


def find_manifest(motion_path: str | Path, base_dir: str | Path) -> Path:
    """Find the nearest manifest without searching above the explicit boundary."""
    motion = Path(os.path.abspath(Path(motion_path).expanduser()))
    base = Path(os.path.abspath(Path(base_dir).expanduser()))
    if not base.is_dir() or not motion.is_relative_to(base) or motion == base:
        raise ValueError(f"motion must be inside base_dir: {motion} (base_dir={base})")
    directory = motion.parent
    while True:
        candidate = directory / "manifest.json"
        if candidate.is_file():
            return candidate
        if directory == base:
            break
        directory = directory.parent
    raise FileNotFoundError(f"manifest.json not found between {motion.parent} and {base}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read manifest {path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
    ):
        raise ValueError(f"manifest version must be 1: {path}")
    names = manifest.get("joint_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(f"joint_names must be unique, nonempty strings: {path}")
    return manifest


def _validate_fps(fps: Any) -> float:
    value = np.asarray(fps)
    if value.ndim != 0 or value.dtype.kind not in "ifu" or not np.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite positive numeric scalar")
    return float(value)


def validate_qpos(qpos: np.ndarray, width: int) -> None:
    """Validate the storage contract without repairing or normalizing data."""
    if qpos.dtype != np.float32 or qpos.ndim != 2 or qpos.shape[1] != width or width < 7:
        raise ValueError(
            f"qpos must be float32 with shape (T, {width}), got {qpos.shape}/{qpos.dtype}"
        )
    if len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("qpos must contain at least two finite frames")
    norms = np.linalg.norm(qpos[:, 3:7].astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("root quaternion must be unit length within 1e-3 (wxyz)")


def load_motion_metadata(motion_path: str | Path, *, base_dir: str | Path) -> MotionMetadata:
    """Read the manifest and per-file FPS without loading the qpos array."""
    path = Path(os.path.abspath(Path(motion_path).expanduser()))
    manifest_path = find_manifest(path, base_dir)
    manifest = _load_manifest(manifest_path)
    return _read_motion_metadata(path, manifest_path, manifest)


def _read_motion_metadata(path, manifest_path, manifest) -> MotionMetadata:
    with np.load(path, allow_pickle=False) as loaded:
        if len(loaded.files) != 2 or set(loaded.files) != {"qpos", "fps"}:
            raise ValueError(f"motion must contain only qpos and fps: {path}")
        fps = _validate_fps(loaded["fps"])
    return MotionMetadata(path, tuple(manifest["joint_names"]), fps, manifest_path, manifest)


def load_motion_metadata_batch(
    motion_paths: Iterable[str | Path],
    *,
    base_dir: str | Path,
    num_workers: int = 1,
    show_progress: bool = False,
) -> list[MotionMetadata]:
    """Read every NPZ while sharing logical manifest lookups within this call."""
    schemas = load_motion_schemas_batch(
        motion_paths, base_dir=base_dir, num_workers=num_workers, show_progress=show_progress
    )

    def read(schema):
        return _read_motion_metadata(schema.path, schema.manifest_path, schema.manifest)

    return list(_validation_map(
        read, schemas, num_workers=num_workers, show_progress=show_progress,
        desc="reading motion metadata", unit="clip",
    ))


def load_motion_schemas_batch(
    motion_paths: Iterable[str | Path],
    *,
    base_dir: str | Path,
    num_workers: int = 1,
    show_progress: bool = False,
) -> list[MotionSchema]:
    """Read shared schemas for logical motion paths without opening NPZ files."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    base = Path(os.path.abspath(Path(base_dir).expanduser()))
    if not base.is_dir():
        raise ValueError(f"base_dir must be a directory: {base}")
    paths = [Path(os.path.abspath(Path(path).expanduser())) for path in motion_paths]
    directories: dict[Path, None] = {}
    for path in paths:
        if not path.is_relative_to(base) or path == base:
            raise ValueError(f"motion must be inside base_dir: {path} (base_dir={base})")
        directory = path.parent
        while directory not in directories:
            directories[directory] = None
            if directory == base:
                break
            directory = directory.parent

    def checked(check, items, desc, unit="path"):
        return _validation_map(
            check, items, num_workers=num_workers, show_progress=show_progress, desc=desc, unit=unit
        )

    def candidate(directory):
        manifest = directory / "manifest.json"
        return manifest if manifest.is_file() else None

    candidates = dict(zip(directories, checked(candidate, directories, "finding input schemas")))

    @lru_cache(maxsize=None)
    def nearest(directory):
        if candidates[directory] is not None:
            return candidates[directory]
        if directory == base:
            raise FileNotFoundError(f"manifest.json not found within base_dir: {base}")
        return nearest(directory.parent)

    manifest_paths = dict.fromkeys(nearest(path.parent) for path in paths)
    manifests = dict(zip(
        manifest_paths, checked(_load_manifest, manifest_paths, "reading input schemas")
    ))

    return [
        MotionSchema(path, tuple(manifests[nearest(path.parent)]["joint_names"]),
                     nearest(path.parent), manifests[nearest(path.parent)])
        for path in paths
    ]


def load_motion(motion_path: str | Path, *, base_dir: str | Path) -> Motion:
    metadata = load_motion_metadata(motion_path, base_dir=base_dir)
    with np.load(metadata.path, allow_pickle=False) as loaded:
        qpos = loaded["qpos"]
    validate_qpos(qpos, len(metadata.qpos_names))
    return Motion(
        metadata.path,
        metadata.joint_names,
        metadata.fps,
        metadata.manifest_path,
        metadata.manifest,
        qpos,
    )


def write_motion(path: str | Path, qpos: np.ndarray, *, fps: float) -> None:
    """Atomically write uncompressed qpos/fps arrays."""
    values = np.asarray(qpos, dtype=np.float32)
    validate_qpos(values, values.shape[1] if values.ndim == 2 else 0)
    rate = _validate_fps(fps)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez(temporary, qpos=values, fps=np.asarray(rate))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON file atomically without following a destination file link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent, text=True)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validation_map(check, items, *, num_workers, show_progress, desc, unit="path"):
    """Run independent filesystem checks with bounded batches and visible progress."""
    from tqdm import tqdm

    with tqdm(
        total=len(items),
        desc=desc,
        unit=unit,
        ncols=max(1, shutil.get_terminal_size().columns - 1),
        disable=not show_progress,
    ) as progress:
        if num_workers == 1:
            for item in items:
                result = check(item)
                progress.update(1)
                yield result
        else:
            # Bound queued futures independently of dataset size (Python >= 3.10).
            iterator = iter(items)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                while batch := list(islice(iterator, 4 * num_workers)):
                    for result in pool.map(check, batch):
                        progress.update(1)
                        yield result


class OutputManifestValidator:
    """Reuse input path checks across the output groups of one batch run.

    Input files and links must stay unchanged during the run. Each output is
    checked afresh, including immediately before its manifests are written.
    Construct a new validator for each run to refresh all input filesystem state.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path,
        motions: Iterable[MotionMetadata | MotionSchema],
        num_workers: int = 1,
        show_progress: bool = False,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        base = Path(os.path.abspath(Path(base_dir).expanduser()))
        motions = list(motions)

        def checked(check, items, desc):
            return _validation_map(
                check, items, num_workers=num_workers, show_progress=show_progress, desc=desc
            )

        # Deduplicate logical paths only; distinct aliases retain separate checks.
        source_paths = dict.fromkeys(
            path for motion in motions for path in (motion.path, motion.manifest_path)
        )

        def source_identity(path):
            resolved = path.resolve()
            try:
                status = resolved.stat()
            except FileNotFoundError:
                return resolved, None
            return resolved, (status.st_dev, status.st_ino)

        sources = list(checked(source_identity, source_paths, "resolving input paths"))
        source_files = {path for path, _ in sources}
        source_file_ids = {identity for _, identity in sources if identity is not None}
        directories: set[Path] = set()
        for motion in motions:
            directory = motion.path.parent
            while directory.is_relative_to(base) and directory not in directories:
                directories.add(directory)
                directory = directory.parent

        def linked_directory(directory):
            return directory.resolve() if directory.is_symlink() else None

        source_directories = {base.resolve()}
        source_directories.update(
            path
            for path in checked(linked_directory, directories, "checking input links")
            if path is not None
        )
        self.base = base
        self.motions = motions
        self.source_paths = source_paths
        self.source_files = source_files
        self.source_file_ids = source_file_ids
        self.source_directories = source_directories
        self._checked = checked
        self._output_tree = {}
        self._output_destinations: dict[Path, tuple[Path, Path]] = {}
        self._output_file_ids: dict[Path, set[tuple[int, int]]] = {}
        for path in source_paths:
            node = self._output_tree
            for part in path.relative_to(base).parts:
                node = node.setdefault(part, {})

    def validate_file_destination(self, path: str | Path) -> None:
        """Check an additional output file before any manifests are written."""
        destination = Path(path).resolve()
        if destination in self.source_files or any(
            destination.is_relative_to(directory) for directory in self.source_directories
        ):
            raise ValueError(f"input and output paths overlap through a symlink: {destination}")
        if destination in self._output_destinations:
            raise ValueError(f"different outputs share the same output destination: {destination}")
        try:
            status = destination.stat()
        except FileNotFoundError:
            return
        identity = (status.st_dev, status.st_ino)
        if identity in self.source_file_ids:
            raise ValueError(f"input and output paths overlap through a hard link: {destination}")
        if any(identity in identities for identities in self._output_file_ids.values()):
            raise ValueError(f"different outputs share the same output file: {destination}")

    def validate(self, scene_root: str | Path) -> None:
        """Reject input overlap and incompatible output schemas without writing."""
        logical_root = Path(os.path.abspath(Path(scene_root).expanduser()))
        root = Path(scene_root).resolve()
        base, motions = self.base, self.motions
        source_files = self.source_files
        source_directories = self.source_directories
        checked = self._checked
        if any(
            root.is_relative_to(directory) or directory.is_relative_to(root)
            for directory in source_directories
        ) or any(path.is_relative_to(root) for path in source_files):
            raise ValueError(f"input and output paths overlap through a symlink: {root}")

        def check_destination(destination):
            if destination in source_files or destination in source_directories or any(
                parent in source_directories for parent in destination.parents
            ):
                raise ValueError(f"input and output paths overlap through a symlink: {destination}")

        # Scan each relevant existing directory once. Missing branches cannot contain
        # symlinks; keep their inherited schema without probing every planned leaf.
        inherited_schemas: dict[Path, Path | None] = {}
        physical_paths = {root: root}
        existing_file_ids: set[tuple[int, int]] = set()

        def record_destinations():
            @lru_cache(maxsize=None)
            def physical(path):
                if path in physical_paths:
                    return physical_paths[path]
                return physical(path.parent) / path.name

            destinations = {}
            for source in self.source_paths:
                relative = source.relative_to(base)
                destination = physical(root / relative)
                owner = (logical_root, relative)
                previous = destinations.get(destination, self._output_destinations.get(destination))
                if previous is not None and previous != owner:
                    raise ValueError(
                        f"different outputs share the same output destination: "
                        f"{previous[0] / previous[1]}, {logical_root / relative}, {destination}"
                    )
                destinations[destination] = owner
            self._output_destinations = {
                path: owner for path, owner in self._output_destinations.items()
                if owner[0] != logical_root
            }
            self._output_destinations.update(destinations)
            self._output_file_ids[logical_root] = existing_file_ids

        def scan_branch(item):
            logical, physical, tree, inherited = item
            try:
                with os.scandir(physical) as entries:
                    selected = {
                        entry.name: entry for entry in entries
                        if entry.name in tree or entry.name == "manifest.json"
                    }
            except FileNotFoundError:
                return [], {logical: inherited}, logical, {logical: physical}, set()
            except NotADirectoryError as exc:
                raise ValueError(f"output path must be a directory: {logical}") from exc
            manifest = selected.get("manifest.json")
            if manifest is not None and manifest.is_file():
                inherited = logical / "manifest.json"
            schemas = {logical: inherited}
            destinations = {logical: physical}
            file_ids = set()
            children = []
            for name, subtree in tree.items():
                entry = selected.get(name)
                destination = physical / name
                if entry is not None and entry.is_symlink():
                    destination = destination.resolve()
                check_destination(destination)
                if entry is not None and entry.is_file():
                    status = entry.stat()
                    identity = (status.st_dev, status.st_ino)
                    if identity in self.source_file_ids:
                        raise ValueError(
                            f"input and output paths overlap through a hard link: {destination}"
                        )
                    file_ids.add(identity)
                destinations[logical / name] = destination
                if subtree:
                    child = logical / name
                    if entry is None:
                        schemas[child] = inherited
                    else:
                        children.append((child, destination, subtree, inherited))
            return children, schemas, None, destinations, file_ids

        pending = [(root, root, self._output_tree, None)]
        while pending:
            following = []
            for children, schemas, missing, destinations, file_ids in checked(
                scan_branch, pending, "checking output paths"
            ):
                physical_paths.update(destinations)
                existing_file_ids.update(file_ids)
                if missing == root:
                    record_destinations()
                    return
                following.extend(children)
                inherited_schemas.update(schemas)
            pending = following

        @lru_cache(maxsize=None)
        def nearest_output_manifest(directory):
            if directory in inherited_schemas:
                return inherited_schemas[directory]
            return nearest_output_manifest(directory.parent)

        output_directories = dict.fromkeys(
            (root / motion.path.relative_to(base)).parent for motion in motions
        )
        nearest = dict(
            zip(
                output_directories,
                checked(nearest_output_manifest, output_directories, "finding output schemas"),
            )
        )
        schemas: dict[Path, set[tuple[str, ...]]] = {}
        expected_manifests: set[Path] = set()
        for motion in motions:
            output_manifest = root / motion.manifest_path.relative_to(base)
            expected_manifests.add(output_manifest)
            manifests = {output_manifest}
            inherited = nearest[(root / motion.path.relative_to(base)).parent]
            if inherited is not None:
                manifests.add(inherited)
            for path in manifests:
                schemas.setdefault(path, set()).add(motion.joint_names)

        def check_schema(item):
            path, expected = item
            if path.is_file():
                actual = tuple(_load_manifest(path)["joint_names"])
                if any(actual != names for names in expected):
                    raise ValueError(
                        f"existing output manifest joint_names differ: {path}; "
                        "use a separate output directory for a different schema"
                    )

        for _ in checked(check_schema, schemas.items(), "checking output schemas"):
            pass

        def check_missing_schema(output_manifest):
            if output_manifest.exists():
                return
            # A missing schema cannot safely be inferred for files from an older run.
            for existing in iter_motion_paths(output_manifest.parent):
                if not existing.is_file():
                    continue
                try:
                    find_manifest(existing, root)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"existing output motion without a manifest: {existing}; "
                        "use a separate output directory or restore its original manifest"
                    ) from exc

        for _ in checked(check_missing_schema, expected_manifests, "checking missing schemas"):
            pass
        record_destinations()

    def write(self, scene_root: str | Path) -> None:
        """Recheck the output and mirror source schemas in their joint order."""
        self.validate(scene_root)
        root = Path(scene_root).resolve()
        base, motions = self.base, self.motions
        manifests = {motion.manifest_path: motion.manifest for motion in motions}
        for source, manifest in manifests.items():
            write_json_atomic(
                root / source.relative_to(base), {"version": 1, "joint_names": manifest["joint_names"]}
            )


def validate_output_manifests(
    *,
    scene_root: str | Path,
    base_dir: str | Path,
    motions: Iterable[MotionMetadata],
    num_workers: int = 1,
    show_progress: bool = False,
) -> None:
    """Check input paths and output schemas afresh before any output is written."""
    OutputManifestValidator(
        base_dir=base_dir, motions=motions, num_workers=num_workers, show_progress=show_progress
    ).validate(scene_root)


def write_output_manifests(
    *,
    scene_root: str | Path,
    base_dir: str | Path,
    motions: Iterable[MotionMetadata],
    num_workers: int = 1,
    show_progress: bool = False,
) -> None:
    """Recheck all paths and mirror source schemas while retaining joint order."""
    OutputManifestValidator(
        base_dir=base_dir, motions=motions, num_workers=num_workers, show_progress=show_progress
    ).write(scene_root)
