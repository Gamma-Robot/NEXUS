#!/usr/bin/env python3
"""Visualize one or more robot qpos motions with Viser."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from threading import RLock
from zipfile import BadZipFile

import mujoco

from terrain_motion_adaptation.robots.g1 import G1_CONFIG
from terrain_motion_adaptation.utils.kinematics import contact_motion, set_qpos_frame
from terrain_motion_adaptation.utils.model import (
    combine_model_and_scene,
    validate_qpos_schema,
    qpos_names_from_model,
    reorder_qpos,
)
from terrain_motion_adaptation.utils.motion import load_motion
from terrain_motion_adaptation.utils.paths import discover_motions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "robots" / "unitree_g1" / "g1.xml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay robot motion in a Viser browser UI.")
    parser.add_argument("--input-dir", required=True, type=Path, help="One NPZ or an NPZ tree.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Manifest search boundary; defaults to the filesystem root.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Robot MJCF XML.")
    parser.add_argument(
        "--scene", type=Path, nargs="+", default=None,
        help="Terrain-only scene XMLs; filenames identify corresponding motion groups."
    )
    parser.add_argument(
        "--viz-ee-contact", action="store_true", help="Initially show diagnostic EE contacts."
    )
    parser.add_argument("--port", type=int, default=None, help="Viser server port.")
    parser.add_argument(
        "--headless", action="store_true", help="Validate every motion without starting Viser."
    )
    return parser.parse_args()


class MotionPlayback:
    """Load matching terrain counterparts before replacing the active model/pose."""

    def __init__(
        self, *, model: mujoco.MjModel, data: mujoco.MjData,
        motions: dict[str, Path], manifest_base: Path,
        groups: dict[str, dict[str, Path]] | None = None,
        scenes: dict[str, Path | None] | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.model, self.data = model, data
        self.catalog = groups if groups is not None else {"plane": motions}
        self.scenes = scenes if scenes is not None else {"plane": None}
        self.model_path = model_path
        self.group = next(iter(self.catalog))
        self.motions = self.catalog[self.group]
        self.manifest_base = manifest_base
        self.select(next(iter(self.motions)))

    @property
    def num_frames(self) -> int:
        return len(self.motion.qpos)

    def select(self, label: str, *, group: str | None = None) -> None:
        selected_group = self.group if group is None else group
        if selected_group not in self.catalog or label not in self.catalog[selected_group]:
            raise ValueError(f"No counterpart for {label!r} in terrain group {selected_group!r}")
        model = self.model
        if selected_group != self.group:
            if self.model_path is None:
                raise ValueError("a robot model path is required to switch terrain scenes")
            model = combine_model_and_scene(self.model_path, self.scenes[selected_group])
        motion = load_motion(self.catalog[selected_group][label], base_dir=self.manifest_base)
        qpos = reorder_qpos(motion.qpos, motion.qpos_names, qpos_names_from_model(model))
        data = self.data if model is self.model else mujoco.MjData(model)
        set_qpos_frame(model, data, qpos[0])
        self.model, self.data = model, data
        self.motion, self._model_qpos = motion, qpos
        self.group, self.label, self.frame = selected_group, label, 0
        self.motions = self.catalog[self.group]

    def apply_frame(self, frame: int) -> None:
        if not 0 <= frame < self.num_frames:
            raise IndexError(f"frame {frame} outside [0, {self.num_frames})")
        set_qpos_frame(self.model, self.data, self._model_qpos[frame])
        self.frame = frame


def _scene_groups(
    motions: dict[str, Path], input_path: Path, scene_paths: list[Path] | None,
) -> tuple[dict[str, dict[str, Path]], dict[str, Path | None]]:
    """Associate logical clip groups with centered terrain-only scene XMLs."""
    scenes = {path.stem: path for path in (PROJECT_ROOT / "assets/scenes").glob("*.xml")}
    scenes.update({"plane": None, "flat": None})
    supplied = {} if scene_paths is None else {path.stem: path for path in scene_paths}
    if scene_paths and len(supplied) != len(scene_paths):
        raise ValueError("scene filenames must have distinct group names")
    scenes.update(supplied)
    first_parts = {label.split("/", 1)[0] for label in motions}
    grouped = any(name in scenes for name in first_parts)
    groups: dict[str, dict[str, Path]] = {}
    if grouped:
        for label, path in motions.items():
            group, separator, key = label.partition("/")
            if not separator or group not in scenes:
                raise ValueError(f"no scene XML for motion group {group!r}; supply --scene")
            groups.setdefault(group, {})[key] = path
    else:
        if len(supplied) > 1:
            raise ValueError("multiple scenes require motions organized in matching group folders")
        group = next(iter(supplied), None)
        if group is None:
            # Preserve logical ancestry, including a single file or nested clip selection.
            candidates = (input_path, *input_path.parents)
            group = next((p.name for p in candidates if p.name in scenes), "plane")
        groups[group] = motions
    return groups, {group: scenes[group] for group in groups}


def _catalog(input_path: Path) -> tuple[dict[str, Path], Path]:
    resolved = Path(os.path.abspath(input_path.expanduser()))
    discovery_base = resolved if resolved.is_dir() else resolved.parent
    discovered = discover_motions(resolved, discovery_base)
    return ({motion.relative_path.as_posix(): motion.path for motion in discovered}, resolved)


def _run_headless(model: mujoco.MjModel, motions: dict[str, Path], manifest_base: Path) -> None:
    data = mujoco.MjData(model)
    total_frames = 0
    for label, path in motions.items():
        motion = load_motion(path, base_dir=manifest_base)
        validate_qpos_schema(model, motion.qpos_names)
        for frame in reorder_qpos(motion.qpos, motion.qpos_names, qpos_names_from_model(model)):
            set_qpos_frame(model, data, frame)
        total_frames += len(motion.qpos)
        print(f"validated {label}: {len(motion.qpos)} frames")
    print(f"validated {len(motions)} motion(s), {total_frames} total frames")


def _run_viser(
    model: mujoco.MjModel,
    motions: dict[str, Path],
    manifest_base: Path,
    port: int | None,
    *,
    groups: dict[str, dict[str, Path]] | None = None,
    scenes: dict[str, Path | None] | None = None,
    model_path: Path | None = None,
    viz_ee_contact: bool = False,
) -> None:
    import mjviser
    import viser

    playback = MotionPlayback(
        model=model, data=mujoco.MjData(model), motions=motions, manifest_base=manifest_base,
        groups=groups, scenes=scenes, model_path=model_path,
    )
    server = viser.ViserServer(port=port) if port is not None else viser.ViserServer()
    scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
    lock = RLock()
    paused = False
    contact_data = None
    markers = {}

    with server.gui.add_folder("Motion"):
        terrain_dropdown = server.gui.add_dropdown(
            "Terrain", options=tuple(playback.catalog), initial_value=playback.group,
            disabled=len(playback.catalog) == 1,
        )
        motion_dropdown = server.gui.add_dropdown(
            "Motion", options=list(playback.motions), initial_value=playback.label
        )
        status = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Pause")
        looping = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_dropdown(
            "Speed", options=("0.25x", "0.5x", "1x", "2x", "4x"), initial_value="1x"
        )
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=playback.num_frames - 1, step=1, initial_value=0
        )
    with server.gui.add_folder("Debug"):
        contact_toggle = server.gui.add_checkbox("Contact markers", initial_value=viz_ee_contact)
        server.gui.add_html(
            '<div role="group" aria-label="Contact legend" '
            'style="display:flex;flex-wrap:wrap;gap:8px 16px;font-size:13px;'
            'padding:4px 12px 12px;box-sizing:border-box;line-height:1.5">'
            '<span style="display:inline-flex;align-items:center;gap:6px">'
            '<span aria-hidden="true" style="width:12px;height:12px;border-radius:50%;'
            'background:#00ff00;border:1px solid #777"></span>Contact</span>'
            '<span style="display:inline-flex;align-items:center;gap:6px">'
            '<span aria-hidden="true" style="width:12px;height:12px;border-radius:50%;'
            'background:#ff0000;border:1px solid #777"></span>No contact</span>'
            '</div>'
        )
    group_visibility = scene.geom_groups_visible.copy()

    def create_scene_controls() -> None:
        scene.geom_groups_visible[:] = group_visibility
        scene._sync_visibilities()
        scene.create_scene_gui(camera_distance=4.0)
        group_labels = {0: "Show terrain", 2: "Robot visuals", 3: "Collision geometry"}
        for group_idx in sorted(set(int(value) for value in playback.model.geom_group)):
            if not 0 <= group_idx < len(scene.geom_groups_visible):
                continue
            control = server.gui.add_checkbox(
                group_labels.get(group_idx, f"Geometry group {group_idx}"),
                initial_value=scene.geom_groups_visible[group_idx],
            )

            @control.on_update
            def _(event, group_idx=group_idx) -> None:
                if event.client is not None:
                    with lock:
                        def update_visibility() -> None:
                            group_visibility[group_idx] = bool(event.target.value)
                            scene.geom_groups_visible[group_idx] = group_visibility[group_idx]
                            scene._sync_visibilities()

                        scene._apply_visualization_change(update_visibility)

    scene_folder = server.gui.add_folder("Scene", expand_by_default=False)
    with scene_folder:
        create_scene_controls()

    def status_text() -> str:
        return (
            f"**Current:** `{playback.group}/{playback.label}`\n\n"
            f"{playback.num_frames} frames · {playback.motion.fps:g} FPS"
        )

    def redraw() -> None:
        nonlocal contact_data
        scene.update_from_mjdata(playback.data)
        if contact_toggle.value:
            try:
                if contact_data is None:
                    status.content = "Computing contact markers…"
                    contact_data = contact_motion(
                        playback.model, playback._model_qpos, playback.motion.timestep, G1_CONFIG
                    )
                    status.content = status_text()
                positions, mask = contact_data
                for col, name in enumerate(G1_CONFIG.end_effectors):
                    if name not in markers:
                        # Inherit the same camera-tracking translation as scene geometry.
                        markers[name] = server.scene.add_icosphere(
                            f"{scene.fixed_bodies_frame.name}/motion_contacts/{name}",
                            radius=0.05, subdivisions=1,
                        )
                    markers[name].position = positions[playback.frame, col]
                    markers[name].color = (0, 255, 0) if mask[playback.frame, col] else (255, 0, 0)
            except ValueError as error:
                contact_toggle.value = False
                status.content = f"**Contact debug unavailable:** {error}\n\n" + status_text()
        for marker in markers.values():
            marker.visible = bool(contact_toggle.value)

    next_frame_time = time.perf_counter() + playback.motion.timestep

    def select(group: str, label: str) -> None:
        nonlocal scene, scene_folder, contact_data, next_frame_time
        changed_scene = group != playback.group
        try:
            playback.select(label, group=group)
        except (OSError, ValueError, EOFError, BadZipFile) as error:
            terrain_dropdown.value = playback.group
            motion_dropdown.value = playback.label
            status.content = f"**Selection failed:** {error}\n\n" + status_text()
            return
        if changed_scene:
            tracking = scene.camera_tracking_enabled
            scene_folder.remove()
            server.scene.reset()
            markers.clear()
            scene = mjviser.ViserMujocoScene(server, playback.model, num_envs=1)
            scene.camera_tracking_enabled = tracking
            scene_folder = server.gui.add_folder("Scene", expand_by_default=False)
            with scene_folder:
                create_scene_controls()
        contact_data = None
        terrain_dropdown.value = playback.group
        motion_dropdown.options = tuple(playback.motions)
        motion_dropdown.value = playback.label
        frame_slider.value = 0
        frame_slider.max = playback.num_frames - 1
        status.content = status_text()
        next_frame_time = time.perf_counter() + playback.motion.timestep / float(str(speed.value)[:-1])
        redraw()

    @terrain_dropdown.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                select(str(terrain_dropdown.value), playback.label)

    @motion_dropdown.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                select(playback.group, str(motion_dropdown.value))

    @play_button.on_click
    def _(event) -> None:
        nonlocal paused, next_frame_time
        if event.client is not None:
            with lock:
                paused = not paused
                play_button.label = "Play" if paused else "Pause"
                next_frame_time = (
                    time.perf_counter() + playback.motion.timestep / float(str(speed.value)[:-1])
                )

    @frame_slider.on_update
    def _(event) -> None:
        nonlocal paused
        if event.client is not None:
            with lock:
                paused = True
                play_button.label = "Play"
                playback.apply_frame(int(frame_slider.value))
                redraw()

    @contact_toggle.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                redraw()

    status.content = status_text()
    try:
        with lock:
            redraw()
        while True:
            with lock:
                now = time.perf_counter()
                if not paused and now >= next_frame_time:
                    frame = playback.frame + 1
                    if frame >= playback.num_frames:
                        frame = 0 if looping.value else playback.num_frames - 1
                        if not looping.value:
                            paused = True
                            play_button.label = "Play"
                    playback.apply_frame(frame)
                    frame_slider.value = frame
                    redraw()
                    next_frame_time = now + playback.motion.timestep / float(str(speed.value)[:-1])
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main() -> None:
    args = _parse_args()
    motions, resolved_input = _catalog(args.input_dir)
    if args.base_dir is not None:
        manifest_base = Path(os.path.abspath(args.base_dir.expanduser()))
    else:
        manifest_base = Path(resolved_input.anchor)
    groups, scenes = _scene_groups(motions, resolved_input, args.scene)
    initial_group = next(iter(groups))
    model = combine_model_and_scene(args.model, scenes[initial_group])
    if args.headless:
        for group, clips in groups.items():
            group_model = (
                model if group == initial_group
                else combine_model_and_scene(args.model, scenes[group])
            )
            print(f"terrain group: {group}")
            _run_headless(group_model, clips, manifest_base)
    else:
        _run_viser(
            model, groups[initial_group], manifest_base, args.port,
            groups=groups, scenes=scenes, model_path=args.model,
            viz_ee_contact=args.viz_ee_contact,
        )


if __name__ == "__main__":
    main()
