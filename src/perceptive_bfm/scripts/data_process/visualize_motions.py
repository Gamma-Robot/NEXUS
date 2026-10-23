"""Replay grouped minimal motions on the formal G1 PyramidTerrain scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import RLock
import time
from zipfile import BadZipFile

import mujoco
import numpy as np

from perceptive_bfm.terrains.config import build_terrain_model
from perceptive_bfm.utils.motion_tree import discover_grouped_motion_files
from terrain_motion_adaptation.robots.g1 import G1_CONFIG
from terrain_motion_adaptation.utils.kinematics import contact_motion, set_qpos_frame
from terrain_motion_adaptation.utils.model import qpos_names_from_model, reorder_qpos
from terrain_motion_adaptation.utils.motion import load_motion


class GroupedMotionPlayback:
    """Apply named terrain counterparts without changing stored motion coordinates."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        group_origins: dict[str, np.ndarray],
        input_dir: Path,
        group: str | None = None,
        motion_key: str | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.input_dir = Path(os.path.abspath(input_dir.expanduser()))
        self.catalog = discover_grouped_motion_files(self.input_dir)
        unknown = set(self.catalog) - set(group_origins)
        if unknown:
            raise ValueError(f"Unknown terrain group(s): {sorted(unknown)}")
        if not self.catalog:
            raise ValueError(f"No grouped motion.npz clips found under {self.input_dir}")
        self.group_origins = group_origins
        self.groups = tuple(name for name in group_origins if name in self.catalog)
        initial_group = group if group is not None else self.groups[0]
        if initial_group not in self.catalog:
            raise ValueError(f"No motions for terrain group {initial_group!r}")
        initial_key = (
            motion_key if motion_key is not None else next(iter(self.catalog[initial_group]))
        )
        self.select(initial_group, initial_key)

    @property
    def motion_keys(self) -> tuple[str, ...]:
        return tuple(self.catalog[self.group])

    def select(self, group: str, motion_key: str | None = None) -> None:
        """Load and validate before replacing the current pose or selection."""
        if group not in self.catalog:
            raise ValueError(f"No motions for terrain group {group!r}")
        key = self.motion_key if motion_key is None else motion_key
        path = self.catalog[group].get(key)
        if path is None:
            raise ValueError(f"No counterpart for {key!r} in terrain group {group!r}")
        motion = load_motion(path, base_dir=self.input_dir)
        qpos = reorder_qpos(motion.qpos, motion.qpos_names, qpos_names_from_model(self.model))
        qpos = qpos.astype(np.float64, copy=True)
        qpos[:, :2] += self.group_origins[group][:2]
        self.group, self.motion_key = group, key
        self.motion, self._qpos = motion, qpos
        self.apply_frame(0)

    def apply_frame(self, frame: int) -> None:
        if not 0 <= frame < len(self._qpos):
            raise IndexError(f"frame {frame} outside [0, {len(self._qpos)})")
        set_qpos_frame(self.model, self.data, self._qpos[frame])
        self.frame = frame


def run_headless(playback: GroupedMotionPlayback) -> None:
    """Apply every stored frame to the model without starting a browser server."""
    frames = 0
    clips = 0
    for group in playback.groups:
        for key in playback.catalog[group]:
            playback.select(group, key)
            for frame in range(len(playback.motion.qpos)):
                playback.apply_frame(frame)
            frames += len(playback.motion.qpos)
            clips += 1
            print(f"validated {group}/{key}: {len(playback.motion.qpos)} frames")
    print(f"validated {clips} motion(s), {frames} total frames")


def run_viser(
    playback: GroupedMotionPlayback, port: int, *, viz_ee_contact: bool = False
) -> None:
    """Display the generated terrain grid with paired motion and playback controls."""
    import mjviser
    import viser

    server = viser.ViserServer(port=port, label="Motion replay")
    scene = mjviser.ViserMujocoScene(server, playback.model, num_envs=1)
    scene.update_from_mjdata(playback.data)
    lock = RLock()
    paused = False
    contact_data = None
    markers = {}

    with server.gui.add_folder("Motion"):
        group_dropdown = server.gui.add_dropdown(
            "Terrain", options=playback.groups, initial_value=playback.group
        )
        motion_dropdown = server.gui.add_dropdown(
            "Motion", options=playback.motion_keys, initial_value=playback.motion_key
        )
        status = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Pause")
        looping = server.gui.add_checkbox("Loop", initial_value=True)
        speed = server.gui.add_dropdown(
            "Speed", options=("0.25x", "0.5x", "1x", "2x", "4x"), initial_value="1x"
        )
        slider = server.gui.add_slider(
            "Frame", min=0, max=len(playback.motion.qpos) - 1, step=1, initial_value=0
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
    with server.gui.add_folder("Scene", expand_by_default=False):
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
                            scene.geom_groups_visible[group_idx] = bool(event.target.value)
                            scene._sync_visibilities()

                        scene._apply_visualization_change(update_visibility)

    def status_text() -> str:
        missing = [
            group for group in playback.groups if playback.motion_key not in playback.catalog[group]
        ]
        text = (
            f"**Current:** `{playback.group}/{playback.motion_key}`\n\n"
            f"{len(playback.motion.qpos)} frames · {playback.motion.fps:g} FPS"
        )
        if missing:
            text += "\n\n**Missing counterparts:** " + ", ".join(missing)
        return text

    def redraw() -> None:
        nonlocal contact_data
        scene.update_from_mjdata(playback.data)
        if contact_toggle.value:
            try:
                if contact_data is None:
                    status.content = "Computing contact markers…"
                    contact_data = contact_motion(
                        playback.model, playback._qpos, playback.motion.timestep, G1_CONFIG
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

    def refresh() -> None:
        nonlocal contact_data, next_frame_time
        contact_data = None
        group_dropdown.value = playback.group
        motion_dropdown.options = playback.motion_keys
        motion_dropdown.value = playback.motion_key
        slider.value = 0
        slider.max = len(playback.motion.qpos) - 1
        status.content = status_text()
        next_frame_time = time.perf_counter() + playback.motion.timestep / float(str(speed.value)[:-1])
        redraw()

    def select(group: str, key: str | None = None) -> None:
        try:
            playback.select(group, key)
        except (OSError, ValueError, EOFError, BadZipFile) as error:
            group_dropdown.value = playback.group
            motion_dropdown.value = playback.motion_key
            status.content = f"**Selection failed:** {error}\n\n" + status_text()
        else:
            refresh()

    @group_dropdown.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                select(str(group_dropdown.value))

    @motion_dropdown.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                select(playback.group, str(motion_dropdown.value))

    @slider.on_update
    def _(event) -> None:
        nonlocal paused
        if event.client is not None:
            with lock:
                paused = True
                play_button.label = "Play"
                playback.apply_frame(int(slider.value))
                redraw()

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

    @contact_toggle.on_update
    def _(event) -> None:
        if event.client is not None:
            with lock:
                redraw()

    status.content = status_text()
    next_frame_time = time.perf_counter() + playback.motion.timestep
    try:
        with lock:
            redraw()
        while True:
            with lock:
                now = time.perf_counter()
                if not paused and now >= next_frame_time:
                    frame = playback.frame + 1
                    if frame >= len(playback.motion.qpos):
                        frame = 0 if looping.value else len(playback.motion.qpos) - 1
                        if not looping.value:
                            paused = True
                            play_button.label = "Play"
                    playback.apply_frame(frame)
                    slider.value = frame
                    redraw()
                    next_frame_time = now + playback.motion.timestep / float(str(speed.value)[:-1])
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Grouped minimal-motion dataset root."
    )
    parser.add_argument("--terrain", choices=("pyramid",), default="pyramid")
    parser.add_argument(
        "--group",
        default=None,
        help="Initial terrain group; defaults to the first available group.",
    )
    parser.add_argument(
        "--motion", default=None, help="Initial group-relative path, including motion.npz."
    )
    parser.add_argument(
        "--viz-ee-contact", action="store_true", help="Initially show diagnostic EE contacts."
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate all groups and frames without starting Viser.",
    )
    args = parser.parse_args()
    model, origins, names = build_terrain_model(args.terrain)
    group_origins = {
        "flat" if name == "flat" else f"{name}_lv0": origins[0, col]
        for col, name in enumerate(names)
    }
    playback = GroupedMotionPlayback(
        model=model,
        group_origins=group_origins,
        input_dir=args.input_dir,
        group=args.group,
        motion_key=args.motion,
    )
    if args.headless:
        run_headless(playback)
    else:
        run_viser(playback, args.port, viz_ee_contact=args.viz_ee_contact)


if __name__ == "__main__":
    main()
