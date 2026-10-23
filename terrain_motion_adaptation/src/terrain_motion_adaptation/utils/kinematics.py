"""Small MuJoCo FK helpers shared by optimization and visualization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from terrain_motion_adaptation.robots.config import RobotConfig


@dataclass(frozen=True)
class FKMotion:
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray


def set_qpos_frame(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    """Apply one kinematic pose and update all derived MuJoCo state."""
    frame = np.asarray(qpos, dtype=np.float64)
    if frame.shape != (model.nq,):
        raise ValueError(f"qpos frame must have shape ({model.nq},), got {frame.shape}")
    data.qpos[:] = frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def fk_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_names: Sequence[str],
    timestep: float,
) -> FKMotion:
    """Expand qpos frames to selected world-space body poses and velocities."""
    frames = np.asarray(qpos, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq}), got {frames.shape}")
    if frames.shape[0] < 2:
        raise ValueError("qpos must contain at least two frames")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError(f"timestep must be finite and positive, got {timestep}")
    try:
        body_ids = [model.body(name).id for name in body_names]
    except KeyError as exc:
        raise ValueError(f"body not found in robot model: {exc}") from exc

    body_pos = np.empty((len(frames), len(body_ids), 3), dtype=np.float64)
    body_quat = np.empty((len(frames), len(body_ids), 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame_index, frame in enumerate(frames):
        set_qpos_frame(model, data, frame)
        body_pos[frame_index] = data.xpos[body_ids]
        body_quat[frame_index] = data.xquat[body_ids]

    return FKMotion(
        root_pos_w=frames[:, :3].copy(),
        root_quat_w=frames[:, 3:7].copy(),
        joint_pos=frames[:, 7:].copy(),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.gradient(body_pos, timestep, axis=0),
    )


def contact_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    timestep: float,
    robot: RobotConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute EE positions and diagnostic contacts relative to the scene terrain.

    Uses the adaptation detector and its default thresholds. Velocities come
    from FK gradients, not stored annotations. A missing ground hit cannot be
    contact. Robot geometry must be outside terrain raycast group zero.
    """
    from terrain_motion_adaptation.optimization.contact import detect_ee_contact
    from terrain_motion_adaptation.optimization.keyframe_ik import query_terrain_height_normal

    try:
        root_id = model.body(robot.root_body).id
    except KeyError as exc:
        raise ValueError(f"contact debug requires body {robot.root_body!r}") from exc
    robot_geoms = model.body_rootid[model.geom_bodyid] == model.body_rootid[root_id]
    if np.any(robot_geoms & (model.geom_group == 0)):
        raise ValueError("contact debug requires robot geometry outside terrain group 0")
    fk = fk_motion(model, qpos, robot.contact_bodies, timestep)
    positions = fk.body_pos_w[:, 1:]
    ground_z = np.full(positions.shape[:2], -np.inf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for frame, points in enumerate(positions):
        for col, (x, y, _) in enumerate(points):
            hit = query_terrain_height_normal(model, data, x, y)
            if hit is not None:
                ground_z[frame, col] = hit[0]
    mask = detect_ee_contact(
        fk.body_pos_w, fk.body_lin_vel_w, list(robot.contact_bodies),
        list(robot.end_effectors), ground_z=ground_z,
    )
    return positions, mask
