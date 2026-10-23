"""Contact-keyframe IK and displacement-smoothing pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from terrain_motion_adaptation.utils.kinematics import fk_motion
from terrain_motion_adaptation.utils.model import (
    validate_qpos_schema,
    validate_adaptation_model,
    qpos_names_from_model,
    reorder_qpos,
)

from terrain_motion_adaptation.optimization.contact import EEContactCfg, detect_ee_contact
from terrain_motion_adaptation.optimization.keyframe_ik import (
    KeyframeIKCfg,
    solve_ik_for_clip,
)
from terrain_motion_adaptation.optimization.smoothing import infill_dense_trajectory
from terrain_motion_adaptation.robots.config import RobotConfig


@dataclass(frozen=True)
class MotionAdaptationConfig:
    contact: EEContactCfg | dict[str, EEContactCfg] = EEContactCfg()
    keyframe_ik: KeyframeIKCfg = KeyframeIKCfg()


class NoActiveKeyframesError(ValueError):
    """The flat motion contains no contact frame that can anchor terrain IK."""


def terrain_adapt_qpos(
    *,
    model: mujoco.MjModel,
    robot: RobotConfig,
    qpos: np.ndarray,
    qpos_names: Sequence[str],
    timestep: float,
    config: MotionAdaptationConfig = MotionAdaptationConfig(),
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Adapt a trajectory using contact-keyframe IK and displacement smoothing."""
    validate_qpos_schema(model, qpos_names)
    validate_adaptation_model(model, robot)
    model_names = qpos_names_from_model(model)
    source_qpos = reorder_qpos(np.asarray(qpos, dtype=np.float64), qpos_names, model_names)
    source = fk_motion(model, source_qpos, robot.contact_bodies, timestep)
    contact_mask = detect_ee_contact(
        source.body_pos_w,
        source.body_lin_vel_w,
        list(robot.contact_bodies),
        list(robot.end_effectors),
        config.contact,
    )
    if not contact_mask.any():
        raise NoActiveKeyframesError("source motion has no active end-effector keyframes")

    sparse = solve_ik_for_clip(
        model=model,
        dx=dx,
        dy=dy,
        dyaw=0.0,
        joint_names=list(model_names[7:]),
        joint_pos_seq=source.joint_pos,
        root_pos_w_seq=source.root_pos_w,
        root_quat_w_seq=source.root_quat_w,
        contact_mask=contact_mask,
        ee_names=list(robot.end_effectors),
        chain_specs=robot.chain_specs,
        frame_selection="sparse",
        cfg=config.keyframe_ik,
    )
    dense = infill_dense_trajectory(
        sparse=sparse,
        source_joint_pos=source.joint_pos,
        source_root_pos_w=source.root_pos_w,
        offset_xy=np.array([dx, dy]),
        num_frames=len(source_qpos),
    )

    output = source_qpos.copy()
    output[:, :3] = dense.root_pos_w
    output[:, 7:] = dense.joint_pos
    if output.shape != source_qpos.shape or not np.isfinite(output).all():
        raise ValueError("terrain optimization produced an invalid qpos trajectory")
    output[:, :2] -= np.array([dx, dy])
    output = output.astype(np.float32)
    limited = model.jnt_limited.astype(bool) & (model.jnt_qposadr >= 7)
    addresses = model.jnt_qposadr[limited]
    limits = model.jnt_range[limited]
    lower, upper = limits.astype(np.float32).T
    # Keep float32 bounds inside the model limits even when rounding points outward.
    lower = np.where(
        lower.astype(np.float64) < limits[:, 0], np.nextafter(lower, np.float32(np.inf)), lower
    )
    upper = np.where(
        upper.astype(np.float64) > limits[:, 1], np.nextafter(upper, np.float32(-np.inf)), upper
    )
    output[:, addresses] = np.clip(output[:, addresses], lower, upper)
    return reorder_qpos(output, model_names, qpos_names)
