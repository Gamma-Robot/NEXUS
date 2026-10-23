from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from perceptive_bfm.sensors.majority_contact import resolve_contact_indices

from ..utils.ground_query import EEGroundRaycaster, resolve_terrain_geom_groups
from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.viewer import DebugVisualizer

    from perceptive_bfm.sensors.majority_contact import MajorityContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(command: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.squeeze(-1)


def action_rate_l2_waist(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    waist_indexes = asset_cfg.joint_ids
    return torch.sum(
        torch.square(
            env.action_manager.action[:, waist_indexes]
            - env.action_manager.prev_action[:, waist_indexes]
        ),
        dim=1,
    )


def motion_joint_pos_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track reference joint positions with an exponential squared-error reward."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.joint_pos - command.robot_joint_pos)
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_vel_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track reference joint velocities with an exponential squared-error reward."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.square(command.joint_vel - command.robot_joint_vel)
    return torch.exp(-error.mean(-1) / std**2)


def joint_torque_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    soft_factor: float = 0.75,
) -> torch.Tensor:
    """Penalize actuator torques above soft_factor times their force limits."""
    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force  # [B, n_act]
    force_range = env.sim.model.actuator_forcerange  # [B, n_act, 2]
    soft_limit = (
        torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs()) * soft_factor
    ).clamp_min(1e-6)
    violation = (force.abs() / soft_limit - 1.0).clamp_min(0.0)
    return violation.sum(dim=-1)


def _feet_support_from_clearance(
    clearance: torch.Tensor,
    current_contact: torch.Tensor,
    sample_foot_indexes: torch.Tensor,
    clearance_threshold: float,
) -> torch.Tensor:
    """Fraction of sole samples left unsupported across the feet in contact."""
    sample_contact = current_contact[:, sample_foot_indexes]
    unsupported = (clearance > clearance_threshold) & sample_contact
    total_support_samples = sample_contact.sum(dim=-1)
    return unsupported.sum(dim=-1).float() / total_support_samples.clamp_min(1)


def _uniform_sole_sample_offsets(
    model: mujoco.MjModel,
    body_id: int,
    sample_spacing: float,
) -> np.ndarray:
    """Uniform body-local grid clipped to the union of a foot's capsule footprints."""
    if sample_spacing <= 0.0:
        raise ValueError(f"sample_spacing must be positive, got {sample_spacing}")

    segments = []
    radii = []
    sole_heights = []
    capsule_axis = np.array([0.0, 0.0, 1.0])
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue

        axis = np.empty(3)
        mujoco.mju_rotVecQuat(axis, capsule_axis, model.geom_quat[geom_id])
        radius = float(model.geom_size[geom_id, 0])
        half_length = float(model.geom_size[geom_id, 1])
        endpoints = model.geom_pos[geom_id] + np.array([-1.0, 1.0])[:, None] * (half_length * axis)
        segments.append(endpoints[:, :2])
        radii.append(radius)
        sole_heights.append(endpoints[:, 2].min() - radius)

    if not segments:
        raise ValueError(f"No capsule collision geoms found on body id {body_id}")

    segments_xy = np.stack(segments)
    capsule_radii = np.asarray(radii)
    expanded_radii = capsule_radii[:, None, None]
    lower = np.min(segments_xy - expanded_radii, axis=(0, 1))
    upper = np.max(segments_xy + expanded_radii, axis=(0, 1))

    grid_shape = np.maximum(np.ceil((upper - lower) / sample_spacing).astype(int), 1)
    grid_spacing = (upper - lower) / grid_shape
    x_coords = lower[0] + (np.arange(grid_shape[0]) + 0.5) * grid_spacing[0]
    y_coords = lower[1] + (np.arange(grid_shape[1]) + 0.5) * grid_spacing[1]
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    candidates = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)

    segment_starts = segments_xy[:, 0]
    segment_vectors = segments_xy[:, 1] - segment_starts
    segment_length_sq = np.sum(np.square(segment_vectors), axis=-1)
    relative = candidates[:, None, :] - segment_starts[None, :, :]
    projection = np.sum(relative * segment_vectors[None, :, :], axis=-1)
    projection /= np.maximum(segment_length_sq[None, :], np.finfo(float).eps)
    projection = np.clip(projection, 0.0, 1.0)
    closest = segment_starts[None, :, :] + projection[..., None] * segment_vectors[None, :, :]
    distance_sq = np.sum(np.square(candidates[:, None, :] - closest), axis=-1)
    inside_sole = np.any(
        distance_sq <= np.square(capsule_radii)[None, :] + 1e-12,
        axis=-1,
    )
    sole_xy = candidates[inside_sole]
    sole_z = np.full((len(sole_xy), 1), min(sole_heights))
    return np.concatenate((sole_xy, sole_z), axis=-1)


class feet_support:
    """Penalize unsupported sole area while a foot is actually in contact.

    A regular grid with target spacing ``sample_spacing`` is clipped to the union of
    the foot's projected capsule collision footprints (40 points per G1 foot at the
    default 0.02 m). Downward terrain rays measure each point's vertical clearance.
    The returned positive cost is the fraction of samples above
    ``clearance_threshold`` across only the feet whose substep-majority contact state
    is active, so it is always in ``[0, 1]``. Use a negative reward weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        sensor = env.scene[cfg.params["sensor_name"]]
        columns, body_ids, _ = resolve_contact_indices(
            sensor, command.robot, command.cfg.foot_body_names
        )
        self.sensor_columns = torch.tensor(columns, dtype=torch.long, device=env.device)

        model = env.sim.mj_model
        sample_offsets = []
        sample_body_indexes = []
        sample_foot_indexes = []
        sample_spacing = float(cfg.params.get("sample_spacing", 0.02))
        for foot_index, body_index in enumerate(body_ids):
            model_body_id = int(command.robot.indexing.body_ids[body_index].item())
            offsets = _uniform_sole_sample_offsets(model, model_body_id, sample_spacing)
            sample_offsets.append(torch.as_tensor(offsets, dtype=torch.float, device=env.device))
            sample_body_indexes.extend([body_index] * len(offsets))
            sample_foot_indexes.extend([foot_index] * len(offsets))

        self.sample_offsets = torch.cat(sample_offsets, dim=0)
        self.sample_body_indexes = torch.tensor(
            sample_body_indexes, dtype=torch.long, device=env.device
        )
        self.sample_foot_indexes = torch.tensor(
            sample_foot_indexes, dtype=torch.long, device=env.device
        )
        self._ground_z = torch.zeros(env.num_envs, len(sample_body_indexes), device=env.device)
        self._ground_raycaster: EEGroundRaycaster | None = None
        self._sole_points_w = torch.zeros(
            env.num_envs, len(sample_body_indexes), 3, device=env.device
        )
        self._sample_contact = torch.zeros(
            env.num_envs, len(sample_body_indexes), dtype=torch.bool, device=env.device
        )
        self._sample_unsupported = torch.zeros_like(self._sample_contact)
        self._sample_state_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._debug_vis_enabled = True

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._ground_z[idx] = 0.0
        self._sample_state_valid[idx] = False

    def _ensure_ground_raycaster(
        self, env: ManagerBasedRlEnv, command: MotionCommand
    ) -> EEGroundRaycaster:
        if self._ground_raycaster is None:
            self._ground_raycaster = EEGroundRaycaster(
                num_envs=env.num_envs,
                num_rays=len(self.sample_body_indexes),
                device=str(env.device),
                terrain_geom_groups=resolve_terrain_geom_groups(
                    env.sim.mj_model, list(command.robot.indexing.geom_ids)
                ),
            )
        return self._ground_raycaster

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        clearance_threshold: float = 0.03,
        sample_spacing: float = 0.02,
    ) -> torch.Tensor:
        del sample_spacing  # Bound at construction; kept so cfg params round-trip.
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        sensor: MajorityContactSensor = env.scene[sensor_name]

        body_pos_w = command.robot.data.body_link_pos_w[:, self.sample_body_indexes]
        body_quat_w = command.robot.data.body_link_quat_w[:, self.sample_body_indexes]
        offsets = self.sample_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
        sole_points_w = body_pos_w + quat_apply(body_quat_w, offsets)

        raycaster = self._ensure_ground_raycaster(env, command)
        self._ground_z = raycaster.query(
            env.sim.wp_model,
            env.sim.wp_data,
            sole_points_w[..., :2],
            self._ground_z,
        )
        clearance = sole_points_w[..., 2] - self._ground_z
        current_contact = sensor.current_contact[:, self.sensor_columns]
        sample_contact = current_contact[:, self.sample_foot_indexes]
        self._sole_points_w = sole_points_w
        self._sample_contact = sample_contact
        self._sample_unsupported = (clearance > clearance_threshold) & sample_contact
        self._sample_state_valid[:] = True
        return _feet_support_from_clearance(
            clearance,
            current_contact,
            self.sample_foot_indexes,
            clearance_threshold,
        )

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draw support state at sole samples, never at terrain ray hit points."""
        if not self._debug_vis_enabled:
            return

        env_indices = list(visualizer.get_env_indices(len(self._sample_state_valid)))
        env_indices = [
            env_idx for env_idx in env_indices if self._sample_state_valid[env_idx].item()
        ]
        if not env_indices:
            return

        points = self._sole_points_w[env_indices].detach().cpu().numpy()
        contacts = self._sample_contact[env_indices].detach().cpu().numpy()
        unsupported = self._sample_unsupported[env_indices].detach().cpu().numpy()
        point_radius = max(0.004 * float(visualizer.meansize), 0.012)

        for env_points, env_contacts, env_unsupported in zip(
            points, contacts, unsupported, strict=True
        ):
            for point, contact, is_unsupported in zip(
                env_points, env_contacts, env_unsupported, strict=True
            ):
                if is_unsupported:
                    color = (1.0, 0.1, 0.1, 0.9)
                elif contact:
                    color = (0.1, 0.9, 0.1, 0.9)
                else:
                    color = (0.5, 0.5, 0.5, 0.35)
                visualizer.add_sphere(center=point, radius=point_radius, color=color)


_EE_GROUPS = {
    "feet": ("foot_body_names", "feet_standing"),
    "hands": ("hand_body_names", "hands_standing"),
}


def _resolve_ee_group(cfg_params: dict) -> tuple[str, str]:
    ee_group = cfg_params.get("ee_group")
    if ee_group not in _EE_GROUPS:
        raise ValueError(
            f"ee_group must be one of {sorted(_EE_GROUPS)}, got {ee_group!r}. "
            "It selects which end-effector subset this reward term tracks."
        )
    return _EE_GROUPS[ee_group]


class ee_air_time_reference:
    """Score end-effector contact timing against the reference motion for the requested group."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        names_attr, self.schedule_attr = _resolve_ee_group(cfg.params)
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        body_names = getattr(command.cfg, names_attr)
        n_ee = len(body_names)
        self.step_dt = env.step_dt
        sensor = env.scene[cfg.params["sensor_name"]]
        columns, _, _ = resolve_contact_indices(sensor, command.robot, body_names)
        self.sensor_columns = torch.tensor(columns, dtype=torch.long, device=env.device)
        self.reward_time = torch.zeros(env.num_envs, n_ee, device=env.device)
        self.prev_contact = torch.zeros(env.num_envs, n_ee, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.reward_time[idx] = 0.0
        self.prev_contact[idx] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        threshold: float,
        ee_group: str,
    ) -> torch.Tensor:
        del ee_group  # bound at construction; kept so the cfg params round-trip
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        sensor: MajorityContactSensor = env.scene[sensor_name]
        current_contact = sensor.current_contact[:, self.sensor_columns]
        first_contact = current_contact & (~self.prev_contact)
        self.prev_contact = current_contact

        contact_diff = getattr(command, self.schedule_attr) ^ current_contact
        self.reward_time = self.reward_time + torch.where(contact_diff, -self.step_dt, self.step_dt)
        reward = ((self.reward_time - threshold).clamp_max(0.0) * first_contact).sum(dim=-1)
        self.reward_time = self.reward_time * (~current_contact)
        return reward


class _ee_air_time_reference_dense:
    """Shape end-effector clearance relative to the reference and the local ground."""

    names_attr: str
    schedule_attr: str
    ground_attr: str
    target_clearance_attr: str

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        sensor = env.scene[cfg.params["sensor_name"]]
        columns, body_ids, _ = resolve_contact_indices(
            sensor, command.robot, getattr(command.cfg, self.names_attr)
        )
        self.sensor_columns = torch.tensor(columns, dtype=torch.long, device=env.device)
        self.ee_body_indexes = torch.tensor(body_ids, dtype=torch.long, device=env.device)

    def _ee_clearance(self, command: MotionCommand) -> torch.Tensor:
        """Height of each robot EE above the terrain beneath it."""
        command.refresh_robot_ee_ground_z()
        ee_z = command.robot.data.body_link_pos_w[:, self.ee_body_indexes, 2]
        return ee_z - getattr(command, self.ground_attr)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        clearance_std: float,
    ) -> torch.Tensor:
        command = cast(MotionCommand, env.command_manager.get_term(command_name))
        sensor: MajorityContactSensor = env.scene[sensor_name]
        current_contact = sensor.current_contact[:, self.sensor_columns]
        target_contact = getattr(command, self.schedule_attr)

        error = self._ee_clearance(command) - getattr(command, self.target_clearance_attr)
        clearance_penalty = torch.exp(-error.square() / clearance_std**2) - 1.0

        mismatch = current_contact ^ target_contact
        penalty = torch.where(mismatch, torch.full_like(clearance_penalty, -1.0), clearance_penalty)
        return penalty.mean(dim=-1)


class feet_air_time_reference_dense(_ee_air_time_reference_dense):
    """Dense contact-timing + clearance shaping for the feet."""

    names_attr = "foot_body_names"
    schedule_attr = "feet_standing"
    ground_attr = "robot_foot_ground_z"
    target_clearance_attr = "target_foot_clearance"


class hand_air_time_reference_dense(_ee_air_time_reference_dense):
    """Dense contact-timing + clearance shaping for the hands."""

    names_attr = "hand_body_names"
    schedule_attr = "hands_standing"
    ground_attr = "robot_hand_ground_z"
    target_clearance_attr = "target_hand_clearance"
