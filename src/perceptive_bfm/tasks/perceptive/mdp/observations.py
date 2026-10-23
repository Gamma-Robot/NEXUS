from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, PinholeCameraPatternCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from perceptive_bfm.sensors.majority_contact import resolve_contact_indices
from perceptive_bfm.utils.noise import CropAndResizeCfg
from perceptive_bfm.utils.visualization import (
    debug_visualize_image,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.sensor import CameraSensor, RayCastSensor

    from perceptive_bfm.sensors.grouped_ray_caster import GroupedRayCasterCamera
    from perceptive_bfm.sensors.majority_contact import MajorityContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def _robot_body_ori_quat_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Robot's own body orientation, as a quaternion (not rot6d), relative to its own
    current anchor. Shape (num_envs, B, 4). Factored out of robot_body_ori_b so
    motion_body_ori_b_future can compute a proper quaternion delta instead of
    subtracting rot6d vectors (rot6d is not a vector space -- subtracting two rot6d
    representations is not a valid orientation difference)."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return ori_b


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    ori_b = _robot_body_ori_quat_b(env, command_name)
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_lin_vel_w_exp = command.robot_anchor_lin_vel_w[:, None, :].repeat(1, num_bodies, 1)
    vel_b = quat_apply_inverse(
        robot_anchor_quat_w_exp, command.robot_body_lin_vel_w - robot_anchor_lin_vel_w_exp
    )
    return vel_b.view(env.num_envs, -1)


def robot_body_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_ang_vel_w_exp = command.robot_anchor_ang_vel_w[:, None, :].repeat(1, num_bodies, 1)
    vel_b = quat_apply_inverse(
        robot_anchor_quat_w_exp, command.robot_body_ang_vel_w - robot_anchor_ang_vel_w_exp
    )
    return vel_b.view(env.num_envs, -1)


def motion_base_lin_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    return quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)


def motion_base_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    return quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)


def motion_anchor_pos_b_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    T = command.joint_pos_future.shape[1]
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, T, -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, T, -1)
    pos, _ = subtract_frame_transforms(
        robot_anchor_pos_w_exp.reshape(-1, 3),
        robot_anchor_quat_w_exp.reshape(-1, 4),
        command.anchor_pos_w_future.reshape(-1, 3),
        command.anchor_quat_w_future.reshape(-1, 4),
    )
    if flatten:
        return pos.view(env.num_envs, -1)
    return pos.view(env.num_envs, T, -1)


def motion_anchor_ori_b_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    T = command.joint_pos_future.shape[1]
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, T, -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, T, -1)
    _, ori = subtract_frame_transforms(
        robot_anchor_pos_w_exp.reshape(-1, 3),
        robot_anchor_quat_w_exp.reshape(-1, 4),
        command.obs_anchor_pos_w_future.reshape(-1, 3),
        command.obs_anchor_quat_w_future.reshape(-1, 4),
    )
    mat = matrix_from_quat(ori)
    if flatten:
        return mat[..., :2].reshape(env.num_envs, -1)
    return mat[..., :2].reshape(env.num_envs, T, -1)


def motion_anchor_lin_vel_b_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    T = command.obs_anchor_lin_vel_w_future.shape[1]
    quat = command.robot_anchor_quat_w[:, None, :].expand(-1, T, -1)
    vel_b = quat_apply_inverse(quat, command.obs_anchor_lin_vel_w_future)
    if flatten:
        return vel_b.reshape(env.num_envs, -1)
    return vel_b


def motion_anchor_ang_vel_b_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    T = command.obs_anchor_ang_vel_w_future.shape[1]
    quat = command.robot_anchor_quat_w[:, None, :].expand(-1, T, -1)
    vel_b = quat_apply_inverse(quat, command.obs_anchor_ang_vel_w_future)
    if flatten:
        return vel_b.reshape(env.num_envs, -1)
    return vel_b


def motion_joint_state_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    joint_state_future = torch.cat(
        [command.obs_joint_pos_future, command.obs_joint_vel_future], dim=-1
    )
    if flatten:
        return joint_state_future.view(env.num_envs, -1)
    return joint_state_future


def motion_projected_gravity_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    """Teacher horizon (`priv`). See `teleop_projected_gravity_future` for the
    `policy` sibling."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[command.cfg.entity_name]

    T = command.obs_anchor_quat_w_future.shape[1]
    gravity_w = asset.data.gravity_vec_w[:, None, :].expand(-1, T, -1).reshape(-1, 3)
    grav_b = quat_apply_inverse(command.obs_anchor_quat_w_future.reshape(-1, 4), gravity_w)
    grav_b = grav_b.view(env.num_envs, T, 3)
    if flatten:
        return grav_b.reshape(env.num_envs, -1)
    return grav_b


def teleop_projected_gravity_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene[command.cfg.entity_name]

    T = command.flat_anchor_quat_w_future_teleop.shape[1]
    gravity_w = asset.data.gravity_vec_w[:, None, :].expand(-1, T, -1).reshape(-1, 3)
    grav_b = quat_apply_inverse(command.flat_anchor_quat_w_future_teleop.reshape(-1, 4), gravity_w)
    grav_b = grav_b.view(env.num_envs, T, 3)
    if flatten:
        return grav_b.reshape(env.num_envs, -1)
    return grav_b


def teleop_anchor_z_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    z = command.flat_anchor_pos_w_future_teleop[..., 2:3]
    if flatten:
        return z.reshape(env.num_envs, -1)
    return z


def motion_body_pos_b_future(
    env: ManagerBasedRlEnv,
    command_name: str,
    include_diff: bool = False,
    flatten: bool = False,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    B = len(command.cfg.body_names)
    T = command.obs_body_pos_w_future.shape[1]
    anchor_pos_w_exp = command.obs_anchor_pos_w_future[:, :, None, :].expand(-1, T, B, -1)
    anchor_quat_w_exp = command.obs_anchor_quat_w_future[:, :, None, :].expand(-1, T, B, -1)
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w_exp.reshape(-1, 3),
        anchor_quat_w_exp.reshape(-1, 4),
        command.obs_body_pos_w_future.reshape(-1, 3),
        command.obs_body_quat_w_future.reshape(-1, 4),
    )
    pos_b = pos_b.view(env.num_envs, T, B, 3)
    out = pos_b
    if include_diff:
        actual_pos_b = robot_body_pos_b(env, command_name).view(env.num_envs, 1, B, 3)
        out = torch.cat([pos_b, pos_b - actual_pos_b], dim=-1)
    if flatten:
        return out.reshape(env.num_envs, -1)
    return out


def motion_body_ori_b_future(
    env: ManagerBasedRlEnv,
    command_name: str,
    include_diff: bool = False,
    flatten: bool = False,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    B = len(command.cfg.body_names)
    T = command.obs_body_quat_w_future.shape[1]
    anchor_pos_w_exp = command.obs_anchor_pos_w_future[:, :, None, :].expand(-1, T, B, -1)
    anchor_quat_w_exp = command.obs_anchor_quat_w_future[:, :, None, :].expand(-1, T, B, -1)
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w_exp.reshape(-1, 3),
        anchor_quat_w_exp.reshape(-1, 4),
        command.obs_body_pos_w_future.reshape(-1, 3),
        command.obs_body_quat_w_future.reshape(-1, 4),
    )
    ori_b = ori_b.view(env.num_envs, T, B, 4)
    ori_b6 = matrix_from_quat(ori_b)[..., :2].reshape(env.num_envs, T, B, 6)
    out = ori_b6
    if include_diff:
        actual_quat_b = _robot_body_ori_quat_b(env, command_name)[:, None, :, :].expand(
            -1, T, -1, -1
        )
        diff_quat_b = quat_mul(ori_b, quat_inv(actual_quat_b))
        diff_rot6d_b = matrix_from_quat(diff_quat_b)[..., :2].reshape(env.num_envs, T, B, 6)
        out = torch.cat([ori_b6, diff_rot6d_b], dim=-1)
    if flatten:
        return out.reshape(env.num_envs, -1)
    return out


def elevation_map(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    coords: str = "xyz",
    clip_height: tuple[float, float] | None = None,
    miss_value: float | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Express ray hits in the sensor frame as a terrain grid.

    The sensor frame is yaw-aligned and gravity-aligned when configured that way.
    A hit below the sensor has negative z. Optional height clipping applies only
    to z; miss_value overrides the z channel for rays with negative distance.

    Grid ordering follows GridPatternCfg.generate_rays: meshgrid(x, y,
    indexing="xy"), flattened row-major, then reshaped to (Ny, Nx). coords="z"
    yields (B, 1, Ny, Nx); coords="xyz" yields (B, 3, Ny, Nx) in x/y/z order.
    flatten=True collapses the grid to (B, C * Ny * Nx); otherwise its spatial
    structure is retained. x/y channels contain sensor-relative grid coordinates.

    Args:
        env: The environment.
        sensor_name: RayCastSensor with a GridPatternCfg ray pattern.
        coords: "z" for height only or "xyz" for all sensor-frame coordinates.
        clip_height: Optional (minimum, maximum) z bounds in meters.
        miss_value: Optional z value for missed rays.
        flatten: Whether to collapse channel and spatial dimensions.

    Returns:
        Terrain observations with C equal to one or three, in the layout above.
    """
    if coords not in ("z", "xyz"):
        raise ValueError(f"coords must be 'z' or 'xyz', got {coords!r}")

    sensor: RayCastSensor = env.scene[sensor_name]
    data = sensor.data
    F, N = sensor.num_frames, sensor.num_rays_per_frame
    B = data.distances.shape[0]

    # Hit point relative to its frame, expressed in the sensor frame. mjlab already
    # bakes ray_alignment into frame_quat_w (yaw-only / gravity-aligned for this
    # sensor), so we rotate by it directly -- no extra yaw extraction needed.
    hit = data.hit_pos_w.view(B, F, N, 3)
    rel = hit - data.frame_pos_w.unsqueeze(2)  # (B, F, N, 3)
    quat_exp = data.frame_quat_w.unsqueeze(2).expand(B, F, N, 4).reshape(-1, 4)
    sensor_coords = quat_apply_inverse(quat_exp, rel.reshape(-1, 3)).view(B, F, N, 3)
    sensor_coords = torch.nan_to_num(sensor_coords)

    if miss_value is not None:
        miss_mask = (data.distances < 0).view(B, F, N)
        sensor_coords[..., 2] = torch.where(
            miss_mask, torch.full_like(sensor_coords[..., 2], miss_value), sensor_coords[..., 2]
        )
    if clip_height is not None:
        sensor_coords[..., 2] = torch.clamp(
            sensor_coords[..., 2], min=clip_height[0], max=clip_height[1]
        )

    pattern = sensor.cfg.pattern
    assert isinstance(
        pattern, GridPatternCfg
    ), f"elevation_map requires a GridPatternCfg sensor, got {type(pattern).__name__}"
    res = pattern.resolution
    nx = int(round(pattern.size[0] / res)) + 1  # grid length (x)
    ny = int(round(pattern.size[1] / res)) + 1  # grid width (y)
    assert (
        F * N == ny * nx
    ), f"elevation_map: ray count {F * N} != grid {ny}*{nx}; check pattern size/resolution"
    assert F == 1, f"elevation_map expects a single-frame sensor, got F={F}"

    # (B, N, 3) -> (B, Ny, Nx, 3) -> (B, 3, Ny, Nx)
    grid = sensor_coords.view(B, ny, nx, 3).permute(0, 3, 1, 2)
    out = grid[:, 2:3] if coords == "z" else grid  # (B, 1, Ny, Nx) or (B, 3, Ny, Nx)
    if flatten:
        return out.reshape(B, -1)
    return out


def _depth_debug_vis(
    depth_norm: torch.Tensor,
    debug_vis_mirror: bool,
    scale_up_vis: int,
    window_name: str,
    mirror_dim: int = -1,
) -> None:
    """Tile original and reflected depth using the backend's sagittal image axis."""
    vis = (
        torch.cat([depth_norm, depth_norm.flip(mirror_dim)], dim=-1)
        if debug_vis_mirror else depth_norm
    )
    # (B, C, H, W) -> (C, H, B, W) -> (C * H, B * W) for batched visualization in a single window
    debug_visualize_image(
        vis.permute(1, 2, 0, 3).flatten(0, 1).flatten(1, 2),
        scale_up_vis=scale_up_vis,
        window_name=window_name,
    )


def raycaster_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    debug_vis: bool = False,
    debug_vis_mirror: bool = False,
    scale_up_vis: int = 5,
    window_name: str = "depth_image",
) -> torch.Tensor:
    sensor: GroupedRayCasterCamera = env.scene[sensor_name]
    # depth_data = sensor.data.output["distance_to_image_plane"]  # (B, H, W, 1)
    depth_data = sensor.data.output["distance_to_image_plane_noised"]  # (B, H, W, 1)
    # depth_data = sensor.data.output["distance_to_image_history"]  # (B, T, H, W, 1)
    # depth_data = sensor.data.output["distance_to_image_plane_noised_history"]  # (B, T, H, W, 1)
    if len(depth_data.shape) == 5:
        # (B, T, H, W, 1) -> (B, T, H, W)
        depth_data = depth_data.squeeze(-1)
    else:
        depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)

    depth_norm = (
        depth_data  # hardcoded as normalization is already applied under current configuration
    )

    if debug_vis:
        try:
            _depth_debug_vis(depth_norm, debug_vis_mirror, scale_up_vis, window_name)
        except Exception as e:
            # don't crash the env if visualization fails
            print(f"Depth debug visualization failed: {e}")

    # return depth_norm / 2.0  # NOTE: legacy bug here!
    return depth_norm


def pinhole_raycast_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float = 2.0,
    debug_vis: bool = False,
    debug_vis_mirror: bool = False,
    scale_up_vis: int = 5,
    window_name: str = "depth_image",
) -> torch.Tensor:
    """Convert body-aligned pinhole hits to normalized optical depth (B, 1, 18, 32)."""
    sensor: RayCastSensor = env.scene[sensor_name]
    pattern = sensor.cfg.pattern
    if not isinstance(pattern, PinholeCameraPatternCfg) or sensor.num_frames != 1:
        raise ValueError("pinhole_raycast_depth requires one pinhole camera frame")
    if sensor.cfg.ray_alignment != "base":
        raise ValueError("pinhole_raycast_depth requires base-aligned rays")
    data = sensor.data
    local_hits = quat_apply_inverse(
        data.frame_quat_w.expand(-1, data.distances.shape[1], -1),
        data.hit_pos_w - data.frame_pos_w,
    )
    # Upstream pinhole rays point along local -Z; misses use -1 distance.
    depth = torch.where(data.distances < 0, cutoff_distance, -local_hits[..., 2])
    depth = (depth / cutoff_distance).clamp(0.0, 1.0)
    depth = depth.reshape(-1, pattern.height, pattern.width, 1)
    processing = CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32))
    depth_norm = processing.func(depth, processing, ()).permute(0, 3, 1, 2)

    if debug_vis:
        try:
            _depth_debug_vis(depth_norm, debug_vis_mirror, scale_up_vis, window_name, mirror_dim=-2)
        except Exception as e:
            print(f"Depth debug visualization failed: {e}")
    return depth_norm


def camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float = 2.0,
    min_depth: float = 0.0,  # 0.28
    debug_vis: bool = False,
    debug_vis_mirror: bool = False,
    scale_up_vis: int = 5,
    window_name: str = "depth_image",
) -> torch.Tensor:
    """Depth observation in CNN-compatible format (B, 1, H, W)."""
    sensor: CameraSensor = env.scene[sensor_name]
    depth_data = sensor.data.depth  # (B, H, W, 1)
    assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
    depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
    depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
    depth_norm = torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)

    if debug_vis:
        try:
            _depth_debug_vis(depth_norm, debug_vis_mirror, scale_up_vis, window_name, mirror_dim=-2)
        except Exception as e:
            # don't crash the env if visualization fails
            print(f"Depth debug visualization failed: {e}")

    return depth_norm


def applied_action(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Return the applied joint-position command, including default offsets and randomization."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target


def applied_torque(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Return scalar actuator output with shape (num_envs, num_actuators)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


def motion_anchor_pos_z_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the signed vertical error between the reference and robot anchors."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return (command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]).unsqueeze(-1)


def motion_anchor_ori_gravity_error(
    env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Return anchor orientation error in the gravity direction."""
    asset: Entity = env.scene[asset_cfg.name]
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)
    robot_projected_gravity_b = quat_apply_inverse(
        command.robot_anchor_quat_w, asset.data.gravity_vec_w
    )
    return motion_projected_gravity_b[:, 2:3] - robot_projected_gravity_b[:, 2:3]


class ee_contact_state:
    """Return measured contact states for configured feet and hands in schedule order."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        sensor = env.scene[cfg.params["sensor_name"]]
        columns, _, _ = resolve_contact_indices(sensor, command.robot, command.ee_body_names)
        self.sensor_columns = torch.tensor(columns, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
        del command_name
        sensor: MajorityContactSensor = env.scene[sensor_name]
        # ObservationManager probes every term's output shape once at construction
        # time, before the first physics substep -- the sensor has no data yet then.
        contact = getattr(sensor, "_current_contact", None)
        if contact is None:
            return torch.zeros(env.num_envs, len(self.sensor_columns), device=env.device)
        return contact[:, self.sensor_columns].float()


def motion_ee_contact_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the reference motion contact schedule for configured end effectors."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return command.ee_contact_active.float()


def domain_motor_params_implicit_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = asset.indexing.ctrl_ids
    kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
    kd_raw = env.sim.model.actuator_biasprm[:, ctrl_ids, 2]
    armature = env.sim.model.dof_armature[:, asset.indexing.joint_v_adr]
    return torch.cat([kp, kd_raw, armature], dim=-1)


def domain_body_materials_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids
    friction = env.sim.model.geom_friction[:, geom_ids, 0]
    solref = env.sim.model.geom_solref[:, geom_ids, :]
    return torch.cat([friction, solref.reshape(friction.shape[0], -1)], dim=-1)


def domain_joint_action_offset_obs(env: ManagerBasedRlEnv, action_term_name: str) -> torch.Tensor:
    return env.action_manager.get_term(action_term_name).offset


def teleop_anchor_pos_rel_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = True
) -> torch.Tensor:
    """Return flat-reference future anchor positions for the policy horizon."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    anchor_pos_future = command.flat_anchor_pos_w_future_teleop  # (N, T, 3)
    anchor_quat_future0 = command.flat_anchor_quat_w_future_teleop[:, 0, :]  # (N, 4)
    anchor_pos_future0 = anchor_pos_future[:, 0, :]  # (N, 3)

    pos_rest = anchor_pos_future[:, 1:, :]  # (N, T-1, 3)
    T = pos_rest.shape[1]
    quat0_exp = anchor_quat_future0[:, None, :].expand(-1, T, -1)
    pos_rel = quat_apply_inverse(
        quat0_exp.reshape(-1, 4),
        (pos_rest - anchor_pos_future0[:, None, :]).reshape(-1, 3),
    ).view(env.num_envs, T, 3)

    if flatten:
        return pos_rel.reshape(env.num_envs, -1)
    return pos_rel


def teleop_anchor_ori_b_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    """`policy`-horizon sibling of `motion_anchor_ori_b_future` -- orientation relative
    to the ROBOT's current anchor, teleop (short) horizon."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    T = command.flat_anchor_quat_w_future_teleop.shape[1]
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, T, -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, T, -1)
    _, ori = subtract_frame_transforms(
        robot_anchor_pos_w_exp.reshape(-1, 3),
        robot_anchor_quat_w_exp.reshape(-1, 4),
        command.flat_anchor_pos_w_future_teleop.reshape(-1, 3),
        command.flat_anchor_quat_w_future_teleop.reshape(-1, 4),
    )
    mat = matrix_from_quat(ori)
    if flatten:
        return mat[..., :2].reshape(env.num_envs, -1)
    return mat[..., :2].reshape(env.num_envs, T, -1)


def teleop_joint_pos_future(
    env: ManagerBasedRlEnv, command_name: str, flatten: bool = False
) -> torch.Tensor:
    """`policy`-horizon sibling of `motion_joint_pos_future`-equivalent -- reads
    `command.flat_joint_pos_future_teleop` (short/near-term window)."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    if flatten:
        return command.flat_joint_pos_future_teleop.reshape(env.num_envs, -1)
    return command.flat_joint_pos_future_teleop
