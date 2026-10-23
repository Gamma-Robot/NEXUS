"""Compose left-right transforms for policy, privileged and terrain observations.

The term tables must match the names and order in perceptive_env_cfg.py. Live
width and image-shape checks detect incompatible observation layouts. The grouped
head camera mirrors image width; body-aligned -Z baselines mirror image height.
Elevation mirrors the lateral grid axis without a height sign change.

The augmentation entry point supports observations and actions together,
observations alone, or actions alone, as required by PPO and distillation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import CameraSensorCfg, PinholeCameraPatternCfg, RayCastSensorCfg
from tensordict import TensorDict

from perceptive_bfm.asset_zoo.robots.unitree_g1.g1_constants import g1_symmetry_maps
from perceptive_bfm.tasks.perceptive.mdp.commands import MotionCommand
from perceptive_bfm.tasks.perceptive.perceptive_env_cfg import PROPRIO_HISTORY_LENGTH
from perceptive_bfm.utils.symmetry import (
    SymmetryTransform,
    cartesian_space_symmetry,
    grid_symmetry,
    joint_space_symmetry,
)

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


def _build_group_transforms(env: VecEnv) -> dict[str, SymmetryTransform]:
    """Pure term-by-term composition, no live-width validation. Deliberately factored
    out of compose_group_symmetry so tests can call it against a lightweight stub."""
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    command = cast(MotionCommand, unwrapped.command_manager.get_term("motion"))

    joint_names = list(robot.joint_names)
    body_names = list(command.cfg.body_names)
    # Feet then hands -- must match MotionCommand.ee_body_names, which is the order
    # the ee_contact_state / motion_ee_contact_state observations are emitted in.
    ee_body_names = list(command.cfg.foot_body_names) + list(command.cfg.hand_body_names)
    t_teacher = command.joint_pos_future.shape[1]
    t_teleop = command.joint_pos_future_teleop.shape[1]
    num_geoms = len(robot.indexing.geom_ids)

    joint_map, spatial_map = g1_symmetry_maps()

    def joint_t(repeat: int = 1) -> SymmetryTransform:
        return joint_space_symmetry(joint_names, joint_map).repeat(repeat)

    def joint_perm_only(repeat: int = 1) -> SymmetryTransform:
        base = joint_space_symmetry(joint_names, joint_map)
        return SymmetryTransform(base.perm, torch.ones_like(base.signs)).repeat(repeat)

    def vec3(sign, repeat: int = 1) -> SymmetryTransform:
        return SymmetryTransform(torch.arange(3), list(sign)).repeat(repeat)

    def scalar(repeat: int = 1) -> SymmetryTransform:
        return SymmetryTransform(torch.arange(1), [1.0]).repeat(repeat)

    def rot6d(repeat: int = 1) -> SymmetryTransform:
        return SymmetryTransform(torch.arange(6), [1, -1, -1, 1, 1, -1]).repeat(repeat)

    def cart(names, sign, repeat: int = 1) -> SymmetryTransform:
        return cartesian_space_symmetry(names, spatial_map, sign=sign).repeat(repeat)

    # Term order below must match priv_terms' dict-insertion order in
    # perceptive_env_cfg.py exactly. motion_anchor_pos_b_future stays terrain-
    # sourced (a local delta against the robot's real subterrain, safe -- see
    # observations.py); motion_anchor_z_future does not exist in priv_terms at
    # all (deleted, a global/absolute quantity) so there is no corresponding
    # entry here.
    priv_term_transforms: list[SymmetryTransform] = [
        vec3((1, -1, 1), t_teacher),  # motion_anchor_pos_b_future
        rot6d(t_teacher),  # motion_anchor_ori_b_future
        vec3((1, -1, 1), t_teacher),  # motion_anchor_lin_vel_b_future
        vec3((-1, 1, -1), t_teacher),  # motion_anchor_ang_vel_b_future
        SymmetryTransform.cat([joint_t(), joint_t()]).repeat(
            t_teacher
        ),  # motion_joint_state_future
        vec3((1, -1, 1), t_teacher),  # motion_projected_gravity_future
        cart(body_names, (1, -1, 1)),  # body_pos
        cart(body_names, (1, -1, -1, 1, 1, -1)),  # body_ori
        cart(body_names, (1, -1, 1)),  # body_lin_vel
        cart(body_names, (-1, 1, -1)),  # body_ang_vel
        cart(body_names, (1, -1, 1, 1, -1, 1), t_teacher),  # motion_body_pos_b_future
        cart(  # motion_body_ori_b_future (rot6d+diff)
            body_names, (1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1), t_teacher
        ),
    ]
    priv_term_transforms.extend(
        [
            vec3((1, -1, 1), PROPRIO_HISTORY_LENGTH),  # base_lin_vel_history
            vec3((-1, 1, -1), PROPRIO_HISTORY_LENGTH),  # base_ang_vel_history
            vec3((1, -1, 1), PROPRIO_HISTORY_LENGTH),  # projected_gravity_history
            joint_t(PROPRIO_HISTORY_LENGTH),  # joint_pos_history
            joint_t(PROPRIO_HISTORY_LENGTH),  # joint_vel_history
            joint_t(),  # actions
            joint_t(),  # applied_action
            joint_t(),  # applied_torque
            scalar(),  # motion_anchor_pos_z_error
            scalar(),  # motion_anchor_ori_gravity_error
            cart(ee_body_names, (1,)),  # ee_contact_state
            cart(ee_body_names, (1,)),  # motion_ee_contact_state
        ]
    )
    priv_transform = SymmetryTransform.cat(priv_term_transforms)

    priv_critic_transform = SymmetryTransform.cat(
        [
            SymmetryTransform.cat([joint_perm_only(), joint_perm_only(), joint_perm_only()]),
            SymmetryTransform(torch.arange(3 * num_geoms), torch.ones(3 * num_geoms)),
            joint_t(),  # domain_joint_action_offset
        ]
    )

    # teleop_anchor_pos_rel_future is genuinely zero-width whenever t_teleop == 1 (see
    # tracking's symmetry_terms.py for the full rationale) -- omit rather than crash.
    policy_term_transforms: list[SymmetryTransform] = []
    if t_teleop > 1:
        policy_term_transforms.append(
            vec3((1, -1, 1), t_teleop - 1)
        )  # teleop_anchor_pos_rel_future
    policy_term_transforms.append(rot6d(t_teleop))  # teleop_anchor_ori_b_future
    policy_term_transforms.append(joint_t(t_teleop))  # teleop_joint_pos_future
    policy_term_transforms.append(scalar(t_teleop))  # teleop_anchor_z_future
    policy_term_transforms.extend(
        [
            vec3((1, -1, 1), t_teleop),  # teleop_projected_gravity_future
            vec3((-1, 1, -1), PROPRIO_HISTORY_LENGTH),  # base_ang_vel
            vec3((1, -1, 1), PROPRIO_HISTORY_LENGTH),  # projected_gravity
            joint_t(PROPRIO_HISTORY_LENGTH),  # joint_pos
            joint_t(PROPRIO_HISTORY_LENGTH),  # joint_vel
            joint_t(PROPRIO_HISTORY_LENGTH),  # actions
        ]
    )
    policy_transform = SymmetryTransform.cat(policy_term_transforms)

    action_transform = joint_t()

    height_scan_sensor = unwrapped.scene["height_scan"]
    hs_pattern = height_scan_sensor.cfg.pattern
    hs_res = hs_pattern.resolution
    hs_nx = int(round(hs_pattern.size[0] / hs_res)) + 1  # grid length (x)
    hs_ny = int(round(hs_pattern.size[1] / hs_res)) + 1  # grid width (y)
    # Flat (B, Ny*Nx) z-only grid -- see elevation_terms in perceptive_env_cfg.py.
    elevation_transform = grid_symmetry(nx=hs_nx, ny=hs_ny)
    # height_map_transform = _HeightMapSymmetry(nx=hs_nx, ny=hs_ny)

    transforms = {
        "policy": policy_transform.to(device),
        "priv": priv_transform.to(device),
        "priv_critic": priv_critic_transform.to(device),
        "elevation": elevation_transform.to(device),
        # "height_map": height_map_transform.to(device),
        "__action__": action_transform.to(device),
    }

    active_terms = getattr(unwrapped.observation_manager, "active_terms", None)
    camera_configured = "depth" in unwrapped.scene.sensors and (
        active_terms is None or "camera" in active_terms
    )
    if camera_configured:
        depth_sensor = unwrapped.scene["depth"]
        rendered_camera = isinstance(depth_sensor.cfg, CameraSensorCfg)
        pinhole_baseline = type(depth_sensor.cfg) is RayCastSensorCfg and isinstance(
            depth_sensor.cfg.pattern, PinholeCameraPatternCfg
        )
        if rendered_camera:
            depth_w = depth_sensor.cfg.width
            depth_h = depth_sensor.cfg.height
        else:
            depth_w = depth_sensor.cfg.pattern.width
            depth_h = depth_sensor.cfg.pattern.height
        # Grouped-camera observations use the pipeline's resize target rather
        # than the raw pattern dimensions. Rendered depth retains its native size.
        noise_pipeline = getattr(depth_sensor.cfg, "noise_pipeline", None) or {}
        crop_and_resize = noise_pipeline.get("crop_and_resize")
        if crop_and_resize is not None and crop_and_resize.resize_shape is not None:
            depth_h, depth_w = crop_and_resize.resize_shape
        if pinhole_baseline:
            # The upstream baseline is resized by pinhole_raycast_depth outside
            # the sensor. Grouped-camera subclasses use their own pipeline above.
            depth_h, depth_w = 18, 32
        if rendered_camera or pinhole_baseline:
            # These fixed baselines look along body -Z with image rows along Y.
            # A sagittal reflection reverses rows. Custom extrinsics require a
            # matching transform; the grouped head calibration stays horizontal.
            pixel_perm = torch.arange(depth_h * depth_w).reshape(depth_h, depth_w).flip(0)
            camera_transform = SymmetryTransform(
                pixel_perm.flatten(), torch.ones(depth_h * depth_w), dim=-2
            )
        else:
            camera_transform = SymmetryTransform(
                torch.arange(depth_w).flip(0), torch.ones(depth_w), dim=-1
            )
        camera_transform._expected_shape = (1, depth_h, depth_w)
        transforms["camera"] = camera_transform.to(device)

    return transforms


def _probe_group_widths(env: VecEnv) -> dict[str, int]:
    """Flat (batch-excluded) widths for groups backed by a plain `SymmetryTransform`
    (all groups currently returned by `_build_group_transforms` qualify -- filtered by
    `hasattr(t, "perm")` for forward-compat with any future group backed by a
    non-`SymmetryTransform` `torch.nn.Module`, e.g. the now-commented-out
    `_HeightMapSymmetry`)."""
    return {
        k: int(t.perm.numel())
        for k, t in _build_group_transforms(env).items()
        if hasattr(t, "perm")
    }


def compose_group_symmetry(env: VecEnv) -> dict[str, SymmetryTransform]:
    transforms = _build_group_transforms(env)
    unwrapped = env.unwrapped
    active_terms = getattr(unwrapped.observation_manager, "active_terms", None)

    for name in ("policy", "priv", "priv_critic", "elevation"):
        # Plain privileged teachers intentionally omit the deployable policy group.
        # Only validate groups configured on this environment; absent groups cannot
        # appear in the rollout observations that augment_observations mirrors.
        if active_terms is not None and name not in active_terms:
            continue
        actual = int(unwrapped.observation_manager.compute_group(name).shape[-1])
        expected = int(transforms[name].perm.numel())
        if actual != expected:
            raise ValueError(
                f"Symmetry transform width mismatch for group '{name}': expected "
                f"{expected} from compose_group_symmetry's term-by-term composition, "
                f"got {actual} from the live observation group. perceptive_env_cfg.py's "
                f"{name}_terms dict was likely edited without updating "
                "symmetry_terms.py's matching term table."
            )

    if "camera" in transforms:
        # Camera tensors are (B, C, H, W), or (B, T, C, H, W) with history.
        # Validate the trailing image shape while allowing leading history dimensions.
        live_camera_shape = tuple(unwrapped.observation_manager.compute_group("camera").shape[1:])
        expected_camera_shape = transforms["camera"]._expected_shape
        if live_camera_shape[-len(expected_camera_shape) :] != expected_camera_shape:
            raise ValueError(
                "Symmetry transform shape mismatch for group 'camera': expected "
                f"trailing shape {expected_camera_shape}, got {live_camera_shape}."
            )

    return transforms


_TRANSFORM_CACHE: dict[int, dict[str, SymmetryTransform]] = {}


def _get_transforms(env: VecEnv) -> dict[str, SymmetryTransform]:
    key = id(env)
    cached = _TRANSFORM_CACHE.get(key)
    if cached is not None:
        return cached
    transforms = compose_group_symmetry(env)
    _TRANSFORM_CACHE[key] = transforms
    return transforms


def augment_observations(
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    env: VecEnv,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    transforms = _get_transforms(env)

    mirrored_obs = None
    if obs is not None:
        mirrored = {}
        for key, tensor in obs.items():
            transform = transforms.get(key)
            mirrored_tensor = transform(tensor) if transform is not None else tensor
            mirrored[key] = torch.cat([tensor, mirrored_tensor], dim=0)
        mirrored_obs = TensorDict(mirrored, batch_size=[obs.batch_size[0] * 2])

    mirrored_actions = None
    if actions is not None:
        action_transform = transforms["__action__"]
        mirrored_actions = torch.cat([actions, action_transform(actions)], dim=0)

    return mirrored_obs, mirrored_actions
