"""Camera domain-randomization events for the grouped ray-caster camera."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _sample_range(
    value_range: tuple[float, float],
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=dtype).uniform_(*value_range)


def sample_camera_intrinsics(
    nominal_matrix: torch.Tensor,
    num_samples: int,
    *,
    fx_scale: tuple[float, float] = (1.0, 1.0),
    fy_scale: tuple[float, float] = (1.0, 1.0),
    ppx_offset: tuple[float, float] = (0.0, 0.0),
    ppy_offset: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Sample pinhole matrices around one nominal matrix."""
    if nominal_matrix.shape != (3, 3):
        raise ValueError(
            f"nominal_matrix must have shape (3, 3), got {tuple(nominal_matrix.shape)}"
        )
    matrices = nominal_matrix.unsqueeze(0).expand(num_samples, -1, -1).clone()
    device, dtype = matrices.device, matrices.dtype
    matrices[:, 0, 0] *= _sample_range(fx_scale, (num_samples,), device=device, dtype=dtype)
    matrices[:, 1, 1] *= _sample_range(fy_scale, (num_samples,), device=device, dtype=dtype)
    matrices[:, 0, 2] += _sample_range(ppx_offset, (num_samples,), device=device, dtype=dtype)
    matrices[:, 1, 2] += _sample_range(ppy_offset, (num_samples,), device=device, dtype=dtype)
    return matrices


def nominal_camera_intrinsics(camera_cfg) -> torch.Tensor:
    """Build the nominal matrix from the sensor configuration."""
    pattern = camera_cfg.pattern
    if camera_cfg.focal_length is not None and camera_cfg.horizontal_aperture is not None:
        vertical_aperture = camera_cfg.vertical_aperture
        if vertical_aperture is None:
            vertical_aperture = camera_cfg.horizontal_aperture * pattern.height / pattern.width
        fx = pattern.width * camera_cfg.focal_length / camera_cfg.horizontal_aperture
        fy = pattern.height * camera_cfg.focal_length / vertical_aperture
        cx = camera_cfg.horizontal_aperture_offset * fx + pattern.width / 2
        cy = camera_cfg.vertical_aperture_offset * fy + pattern.height / 2
    else:
        half_fovy = torch.tensor(pattern.fovy * torch.pi / 360.0)
        fx = fy = 0.5 * pattern.height / torch.tan(half_fovy)
        cx = pattern.width / 2
        cy = pattern.height / 2
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def sample_camera_offset(
    nominal_position: torch.Tensor,
    nominal_orientation: torch.Tensor,
    num_samples: int,
    *,
    pos_offset_range: tuple[float, float] = (0.0, 0.0),
    rot_offset_deg_range: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local camera offsets and normalized local quaternions."""
    if nominal_position.shape != (3,) or nominal_orientation.shape != (4,):
        raise ValueError("nominal camera offset must be position (3,) and quaternion (4,)")
    positions = nominal_position.unsqueeze(0).expand(num_samples, -1).clone()
    orientations = nominal_orientation.unsqueeze(0).expand(num_samples, -1).clone()
    if pos_offset_range != (0.0, 0.0) or rot_offset_deg_range != (0.0, 0.0):
        device, dtype = positions.device, positions.dtype
        positions += _sample_range(pos_offset_range, (num_samples, 3), device=device, dtype=dtype)
        orientations = orientations / torch.linalg.vector_norm(
            orientations, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        angles = torch.deg2rad(
            _sample_range(rot_offset_deg_range, (num_samples, 3), device=device, dtype=dtype)
        )
        delta = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        orientations = _quat_mul(delta, orientations)
        orientations = orientations / torch.linalg.vector_norm(
            orientations, dim=-1, keepdim=True
        ).clamp_min(1e-8)
    return positions, orientations


def _resolve_env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


def camera_intrinsic(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    sensor_name: str = "depth",
    fx_scale: tuple[float, float] = (0.98, 1.02),
    fy_scale: tuple[float, float] = (0.98, 1.02),
    ppx_offset: tuple[float, float] = (-2.0, 2.0),
    ppy_offset: tuple[float, float] = (-2.0, 2.0),
) -> None:
    """Randomize grouped-camera focal lengths and principal point at reset."""
    env_ids = _resolve_env_ids(env, env_ids)
    sensor = env.scene[sensor_name]
    nominal = nominal_camera_intrinsics(sensor.cfg).to(device=env.device)
    matrices = sample_camera_intrinsics(
        nominal,
        len(env_ids),
        fx_scale=fx_scale,
        fy_scale=fy_scale,
        ppx_offset=ppx_offset,
        ppy_offset=ppy_offset,
    )
    sensor.set_intrinsic_matrices(matrices, env_ids=env_ids)


def camera_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    sensor_name: str = "depth",
    pos_offset_range: tuple[float, float] = (-0.025, 0.025),
) -> None:
    """Randomize grouped-camera local position at reset."""
    env_ids = _resolve_env_ids(env, env_ids)
    sensor = env.scene[sensor_name]
    offset = sensor.cfg.offset
    position = torch.as_tensor(offset.pos, device=env.device, dtype=torch.float32)
    orientation = torch.as_tensor(offset.rot, device=env.device, dtype=torch.float32)
    positions, orientations = sample_camera_offset(
        position,
        orientation,
        len(env_ids),
        pos_offset_range=pos_offset_range,
    )
    sensor.set_offset(positions, orientations, env_ids=env_ids)


def camera_quat(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    sensor_name: str = "depth",
    roll_range: tuple[float, float] = (-torch.pi / 180, torch.pi / 180),
    pitch_range: tuple[float, float] = (-torch.pi / 180, torch.pi / 180),
    yaw_range: tuple[float, float] = (-torch.pi / 180, torch.pi / 180),
) -> None:
    """Randomize grouped-camera local orientation at reset; ranges are radians."""
    env_ids = _resolve_env_ids(env, env_ids)
    sensor = env.scene[sensor_name]
    offset = sensor.cfg.offset
    position = torch.as_tensor(offset.pos, device=env.device, dtype=torch.float32)
    orientation = torch.as_tensor(offset.rot, device=env.device, dtype=torch.float32)
    angles = torch.stack(
        (
            _sample_range(roll_range, (len(env_ids),), device=env.device, dtype=torch.float32),
            _sample_range(pitch_range, (len(env_ids),), device=env.device, dtype=torch.float32),
            _sample_range(yaw_range, (len(env_ids),), device=env.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    _, orientations = sample_camera_offset(
        position,
        orientation,
        len(env_ids),
        rot_offset_deg_range=(0.0, 0.0),
    )
    delta = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    orientations = _quat_mul(delta, orientations)
    orientations = orientations / torch.linalg.vector_norm(
        orientations, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    sensor.set_offset(orientations=orientations, env_ids=env_ids)
