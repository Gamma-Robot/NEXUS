"""Sensor configurations for perceptive task variants."""

from .camera import (
    DepthBackend,
    DepthCameraProfile,
    depth_observation_func_name,
    make_depth_sensor_cfg,
    make_grouped_raycaster_camera_cfg,
    make_head_grouped_raycaster_camera_cfg,
    make_pinhole_raycast_camera_cfg,
    make_render_camera_cfg,
)

__all__ = [
    "DepthBackend",
    "DepthCameraProfile",
    "depth_observation_func_name",
    "make_depth_sensor_cfg",
    "make_grouped_raycaster_camera_cfg",
    "make_head_grouped_raycaster_camera_cfg",
    "make_pinhole_raycast_camera_cfg",
    "make_render_camera_cfg",
]
