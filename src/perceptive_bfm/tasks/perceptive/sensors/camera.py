"""Depth-sensor configurations shared by perceptive task variants."""

from __future__ import annotations

import math
from enum import Enum

from mjlab.sensor import CameraSensorCfg, ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg

from perceptive_bfm.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from perceptive_bfm.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    DepthSkyArtifactNoiseCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)


class DepthBackend(str, Enum):
    """Available depth-sensor implementations for the perceptive task."""

    GROUPED_RAYCASTER = "grouped_raycaster"
    PINHOLE_RAYCAST = "pinhole_raycast"
    RENDER = "render"


class DepthCameraProfile(str, Enum):
    """Available grouped-raycaster calibrations for the perceptive task."""

    HEAD = "head"


def _pinhole_pattern() -> PinholeCameraPatternCfg:
    return PinholeCameraPatternCfg(
        width=48,
        height=27,
        fovy=58.0,
    )


def _noise_pipeline(noise_enabled: bool = True):
    """Build the shared depth-processing pipeline for every camera profile."""
    pipeline = {}
    if noise_enabled:
        pipeline.update(
            {
                "depth_contour_noise": DepthContourNoiseCfg(
                    contour_threshold=1.8,  # in [m]
                    maxpool_kernel_size=1,
                ),
                "depth_artifact_noise": DepthArtifactNoiseCfg(),
                "reflection_artifact_noise": DepthArtifactNoiseCfg(noise_value=30.0),
                "stereo_noise": RangeBasedGaussianNoiseCfg(
                    max_value=1.2,
                    min_value=0.12,
                    noise_std=0.02,
                ),
                "sky_artifact_noise": DepthSkyArtifactNoiseCfg(),
                "gaussian_blur_noise": GaussianBlurNoiseCfg(
                    kernel_size=3,
                    sigma=0.5,
                ),
                "stereo_too_close_noise": StereoTooCloseNoiseCfg(),
            }
        )
    pipeline.update(
        {
            # These two processing stages also run on the onboard device.
            "normalize": DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
            "crop_and_resize": CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(18, 32)),
        }
    )
    return pipeline


def make_head_grouped_raycaster_camera_cfg(
    camera_noise_enabled: bool = True,
) -> NoisyGroupedRayCasterCameraCfg:
    """Build the existing head-mounted raycaster camera."""
    return NoisyGroupedRayCasterCameraCfg(
        name="depth",
        frame=ObjRef(type="body", name="", entity="robot"),
        pattern=_pinhole_pattern(),
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(87) / 2),
        vertical_aperture=2 * math.tan(math.radians(58) / 2),
        ray_alignment="base",
        include_geom_groups=(0, 2),
        exclude_parent_body=False,
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
                0.015,
                0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
            ),
            rot=(
                math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
                math.sin(math.radians(0.5) / 2),
                math.sin(math.radians(48) / 2),
                0.0,
            ),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        mesh_filter_max_hops=24,
        noise_pipeline=_noise_pipeline(camera_noise_enabled),
        update_period=1 / 60,
        debug_vis=True,
        depth_clipping_behavior="max",
        min_distance=0.05,
        max_distance=1e6,
    )


def make_grouped_raycaster_camera_cfg(
    camera_profile: DepthCameraProfile = DepthCameraProfile.HEAD,
    camera_noise_enabled: bool = True,
) -> NoisyGroupedRayCasterCameraCfg:
    """Build the grouped raycaster for the selected camera calibration."""
    if camera_profile is DepthCameraProfile.HEAD:
        return make_head_grouped_raycaster_camera_cfg(camera_noise_enabled)
    raise ValueError(f"Unsupported depth camera profile: {camera_profile}")


def make_pinhole_raycast_camera_cfg() -> RayCastSensorCfg:
    """Build the upstream mjlab raycast+pinhole baseline."""
    return RayCastSensorCfg(
        name="depth",
        frame=ObjRef(type="body", name="", entity="robot"),
        pattern=_pinhole_pattern(),
        ray_alignment="base",
        include_geom_groups=(0, 2),
        exclude_parent_body=False,
        max_distance=1e6,
        debug_vis=True,
    )


def make_render_camera_cfg() -> CameraSensorCfg:
    """Build the upstream mjlab MuJoCo-Warp rendered-depth camera."""
    return CameraSensorCfg(
        name="depth",
        width=48,
        height=27,
        fovy=58.0,
        data_types=("depth",),
        use_textures=False,
        use_shadows=False,
        enabled_geom_groups=(0, 2),
    )


def make_depth_sensor_cfg(
    backend: DepthBackend,
    camera_profile: DepthCameraProfile = DepthCameraProfile.HEAD,
    camera_noise_enabled: bool = True,
):
    """Build the selected depth sensor configuration."""
    if backend is DepthBackend.GROUPED_RAYCASTER:
        return make_grouped_raycaster_camera_cfg(camera_profile, camera_noise_enabled)
    if backend is DepthBackend.PINHOLE_RAYCAST:
        return make_pinhole_raycast_camera_cfg()
    if backend is DepthBackend.RENDER:
        return make_render_camera_cfg()
    raise ValueError(f"Unsupported depth backend: {backend}")


def depth_observation_func_name(backend: DepthBackend) -> str:
    """Return the observation function name for the selected sensor."""
    if backend is DepthBackend.GROUPED_RAYCASTER:
        return "raycaster_depth"
    if backend is DepthBackend.PINHOLE_RAYCAST:
        return "pinhole_raycast_depth"
    if backend is DepthBackend.RENDER:
        return "camera_depth"
    raise ValueError(f"Unsupported depth backend: {backend}")
