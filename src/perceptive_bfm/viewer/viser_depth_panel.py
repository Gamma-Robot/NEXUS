"""Viser GUI panel showing the raycaster depth camera's images.

mjlab's own camera panel (``ViserCameraOverlays``) only picks up sensors that are
``isinstance(..., CameraSensor)``, so the perceptive task's
``NoisyGroupedRayCasterCamera`` -- a ``RayCastSensor`` subclass -- never gets one
and ``play.py --viewer viser`` shows no depth image at all. Nothing needs
rendering to fix that: the images already sit on the sensor as tensors, so this
module is only GUI widgets plus a per-frame push.

Two images per sensor (see
docs/superpowers/specs/2026-08-01-viser-depth-panel-design.md):

* raw -- ``distance_to_image_plane``, in metres, scaled by a slider (needed
  because the perceptive camera runs at ``max_distance=1e6``, so miss rays would
  flatten any fixed normalization);
* policy input -- ``distance_to_image_plane_noised``, what the noise pipeline
  hands ``raycaster_depth`` and hence the policy, already in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from perceptive_bfm.sensors.grouped_ray_caster.grouped_ray_caster_camera import (
    GroupedRayCasterCamera,
)

if TYPE_CHECKING:
    import viser

RAW_KEY = "distance_to_image_plane"
NOISED_KEY = "distance_to_image_plane_noised"

MIN_DISPLAY_SIZE = 128
"""Long-side pixel target the images are nearest-upsampled to -- the policy input
is 18x32 and unreadable at native size. Matches mjlab's ViserCameraViewer."""

DEFAULT_RAW_SCALE_M = 2.0
"""Metres mapped to full white in the raw panel. Matches the perceptive camera's
DepthNormalizationCfg(depth_range=(0.0, 2.0)), so both panels share a scale by
default and can be compared directly."""


def discover_raycaster_cameras(env: Any) -> list[GroupedRayCasterCamera]:
    """Every raycaster depth camera in ``env``'s scene, in scene order. ``env``
    must be the unwrapped env."""
    return [s for s in env.scene.sensors.values() if isinstance(s, GroupedRayCasterCamera)]


def depth_to_uint8_rgb(depth_hw: torch.Tensor, scale: float) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) uint8 greyscale, with ``scale`` mapping to white.

    Anything at or beyond ``scale`` clips to 255, so the perceptive camera's
    huge miss-ray values show as flat white rather than dragging everything else
    to black.
    """
    depth = depth_hw.detach().float().cpu().numpy()
    normalized = np.clip(depth / max(scale, 1e-6), 0.0, 1.0)
    grey = (normalized * 255).astype(np.uint8)
    return np.repeat(grey[:, :, np.newaxis], 3, axis=-1)


def upsample_nearest(image: np.ndarray, min_display_size: int = MIN_DISPLAY_SIZE) -> np.ndarray:
    """Nearest-neighbour upsample so the long side reaches ``min_display_size``."""
    scale = max(1, min_display_size // max(image.shape[0], image.shape[1]))
    if scale == 1:
        return image
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


@dataclass
class _ImagePanel:
    """One image widget plus the checkbox gating it. The handle is destroyed and
    rebuilt on toggle rather than hidden, because viser's image handles have no
    ``visible`` attribute -- hence ``folder``, which is re-entered so a rebuilt
    image lands back inside its own folder instead of at the GUI root."""

    label: str
    key: str
    folder: Any
    checkbox: Any
    handle: Any = None

    def update(self, server: viser.ViserServer, image: np.ndarray | None) -> None:
        if not self.checkbox.value:
            if self.handle is not None:
                self.handle.remove()
                self.handle = None
            return
        assert image is not None
        if self.handle is None:
            with self.folder:
                self.handle = server.gui.add_image(image, label=self.label, format="jpeg")
        else:
            self.handle.image = image

    def cleanup(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@dataclass
class _SensorPanels:
    sensor: GroupedRayCasterCamera
    raw: _ImagePanel
    raw_scale: Any
    policy: _ImagePanel | None = None


class ViserRaycasterDepthPanel:
    """Depth-image panels for every ``GroupedRayCasterCamera`` in the scene.

    Builds nothing when the scene has none, so flat ``Tracking-*`` tasks are
    unaffected.
    """

    def __init__(self, server: viser.ViserServer, env: Any) -> None:
        self._server = server
        self._env = env
        self._panels: list[_SensorPanels] = []

    def setup(self) -> None:
        for sensor in discover_raycaster_cameras(self._env):
            self._panels.append(self._setup_sensor(sensor))

    def update(self, env_idx: int) -> None:
        """Push the selected env's images. Called every frame the viewer refreshes
        its camera feeds."""
        for panels in self._panels:
            output = panels.sensor.data.output
            panels.raw.update(
                self._server,
                self._render(output, panels.raw, env_idx, panels.raw_scale.value),
            )
            if panels.policy is not None:
                panels.policy.update(
                    self._server, self._render(output, panels.policy, env_idx, 1.0)
                )

    def cleanup(self) -> None:
        for panels in self._panels:
            panels.raw.cleanup()
            if panels.policy is not None:
                panels.policy.cleanup()

    # Private helpers.

    def _setup_sensor(self, sensor: GroupedRayCasterCamera) -> _SensorPanels:
        name = sensor.cfg.name
        folder = self._server.gui.add_folder(name)
        with folder:
            raw_checkbox = self._server.gui.add_checkbox("Raw", initial_value=True)
            raw_scale = self._server.gui.add_slider(
                "Raw scale (m)",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=DEFAULT_RAW_SCALE_M,
            )
            raw = _ImagePanel(f"{name}_raw", RAW_KEY, folder, raw_checkbox)
            raw.handle = self._server.gui.add_image(
                self._blank(sensor, RAW_KEY), label=raw.label, format="jpeg"
            )

            policy = None
            if NOISED_KEY in sensor.data.output:
                policy_checkbox = self._server.gui.add_checkbox("Policy input", initial_value=True)
                policy = _ImagePanel(f"{name}_policy_input", NOISED_KEY, folder, policy_checkbox)
                policy.handle = self._server.gui.add_image(
                    self._blank(sensor, NOISED_KEY), label=policy.label, format="jpeg"
                )
        return _SensorPanels(sensor=sensor, raw=raw, raw_scale=raw_scale, policy=policy)

    @staticmethod
    def _render(output: dict, panel: _ImagePanel, env_idx: int, scale: float) -> np.ndarray | None:
        if not panel.checkbox.value:
            return None  # Skip the device->host copy for a hidden panel.
        return upsample_nearest(depth_to_uint8_rgb(output[panel.key][env_idx, :, :, 0], scale))

    @staticmethod
    def _blank(sensor: GroupedRayCasterCamera, key: str) -> np.ndarray:
        height, width = sensor.data.output[key].shape[1:3]
        return upsample_nearest(np.zeros((height, width, 3), dtype=np.uint8))
