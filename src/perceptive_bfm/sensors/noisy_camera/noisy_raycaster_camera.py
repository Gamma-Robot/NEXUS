from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..grouped_ray_caster import GroupedRayCasterCamera
from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera(NoisyCameraMixin, GroupedRayCasterCamera):
    """NoisyRayCasterCamera with noise pipeline and history buffers.

    Uses GroupedRayCasterCamera as the ray-caster base implementation.
    """

    cfg: NoisyRayCasterCameraCfg

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        self.build_noise_pipeline()
        self.build_history_buffers()

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset the sensor and noise pipeline."""
        super().reset(env_ids)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    """
    Implementation
    """

    def postprocess_rays(self) -> None:
        """Fills the buffers of the sensor data."""
        super().postprocess_rays()
        if self._update_period_s > 0.0:
            env_ids = self.refresh_mask.nonzero(as_tuple=False).squeeze(-1)
        else:
            env_ids = self._ALL_INDICES
        if env_ids.numel() == 0:
            return
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
