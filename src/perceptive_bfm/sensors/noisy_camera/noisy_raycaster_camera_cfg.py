from dataclasses import dataclass

from ..grouped_ray_caster import GroupedRayCasterCameraCfg
from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_raycaster_camera import NoisyRayCasterCamera


@dataclass(kw_only=True)
class NoisyRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    """
    Configuration class for the NoisyRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type = NoisyRayCasterCamera

    def build(self) -> NoisyRayCasterCamera:
        return NoisyRayCasterCamera(self)
