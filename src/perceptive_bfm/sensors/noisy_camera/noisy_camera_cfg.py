from dataclasses import dataclass, field

from perceptive_bfm.utils.noise import ImageNoiseCfg, NoiseCfg


@dataclass(kw_only=True)
class NoisyCameraCfgMixin:
    """
    Configuration class for the NoisyCamera sensor and manages image transforms and their parameters.
    """

    noise_pipeline: dict[str, ImageNoiseCfg | NoiseCfg] = field(default_factory=dict)
    """Configurations for the noise pipeline. The keys can be self-defined names.
    NOTE: All enabled items in cfg.data_types will be applied with the noise pipeline.
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    NOTE: After python 3.8, the dict is ordered by insertion order.
    """

    data_histories: dict[str, int] = field(default_factory=dict)
    """ Configurations for adding history to specified data_types. Please specify which `data_type`
    you want to add history and the history length. The stacked history observation will
    be placed in sensor.data[f"{data_type}_history"]
    NOTE: If you want to add history to the noised output, you need to specify the data_type as the one with _noised postfix.
    """
