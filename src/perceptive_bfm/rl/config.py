from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from mjlab.rl.config import RslRlBaseRunnerCfg, RslRlModelCfg


@dataclass
class RslRlSymmetryCfg:
    """Configuration for observation and action symmetry transformations."""

    data_augmentation_func: Callable
    """Function(env, obs, actions) -> (aug_obs, aug_actions). Pass a direct
    function reference (e.g. symmetry_terms.augment_observations) -- rsl-rl's
    resolve_callable() accepts an already-callable object unchanged, so no
    dotted-path string is needed."""
    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation."""
    use_mirror_loss: bool = False
    """Whether to add the symmetry-mirror loss to the total loss."""
    mirror_loss_coeff: float = 1.0
    """Weight for the symmetry-mirror loss when use_mirror_loss is True."""


@dataclass
class RslRlDistillationAlgorithmCfg:
    """Config for the distillation algorithm."""

    num_learning_epochs: int = 1
    """The number of updates performed with each sample."""
    gradient_length: int = 15
    """Recurrent gradient window length; non-recurrent students use one-step samples."""
    num_mini_batches: int = 4
    """Minibatches per epoch over pooled samples or recurrent windows."""
    learning_rate: float = 1e-3
    """The learning rate for the student policy."""
    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""
    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy."""
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use for the student policy."""
    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Optional symmetry augmentation configuration."""
    symmetry_bc_coef: float = 1.0
    """Weight of the symmetric-pair BC loss when symmetry_cfg is set. Same
    weight as the ordinary BC loss by default (direct supervision, not a
    self-consistency regularizer)."""
    class_name: str = "perceptive_bfm_rl.algorithms:Distillation"
    """The algorithm class resolved by the runner."""


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "DistillationRunner"
    """The runner class name. Default is DistillationRunner."""
    student: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    """The student configuration."""
    teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    """The teacher configuration."""
    algorithm: RslRlDistillationAlgorithmCfg = field(default_factory=RslRlDistillationAlgorithmCfg)
    """The algorithm configuration."""


@dataclass
class RslRlPpoAlgorithmCfg:
    """PPO configuration including optional RND and symmetry settings."""

    num_learning_epochs: int = 5
    """The number of learning epochs per update."""
    num_mini_batches: int = 4
    """The number of mini-batches per update."""
    clip_param: float = 0.2
    """The clipping parameter for the policy."""
    gamma: float = 0.99
    """The discount factor."""
    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    value_loss_coef: float = 1.0
    """The coefficient for the value loss."""
    entropy_coef: float = 0.005
    """The coefficient for the entropy loss."""
    learning_rate: float = 1.0e-3
    """The learning rate."""
    max_grad_norm: float = 1.0
    """The maximum gradient norm for the policy."""
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use."""
    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    """The learning rate schedule."""
    desired_kl: float = 0.01
    """The desired KL divergence between the new and old policies."""
    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch."""
    rnd_cfg: dict[str, Any] | None = None
    """Random Network Distillation config. None disables RND."""
    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Symmetry augmentation config. None (default) disables it --
    behavior is then identical to PPO without symmetry support."""
    class_name: str = "perceptive_bfm_rl.algorithms:PPO"
    """The algorithm class name resolved by RSL-RL."""
