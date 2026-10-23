"""Teacher-student distillation configuration for perceptive G1 control."""

from mjlab.rl.config import RslRlModelCfg
from perceptive_bfm.rl.config import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlSymmetryCfg,
)
from perceptive_bfm.tasks.perceptive.mdp import symmetry_terms

SYMMETRY_CFG = RslRlSymmetryCfg(
    data_augmentation_func=symmetry_terms.augment_observations,
)

DEPTH_IMAGE_ENCODER_CNN_CFG = {
    "camera": {
        "output_channels": (32, 32, 64),
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 1),
        "padding": "zeros",
        "activation": "elu",
        "global_pool": "max",
        "flatten": True,
    },
}


def unitree_g1_perceptive_distillation_runner_cfg(
    symmetry_enabled: bool = False,
) -> RslRlDistillationRunnerCfg:
    """Create supervised distillation runner configuration for Unitree G1 perceptive task."""
    return RslRlDistillationRunnerCfg(
        student=RslRlModelCfg(
            class_name="CNNModel",
            hidden_dims=(1024, 1024, 512, 512, 256),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.0,
                "std_type": "scalar",
                "learn_std": False,
            },
            cnn_cfg=DEPTH_IMAGE_ENCODER_CNN_CFG,
        ),
        teacher=RslRlModelCfg(
            hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlDistillationAlgorithmCfg(
            num_learning_epochs=2,
            learning_rate=1.0e-3,
            gradient_length=1,
            num_mini_batches=8,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if symmetry_enabled else None,
            class_name="perceptive_bfm_rl.algorithms:Distillation",
        ),
        experiment_name="g1_perceptive",
        upload_model=False,
        save_interval=500,
        num_steps_per_env=16,
        max_iterations=10_000,
        obs_groups={
            "student": ("policy", "camera"),
            "teacher": ("priv", "elevation"),
        },
    )
