"""PPO configurations for perceptive G1 control."""

from mjlab.rl.config import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from perceptive_bfm.rl.config import RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from perceptive_bfm.tasks.perceptive.mdp import symmetry_terms

SYMMETRY_CFG = RslRlSymmetryCfg(
    data_augmentation_func=symmetry_terms.augment_observations,
    use_mirror_loss=True,
    use_data_augmentation=False,
    mirror_loss_coeff=1.0,
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


def unitree_g1_perceptive_ppo_runner_cfg(symmetry_enabled: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 perceptive task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="CNNModel",
            hidden_dims=(1024, 1024, 512, 512, 256),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            cnn_cfg=DEPTH_IMAGE_ENCODER_CNN_CFG,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(1024, 1024, 512, 512, 256),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if symmetry_enabled else None,
        ),
        experiment_name="g1_perceptive",
        upload_model=False,
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={
            "actor": ("policy", "camera"),
            "critic": ("priv", "priv_critic", "elevation"),
        },
    )


def unitree_g1_perceptive_ppo_teacher_runner_cfg(
    symmetry_enabled: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 perceptive task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if symmetry_enabled else None,
        ),
        experiment_name="g1_perceptive",
        upload_model=False,
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={
            "actor": ("priv", "elevation"),
            "critic": ("priv", "priv_critic", "elevation"),
        },
    )
