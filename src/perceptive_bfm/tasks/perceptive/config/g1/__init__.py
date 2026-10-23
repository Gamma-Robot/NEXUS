from perceptive_bfm.tasks.perceptive.rl import (
    MotionTrackingDistillationRunner,
    MotionTrackingOnPolicyRunner,
)
from perceptive_bfm.tasks.registry import register_mjlab_task
from .distillation_cfg import unitree_g1_perceptive_distillation_runner_cfg
from .env_cfgs import unitree_g1_rough_perceptive_env_cfg
from .rl_cfg import (
    unitree_g1_perceptive_ppo_runner_cfg,
    unitree_g1_perceptive_ppo_teacher_runner_cfg,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(reward_curriculum=True, terrain="pyramid"),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(play=True, terrain="pyramid"),
    rl_cfg=unitree_g1_perceptive_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1-Symmetry",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(reward_curriculum=True, terrain="pyramid"),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(play=True, terrain="pyramid"),
    rl_cfg=unitree_g1_perceptive_ppo_runner_cfg(symmetry_enabled=True),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1-Teacher",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(
        reward_curriculum=True, terrain="pyramid", camera_enabled=False, flat_motion_enabled=False
    ),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(
        play=True, terrain="pyramid", camera_enabled=False
    ),
    rl_cfg=unitree_g1_perceptive_ppo_teacher_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1-Teacher-Symmetry",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(
        reward_curriculum=True, terrain="pyramid", camera_enabled=False, flat_motion_enabled=False
    ),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(
        play=True, terrain="pyramid", camera_enabled=False
    ),
    rl_cfg=unitree_g1_perceptive_ppo_teacher_runner_cfg(symmetry_enabled=True),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1-BC",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(terrain="pyramid"),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(play=True, terrain="pyramid"),
    rl_cfg=unitree_g1_perceptive_distillation_runner_cfg(),
    runner_cls=MotionTrackingDistillationRunner,
)

register_mjlab_task(
    task_id="Perceptive-PyramidTerrain-Unitree-G1-BC-Symmetry",
    env_cfg=unitree_g1_rough_perceptive_env_cfg(terrain="pyramid"),
    play_env_cfg=unitree_g1_rough_perceptive_env_cfg(play=True, terrain="pyramid"),
    rl_cfg=unitree_g1_perceptive_distillation_runner_cfg(symmetry_enabled=True),
    runner_cls=MotionTrackingDistillationRunner,
)
