"""Policy-export metadata for the ordered observation groups and robot actuators.

Observation terms follow the model group order rather than an assumed actor group name."""

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def get_base_metadata(
    env: ManagerBasedRlEnv, run_path: str, actor_obs_groups: list[str]
) -> dict[str, list | str | float]:
    """Get base metadata common to all RL policy exports.

    Args:
        env: The RL environment.
        run_path: W&B run path or other identifier.
        actor_obs_groups: env-level ObservationManager group names feeding the
            deployed actor network, in concatenation order (e.g. the exported
            model's ``.obs_groups`` followed by ``.obs_groups_2d`` for CNNs).
    """
    robot: Entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    # Build mapping from joint name to actuator ID for natural joint order.
    # Each spec actuator controls exactly one joint (via its target field).
    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
    # Get actuator IDs in natural joint order (same order as robot.joint_names).
    ctrl_ids_natural = [
        joint_name_to_ctrl_id[jname]
        for jname in robot.joint_names  # global joint order
        if jname in joint_name_to_ctrl_id  # skip non-actuated joints
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

    observation_names: list = []
    observation_term_scale: list = []
    observation_term_flatten_history_dim: list = []
    observation_term_history_length: list = []
    observation_term_clip: list = []

    for group in actor_obs_groups:
        for active_term in env.observation_manager.active_terms[group]:
            cfg = env.observation_manager.get_term_cfg(group, active_term)
            observation_names.append(active_term)

            if cfg.scale is None:
                observation_term_scale.append(1.0)
            else:
                raw_scale = cfg.scale
                scale = (
                    raw_scale.cpu().tolist() if isinstance(raw_scale, torch.Tensor) else raw_scale
                )
                observation_term_scale.append(scale)

            raw_clip = cfg.clip
            if raw_clip is None:
                observation_term_clip.append([float("-inf"), float("inf")])
            else:
                observation_term_clip.append(list(raw_clip))

            observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
            observation_term_history_length.append(cfg.history_length)

    return {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness.tolist(),
        "joint_damping": joint_damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_terms_scale": observation_term_scale,
        "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
        "observation_terms_history_length": observation_term_history_length,
        "observation_terms_clip": observation_term_clip,
        "action_scale": (
            joint_action._scale[0].cpu().tolist()
            if isinstance(joint_action._scale, torch.Tensor)
            else joint_action._scale
        ),
    }
