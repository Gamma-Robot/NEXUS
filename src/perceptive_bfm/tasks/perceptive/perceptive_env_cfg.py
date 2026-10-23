from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from perceptive_bfm.tasks.perceptive import mdp
from perceptive_bfm.tasks.perceptive.mdp import MotionCommandCfg
from perceptive_bfm.tasks.perceptive.sensors import (
    DepthBackend,
    DepthCameraProfile,
    depth_observation_func_name,
    make_depth_sensor_cfg,
)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

PROPRIO_HISTORY_LENGTH = 10  # length of proprioceptive observation history
EXTERO_HISTORY_LENGTH = 4  # length of exteroceptive observation history (e.g. height scan, camera)
CAMERA_DR_ENABLED = True
CAMERA_NOISE_ENABLED = False
DEPTH_BACKEND = DepthBackend.GROUPED_RAYCASTER
DEPTH_CAMERA_PROFILE = DepthCameraProfile.HEAD


def make_perceptive_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base perceptive task configuration."""

    ##
    # Sensors
    ##
    height_scan = RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(type="body", name="", entity="robot"),  # Set per robot.
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        ray_alignment="yaw",
        max_distance=5.0,
        include_geom_groups=(0,),  # Terrain only.
        debug_vis=True,
    )
    cam = make_depth_sensor_cfg(
        DEPTH_BACKEND,
        camera_profile=DEPTH_CAMERA_PROFILE,
        camera_noise_enabled=CAMERA_NOISE_ENABLED,
    )

    ##
    # Observations
    ##

    priv_terms = {
        "motion_anchor_pos_b_future": ObservationTermCfg(
            func=mdp.motion_anchor_pos_b_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "motion_anchor_ori_b_future": ObservationTermCfg(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "motion_anchor_lin_vel_b_future": ObservationTermCfg(
            func=mdp.motion_anchor_lin_vel_b_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "motion_anchor_ang_vel_b_future": ObservationTermCfg(
            func=mdp.motion_anchor_ang_vel_b_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "motion_joint_state_future": ObservationTermCfg(
            func=mdp.motion_joint_state_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "motion_projected_gravity_future": ObservationTermCfg(
            func=mdp.motion_projected_gravity_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "body_pos": ObservationTermCfg(
            func=mdp.robot_body_pos_b, params={"command_name": "motion"}
        ),
        "body_ori": ObservationTermCfg(
            func=mdp.robot_body_ori_b, params={"command_name": "motion"}
        ),
        "body_lin_vel": ObservationTermCfg(
            func=mdp.robot_body_lin_vel_b, params={"command_name": "motion"}
        ),
        "body_ang_vel": ObservationTermCfg(
            func=mdp.robot_body_ang_vel_b, params={"command_name": "motion"}
        ),
        "motion_body_pos_b_future": ObservationTermCfg(
            func=mdp.motion_body_pos_b_future,
            params={"command_name": "motion", "include_diff": True, "flatten": True},
        ),
        "motion_body_ori_b_future": ObservationTermCfg(
            func=mdp.motion_body_ori_b_future,
            params={"command_name": "motion", "include_diff": True, "flatten": True},
        ),
        "base_lin_vel_history": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "base_ang_vel_history": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "projected_gravity_history": ObservationTermCfg(
            func=mdp.projected_gravity,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos_history": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel_history": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "applied_action": ObservationTermCfg(func=mdp.applied_action),
        "applied_torque": ObservationTermCfg(func=mdp.applied_torque),
        "motion_anchor_pos_z_error": ObservationTermCfg(
            func=mdp.motion_anchor_pos_z_error, params={"command_name": "motion"}
        ),
        "motion_anchor_ori_gravity_error": ObservationTermCfg(
            func=mdp.motion_anchor_ori_gravity_error, params={"command_name": "motion"}
        ),
        "ee_contact_state": ObservationTermCfg(
            func=mdp.ee_contact_state,
            params={"command_name": "motion", "sensor_name": "contact_forces"},
        ),
        "motion_ee_contact_state": ObservationTermCfg(
            func=mdp.motion_ee_contact_state, params={"command_name": "motion"}
        ),
    }

    priv_critic_terms = {
        "domain_motor_params_implicit": ObservationTermCfg(
            func=mdp.domain_motor_params_implicit_obs
        ),
        "domain_body_materials": ObservationTermCfg(func=mdp.domain_body_materials_obs),
        "domain_joint_action_offset": ObservationTermCfg(
            func=mdp.domain_joint_action_offset_obs, params={"action_term_name": "joint_pos"}
        ),
    }

    policy_terms = {
        "teleop_anchor_pos_rel_future": ObservationTermCfg(
            func=mdp.teleop_anchor_pos_rel_future, params={"command_name": "motion"}
        ),
        "teleop_anchor_ori_b_future": ObservationTermCfg(
            func=mdp.teleop_anchor_ori_b_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "teleop_joint_pos_future": ObservationTermCfg(
            func=mdp.teleop_joint_pos_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "teleop_anchor_z_future": ObservationTermCfg(
            func=mdp.teleop_anchor_z_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "teleop_projected_gravity_future": ObservationTermCfg(
            func=mdp.teleop_projected_gravity_future,
            params={"command_name": "motion", "flatten": True},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"biased": True},
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        ),
    }

    cam_terms = {
        "camera_depth": ObservationTermCfg(
            func=getattr(mdp, depth_observation_func_name(DEPTH_BACKEND)),
            params={"sensor_name": "depth"},
        )
    }

    # Flatten sensor-relative terrain height for the privileged actor and critics.
    elevation_terms = {
        "elevation_map": ObservationTermCfg(
            func=mdp.elevation_map,
            params={
                "sensor_name": "height_scan",
                "coords": "z",
                "clip_height": (-1.5, 0.0),
            },
        )
    }

    observations = {
        "priv": ObservationGroupCfg(
            terms=priv_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "priv_critic": ObservationGroupCfg(
            terms=priv_critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "camera": ObservationGroupCfg(
            terms=cam_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "elevation": ObservationGroupCfg(
            terms=elevation_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "motion": MotionCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
            # Override in robot cfg.
            motion_file="",
            anchor_body_name="",
            body_names=(),
        )
    }

    ##
    # Events
    ##

    events: dict[str, EventTermCfg] = {
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.01, 0.01),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,  # All foot geoms share the same friction.
            },
        ),
    }
    # The upstream baseline sensors do not expose grouped-camera calibration setters.
    if CAMERA_DR_ENABLED and DEPTH_BACKEND is DepthBackend.GROUPED_RAYCASTER:
        events.update(
            {
                "camera_intrinsic": EventTermCfg(
                    mode="reset",
                    func=mdp.camera_intrinsic,
                    params={"sensor_name": "depth"},
                ),
                "camera_pos": EventTermCfg(
                    mode="reset",
                    func=mdp.camera_pos,
                    params={"sensor_name": "depth"},
                ),
                "camera_quat": EventTermCfg(
                    mode="reset",
                    func=mdp.camera_quat,
                    params={"sensor_name": "depth"},
                ),
            }
        )

    ##
    # Rewards
    ##

    rewards: dict[str, RewardTermCfg] = {
        "motion_global_anchor_pos": RewardTermCfg(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_anchor_ori": RewardTermCfg(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_global_anchor_lin_vel": RewardTermCfg(
            func=mdp.motion_global_anchor_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_global_anchor_ang_vel": RewardTermCfg(
            func=mdp.motion_global_anchor_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.0},
        ),
        "motion_body_pos": RewardTermCfg(
            func=mdp.motion_relative_body_position_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.0},
        ),
        "motion_joint_pos": RewardTermCfg(
            func=mdp.motion_joint_pos_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.5},
        ),
        "motion_joint_vel": RewardTermCfg(
            func=mdp.motion_joint_vel_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 3.0},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-2e-2),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
        "action_rate_l2_waist": RewardTermCfg(
            func=mdp.action_rate_l2_waist,
            weight=-2.5e-1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*waist.*joint",))},
        ),
        "survival": RewardTermCfg(func=mdp.is_alive, weight=3.0),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-5e-4),
        "joint_torque_limits": RewardTermCfg(
            func=mdp.joint_torque_limits,
            weight=-1e-2,
            params={"soft_factor": 0.75},
        ),
        # Contact-timing rewards, one pair per EE group. The sparse pair shares an
        # implementation (contact timing has no geometry in it); the dense pair does
        # not, since a foot is an extended sole and a hand is a point. The dense
        # terms shape each EE's clearance above the terrain under it toward the
        # REFERENCE motion's own clearance -- no hand-tuned target heights, so they
        # stay correct for crawling/climbing as well as walking. `clearance_std` is
        # the tracking tolerance in metres: error of that size scores ~-0.63.
        "feet_air_time_ref": RewardTermCfg(
            func=mdp.ee_air_time_reference,
            weight=5.0,
            params={
                "command_name": "motion",
                "sensor_name": "contact_forces",
                "threshold": 0.5,
                "ee_group": "feet",
            },
        ),
        "feet_air_time_ref_dense": RewardTermCfg(
            func=mdp.feet_air_time_reference_dense,
            weight=1.0,
            params={
                "command_name": "motion",
                "sensor_name": "contact_forces",
                "clearance_std": 0.05,
            },
        ),
        "hand_air_time_ref": RewardTermCfg(
            func=mdp.ee_air_time_reference,
            weight=5.0,
            params={
                "command_name": "motion",
                "sensor_name": "contact_forces",
                "threshold": 0.5,
                "ee_group": "hands",
            },
        ),
        # Looser than the feet: a hand's height is far less critical than a foot's
        # (nothing trips over it), so it should not compete with the feet's shaping.
        "hand_air_time_ref_dense": RewardTermCfg(
            func=mdp.hand_air_time_reference_dense,
            weight=1.0,
            params={
                "command_name": "motion",
                "sensor_name": "contact_forces",
                "clearance_std": 0.10,
            },
        ),
        "feet_support": RewardTermCfg(
            func=mdp.feet_support,
            weight=-1.0,
            params={
                "command_name": "motion",
                "sensor_name": "contact_forces",
                "clearance_threshold": 0.03,
                "sample_spacing": 0.02,
            },
        ),
    }

    ##
    # Terminations
    ##

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "anchor_pos_z": TerminationTermCfg(
            func=mdp.bad_anchor_pos_z_only,
            # params={"command_name": "motion", "threshold": 0.25},
            # params={"command_name": "motion", "threshold": 0.35},
            params={"command_name": "motion", "threshold": float("inf")},
        ),
        "anchor_clearance": TerminationTermCfg(
            func=mdp.bad_anchor_clearance,
            params={"command_name": "motion", "threshold": 0.25},
        ),
        # "anchor_pos_xy": TerminationTermCfg(
        #     func=mdp.bad_anchor_pos_xy,
        #     params={"command_name": "motion", "threshold": 0.25},
        # ),
        "anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.8,
            },
        ),
        "ee_body_pos": TerminationTermCfg(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                # "threshold": 0.25,
                # "threshold": 0.35,
                "threshold": float("inf"),
                "body_names": (),  # Set per-robot.
            },
        ),
        "ee_body_pos_anchor_relative": TerminationTermCfg(
            func=mdp.bad_motion_body_pos_anchor_relative_z,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": (),  # Set per-robot.
            },
        ),
    }

    ##
    # Curriculum
    ##

    # reference reward curriculum: ramp the regularization rewards' weights 0.5x -> 1.0x
    # over training (every term not named "motion_*"). Ported from tracking's
    # tracking_env_cfg.py (same params).
    curriculum: dict[str, CurriculumTermCfg] = {
        "reward_scale": CurriculumTermCfg(
            func=mdp.modify_reward_scale,
            params={
                "num_steps_fallback": 30_000 * 24,
                "task_reward_prefix": "motion_",
                "factor_range": (0.5, 1.0),
            },
        ),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=(height_scan, cam),
            num_envs=1,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            # nconmax=35,
            # njmax=250,
            nconmax=50,
            njmax=384,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )
