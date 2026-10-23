"""Unitree G1 rough perceptive environment configurations."""

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
    CameraSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
)

from perceptive_bfm.asset_zoo.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from perceptive_bfm.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from perceptive_bfm.sensors.majority_contact import MajorityContactSensorCfg
from perceptive_bfm.tasks.perceptive.mdp import MotionCommandCfg
from perceptive_bfm.tasks.perceptive.perceptive_env_cfg import (
    make_perceptive_env_cfg,
)
from perceptive_bfm.terrains.config import _build_pyramid_terrain

_TERRAIN_BUILDERS = {"pyramid": _build_pyramid_terrain}


def _disable_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
    """Remove the complete depth-camera pipeline from a built environment config."""
    sensors = tuple(cfg.scene.sensors or ())
    depth_sensors = tuple(sensor for sensor in sensors if sensor.name == "depth")
    if len(depth_sensors) != 1:
        raise ValueError(
            f"Expected exactly one depth sensor before pruning, got {len(depth_sensors)}"
        )
    if "camera" not in cfg.observations:
        raise ValueError("Expected a camera observation group before pruning")

    cfg.scene.sensors = tuple(sensor for sensor in sensors if sensor.name != "depth")
    del cfg.observations["camera"]
    for event_name in ("camera_intrinsic", "camera_pos", "camera_quat"):
        cfg.events.pop(event_name, None)


def _disable_flat_motion(cfg: ManagerBasedRlEnvCfg) -> None:
    """Prune flat-source consumers before disabling their motion/FK cache."""
    motion_cfg = cfg.commands["motion"]
    if not isinstance(motion_cfg, MotionCommandCfg):
        raise TypeError(f"Expected MotionCommandCfg, got {type(motion_cfg).__name__}")
    if motion_cfg.obs_motion_source != "terrain":
        raise ValueError("Flat motion can only be disabled when obs_motion_source='terrain'")
    if "policy" not in cfg.observations:
        raise ValueError("Expected a policy observation group before pruning flat motion")

    del cfg.observations["policy"]
    motion_cfg.flat_motion_enabled = False
    motion_cfg.viz.show_src_ghost = False
    motion_cfg.viz.show_ee_contact = False


def unitree_g1_rough_perceptive_env_cfg(
    play: bool = False,
    reward_curriculum: bool = False,
    terrain: Literal["pyramid"] = "pyramid",
    camera_enabled: bool = True,
    flat_motion_enabled: bool = True,
) -> ManagerBasedRlEnvCfg:
    cfg = make_perceptive_env_cfg()

    if not camera_enabled:
        _disable_depth_camera(cfg)
    if not flat_motion_enabled:
        _disable_flat_motion(cfg)

    # cfg.sim.nconmax = 200
    # cfg.sim.njmax = 1500
    # cfg.sim.mujoco.ccd_iterations = 128
    # cfg.sim.mujoco.iterations = 10
    # cfg.sim.mujoco.ls_iterations = 20

    cfg.scene.terrain = _TERRAIN_BUILDERS[terrain]()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors += (self_collision_cfg,)

    # EE-ground contact for the feet/hand air_time rewards. A MajorityContactSensor adds a
    # per-control-step substep-majority `current_contact` (the reference contact cache). Config
    # matches tracking's `config/g1/env_cfgs.py` `contact_forces` sensor: `substeps` =
    # decimation is the majority window; secondary "terrain" body match works the same
    # way on generator (rough) terrain as it does on tracking's flat plane.
    contact_cfg = MajorityContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(
            mode="subtree",
            # All four EEs, not just the feet: the hand air-time rewards need real
            # wrist-terrain contact too. One shared sensor -- resolve_contact_indices
            # maps each reward's own body-name tuple onto the right columns.
            pattern=r"^(left|right)_(ankle_roll|wrist_yaw)_link$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        global_frame=True,
        num_slots=1,
        track_air_time=True,
        history_length=3,
        debug=False,
        substeps=cfg.decimation,
    )
    cfg.scene.sensors += (contact_cfg,)

    # Set raycast sensor frame to G1 pelvis.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "height_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "pelvis"

    # Set camera sensor's extrinsic parameters.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "depth":
            if isinstance(sensor, (GroupedRayCasterCameraCfg, RayCastSensorCfg)):
                assert isinstance(sensor.frame, ObjRef)
                if not sensor.frame.name:
                    sensor.frame.name = "torso_link"
            else:
                assert isinstance(sensor, CameraSensorCfg)
                sensor.parent_body = "robot/torso_link"
                # sensor.pos = (0.0576235, 0.01753, 0.41987)
                # sensor.quat = convert_camera_orientation(
                #   quat=(0.9149597, 0, 0.4035453, 0),  # rpy = (0, 0.8307767239493009, 0)
                #   from_convention="world",
                #   to_convention="opengl",
                # )
                # sensor.quat = convert_camera_orientation(
                #   quat=(1, 0, 0, 0),
                #   from_convention="world",
                #   to_convention="opengl",
                # )
                # sensor.fovy = 58.0  #55.2

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    # motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.anchor_body_name = "pelvis"
    motion_cmd.body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    motion_cmd.foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    motion_cmd.hand_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.terminations["ee_body_pos_anchor_relative"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    cfg.viewer.body_name = "torso_link"

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        if flat_motion_enabled:
            cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

        # EE contact debug visualization (green=contact, red=swing) -- play-only,
        # since update_ee_contact_viz costs nothing when show_ee_contact is off and
        # there's no reason to pay it during real training.
        if flat_motion_enabled:
            motion_cmd.ee_contact_viz_body_names = (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
            motion_cmd.viz.show_ee_contact = True

        if camera_enabled:
            # Enable depth visualization for debugging. debug_vis_mirror additionally
            # shows each env's frame flipped left-right (the transform symmetry_terms.py's
            # camera_transform applies to this group) side by side, as a visual sanity
            # check of depth symmetry correctness.
            cfg.observations["camera"].terms["camera_depth"].params["debug_vis"] = True
            cfg.observations["camera"].terms["camera_depth"].params["debug_vis_mirror"] = True

    # Reward curriculum is opt-in and never runs during play (regularization
    # weights then stay at full configured strength, i.e. reference progress=1.0).
    if not reward_curriculum or play:
        cfg.curriculum = {}

    return cfg
