"""Unitree G1 contact bodies and kinematic chains."""

from terrain_motion_adaptation.optimization.keyframe_ik import ChainSpec
from terrain_motion_adaptation.robots.config import RobotConfig

G1_CHAIN_SPECS: dict[str, ChainSpec] = {
    "left_ankle_roll_link": ChainSpec(
        hip_body="left_hip_pitch_link",
        mid_body="left_knee_link",
        end_body="left_ankle_roll_link",
        chain_joint_names=(
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
        ),
        orient_joint_names=("left_ankle_pitch_joint", "left_ankle_roll_joint"),
    ),
    "right_ankle_roll_link": ChainSpec(
        hip_body="right_hip_pitch_link",
        mid_body="right_knee_link",
        end_body="right_ankle_roll_link",
        chain_joint_names=(
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
        ),
        orient_joint_names=("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    ),
    "left_wrist_yaw_link": ChainSpec(
        hip_body="left_shoulder_pitch_link",
        mid_body="left_elbow_link",
        end_body="left_wrist_yaw_link",
        chain_joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ),
        orient_joint_names=(),  # rigid hand capsule -- see ChainSpec docstring
    ),
    "right_wrist_yaw_link": ChainSpec(
        hip_body="right_shoulder_pitch_link",
        mid_body="right_elbow_link",
        end_body="right_wrist_yaw_link",
        chain_joint_names=(
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ),
        orient_joint_names=(),
    ),
}


G1_CONFIG = RobotConfig(
    root_body="pelvis",
    end_effectors=(
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
    chain_specs=G1_CHAIN_SPECS,
)
