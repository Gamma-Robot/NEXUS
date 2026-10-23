"""Pure-data G1 model and keyframe definitions with no mjlab dependency."""

from pathlib import Path

G1_XML: Path = Path(__file__).parent / "robots" / "unitree_g1" / "xmls" / "g1.xml"
assert G1_XML.exists()

HOME_ROOT_POS = (0.0, 0.0, 0.783675)
HOME_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}

KNEES_BENT_ROOT_POS = (0.0, 0.0, 0.76)
KNEES_BENT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
