"""MuJoCo robot/scene composition and qpos schema helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from terrain_motion_adaptation.utils.motion import ROOT_QPOS_NAMES

if TYPE_CHECKING:
    from terrain_motion_adaptation.robots.config import RobotConfig

_PLANE_SCENE_XML = """
<mujoco model="plane">
  <worldbody>
    <body name="terrain">
      <geom name="terrain/ground" type="plane" size="10 10 0.1"
            rgba="0.55 0.58 0.62 1" group="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_scene_spec(scene_path: str | Path | None) -> mujoco.MjSpec:
    if scene_path is None:
        spec = mujoco.MjSpec.from_string(_PLANE_SCENE_XML)
        label = "default plane"
    else:
        path = Path(scene_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene XML not found: {path}")
        spec = mujoco.MjSpec.from_file(str(path))
        label = str(path)

    scene_model = spec.compile()
    if scene_model.njnt or scene_model.nu or scene_model.nsensor:
        raise ValueError(
            f"scene must contain terrain only, without joints/actuators/sensors: {label}"
        )
    try:
        scene_model.body("terrain")
    except KeyError as exc:
        raise ValueError(f"scene must contain a body named 'terrain': {label}") from exc
    return spec


def combine_model_and_scene(
    model_path: str | Path, scene_path: str | Path | None
) -> mujoco.MjModel:
    """Attach a robot MJCF to a terrain-only scene and compile the result."""
    robot_path = Path(model_path).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"robot model XML not found: {robot_path}")
    scene_spec = _load_scene_spec(scene_path)
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    attachment = scene_spec.worldbody.add_frame(name="nexus/robot_attachment")
    scene_spec.attach(robot_spec, prefix="", suffix="", frame=attachment)
    try:
        model = scene_spec.compile()
    except ValueError as exc:
        raise ValueError(
            f"failed to combine robot {robot_path} with scene {scene_path}: {exc}"
        ) from exc
    if model.nq < 7:
        raise ValueError("combined robot must have a floating base with at least seven qpos values")
    return model


def qpos_names_from_model(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return qpos names for one free-root, scalar-joint robot."""
    scalar_joints: list[tuple[int, str]] = []
    free_joints: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            free_joints.append(qpos_address)
        elif joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            if name is None:
                raise ValueError(f"scalar joint at id {joint_id} has no name")
            scalar_joints.append((qpos_address, name))
        else:
            raise ValueError(
                "Motion adaptation currently supports only free, hinge and slide joints"
            )
    if free_joints != [0]:
        raise ValueError(
            f"robot must have exactly one free joint at qpos address 0, got {free_joints}"
        )
    scalar_joints.sort()
    return ROOT_QPOS_NAMES + tuple(name for _, name in scalar_joints)


def validate_qpos_schema(model: mujoco.MjModel, qpos_names: Sequence[str]) -> None:
    expected = qpos_names_from_model(model)
    _qpos_indices(qpos_names, expected)


def _qpos_indices(source_names: Sequence[str], target_names: Sequence[str]) -> list[int]:
    source = tuple(source_names)
    target = tuple(target_names)
    if source[:7] != ROOT_QPOS_NAMES or target[:7] != ROOT_QPOS_NAMES:
        raise ValueError("qpos must begin with fixed xyz/wxyz root columns")
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("qpos joint names contain duplicates")
    if set(source) != set(target):
        raise ValueError(
            f"qpos names mismatch: missing={sorted(set(target)-set(source))}, extra={sorted(set(source)-set(target))}"
        )
    return [source.index(name) for name in target]


def reorder_qpos(
    qpos: np.ndarray, source_names: Sequence[str], target_names: Sequence[str]
) -> np.ndarray:
    """Map named scalar joints while keeping the fixed root columns intact."""
    values = np.asarray(qpos)
    if values.ndim != 2 or values.shape[1] != len(source_names):
        raise ValueError("qpos width does not match source names")
    return values[:, _qpos_indices(source_names, target_names)].copy()


def validate_adaptation_model(model: mujoco.MjModel, robot: RobotConfig) -> None:
    """Validate the existing terrain-raycast and end-effector geometry contract."""
    root_body = model.body(robot.root_body).id
    tree_root = model.body_rootid[root_body]
    robot_geoms = model.body_rootid[model.geom_bodyid] == tree_root
    if np.any(robot_geoms & (model.geom_group == 0)):
        raise ValueError("robot geometry must not use terrain raycast group 0")

    collision_geoms = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    terrain_collisions = ~robot_geoms & collision_geoms
    if np.any(terrain_collisions & (model.geom_group != 0)):
        raise ValueError("terrain collision geometry must use group 0")
    if not np.any(~robot_geoms & (model.geom_group == 0)):
        raise ValueError("terrain geometry is required in raycast group 0")

    for name in robot.end_effectors:
        body_id = model.body(name).id
        capsules = (
            (model.geom_bodyid == body_id)
            & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE))
            & collision_geoms
        )
        if not np.any(capsules):
            raise ValueError(
                f"end-effector {name!r} requires capsule collision geometry directly on its body"
            )
