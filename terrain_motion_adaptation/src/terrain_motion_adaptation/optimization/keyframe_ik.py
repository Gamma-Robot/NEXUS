"""Terrain keyframe IK using geometric two-bone targets and bounded joint solves.

Each keyframe is solved independently without cross-frame state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation

DEFAULT_MIN_KNEE_BEND = math.radians(2)
DEFAULT_MAX_KNEE_BEND = math.radians(145)


@dataclass(frozen=True)
class KeyframeIKCfg:
    ground_normal_min_z: float = 0.18
    ground_align_min_z: float = 0.35
    enable_penetration_correction: bool = True
    penetration_clearance: float = 0.0
    # Raycast origin z (absolute world height, NOT an offset from the queried
    # EE's own z): the source motion is flat, so a foot's z sits near 0 --
    # raycasting down from there would miss terrain raised above z=0, since the
    # ray origin would already be embedded below the surface. Must be a fixed
    # height above any terrain feature in the patch.
    raycast_origin_z: float = 5.0
    # Orientation-alignment cap (feet only, see ChainSpec.orient_joint_names):
    # how far the foot may tilt to match the terrain normal, so a steep normal
    # near e.g. a step edge can't twist the ankle to an unrealistic angle.
    max_tilt_angle: float = math.radians(40)
    # Extra clearance added on top of the sole/palm sample correction (see
    # solve_ik_for_clip). Zero adds no clearance beyond the sampled surface.
    sole_correction_clearance: float = 0.0
    # Mid-link (knee/elbow) penetration correction -- see _swivel_mid_clear.
    # The chain solve places the EE on terrain and lets the mid-link land
    # wherever the two-bone solution puts it, which on displaced terrain drives
    # knees and elbows into the ground (measured ~3.5-5x worse than the flat
    # source, while the EEs it does constrain got ~3x better).
    enable_mid_link_correction: bool = True
    mid_link_clearance: float = 0.0
    # Penetration shallower than this is left alone -- the common case then
    # costs only the clearance raycasts, no re-solve.
    mid_link_tolerance: float = 0.005
    # Safety cap, in the same spirit as max_tilt_angle: an unbounded swivel can
    # rotate a knee into a sideways bend that is geometrically valid and
    # anatomically wrong.
    max_swivel_angle: float = math.radians(45)
    swivel_search_steps: int = 12
    max_swivel_solves: int = 3
    # How far the end-effector may move as a side effect of a swivel attempt.
    # solve_two_bone_ik's swivel is endpoint-exact, but the numeric chain solve
    # that realizes it balances mid-position against end-position and posture,
    # so a large mid-target change can drag the EE off the terrain contact the
    # earlier passes solved for (observed: a wrist 2cm deeper into a stair).
    # An attempt that exceeds this is rejected even if the mid-link cleared.
    max_swivel_ee_drift: float = 0.005
    # The same budget for a limb that is NOT in contact this frame. A swinging
    # limb's EE has no terrain contact to preserve -- only the requirement that
    # it not penetrate, which is checked separately -- so pinning it as tightly
    # as a planted foot needlessly starves the correction.
    max_swivel_ee_drift_free: float = 0.05


def terrain_only_geomgroup(terrain_group: int = 0) -> np.ndarray:
    """Return an ``mj_ray`` mask selecting only ``terrain_group``.

    Robot geoms must use other groups: ``bodyexclude`` excludes only one body,
    not its entire kinematic subtree. The solver uses terrain group 0; the
    supplied G1 model uses groups 2 and 3 for robot visuals and collisions.
    """
    mask = np.zeros(6, dtype=np.uint8)
    mask[terrain_group] = 1
    return mask


def query_terrain_height_normal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    geomgroup: np.ndarray | None = None,
    cfg: KeyframeIKCfg = KeyframeIKCfg(),
) -> tuple[float, np.ndarray] | None:
    """Terrain height + world-space normal at (x, y), via a fixed-height-origin
    downward raycast (see ``KeyframeIKCfg.raycast_origin_z``).

    Returns ``None`` if there is no hit, or the nearest hit's normal.z is below
    ``cfg.ground_normal_min_z`` (treated as a wall, not ground).
    ``geomgroup``: restrict which geoms can be hit -- pass
    ``terrain_only_geomgroup()`` (the default here) to exclude the robot via
    its geom group rather than ``mj_ray``'s per-body ``bodyexclude`` (which
    does not exclude a whole kinematic tree, only one body).
    """
    if geomgroup is None:
        geomgroup = terrain_only_geomgroup()
    pnt = np.array([x, y, cfg.raycast_origin_z], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    normal = np.zeros(3, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, geomid, normal)
    if dist < 0:
        return None
    if normal[2] < cfg.ground_normal_min_z:
        return None
    terrain_z = cfg.raycast_origin_z - dist
    return terrain_z, normal


def solve_chain_joint_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chain_joint_names: tuple[str, ...],
    mid_body_name: str,
    end_body_name: str,
    base_qpos: np.ndarray,
    initial_guess: np.ndarray,
    target_mid_pos: np.ndarray,
    target_end_pos: np.ndarray,
    posture_target: np.ndarray | None = None,
    posture_weight: float = 0.08,
) -> np.ndarray:
    """Numerically solve this chain's joint angles so FK places ``mid_body_name``
    at ``target_mid_pos`` and ``end_body_name`` at ``target_end_pos``.

    Initial angles outside joint limits by at most 1e-4 radians are clipped
    to absorb motion quantization error. Larger violations and non-finite
    angles raise ``ValueError``.

    Uses the compiled model's FK to account for fixed rotations between joint
    frames. A direct XYZ Euler decomposition would ignore these rotations.

    A 4-DOF chain (e.g. hip pitch/roll/yaw + knee) solving a 2-body position
    target (6 residuals) is not automatically well-conditioned just because it
    looks overdetermined by DOF count: the knee-position target alone already
    fixes the 3 hip DOF (generically), and that 3-DOF-to-3-target sub-problem
    has multiple branches (like Euler-angle IK), some wildly different in
    joint-angle space but equally valid in position. ``initial_guess`` alone
    does NOT prevent this -- it is only the optimizer's starting point, not a
    tether; scipy's trust-region solve is free to cross into a different
    branch's basin. ``posture_target`` (typically the frame's source chain angles;
    pass ``None`` to skip regularization entirely) adds a genuine per-iteration
    pull back toward a known-good branch via an extra residual block.
    """
    qpos_addrs = [model.joint(name).qposadr[0] for name in chain_joint_names]
    mid_id = model.body(mid_body_name).id
    end_id = model.body(end_body_name).id
    lower = np.array([model.joint(name).range[0] for name in chain_joint_names])
    upper = np.array([model.joint(name).range[1] for name in chain_joint_names])
    initial_guess = np.asarray(initial_guess, dtype=np.float64)
    if initial_guess.shape != lower.shape:
        raise ValueError("initial_guess must contain one angle per chain joint")
    # Float32 motion storage can round a limit outward. Check before clipping
    # so invalid source poses are not silently accepted.
    bounds_tolerance = 1e-4  # radians, absolute (no relative tolerance)
    invalid = (
        ~np.isfinite(initial_guess)
        | (initial_guess < lower - bounds_tolerance)
        | (initial_guess > upper + bounds_tolerance)
    )
    if np.any(invalid):
        names = ", ".join(name for name, bad in zip(chain_joint_names, invalid) if bad)
        raise ValueError(
            "Initial guess is outside of provided bounds "
            f"(tolerance {bounds_tolerance:g} rad) or non-finite for joints: {names}"
        )
    initial_guess = np.clip(initial_guess, lower, upper)

    def residual(angles: np.ndarray) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[qpos_addrs] = angles
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pos_res = np.concatenate(
            [data.xpos[mid_id] - target_mid_pos, data.xpos[end_id] - target_end_pos]
        )
        if posture_target is None:
            return pos_res
        return np.concatenate([pos_res, posture_weight * (angles - posture_target)])

    result = scipy.optimize.least_squares(
        residual, initial_guess, bounds=(lower, upper), method="trf"
    )
    return result.x


def sole_sample_offsets(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    """Body-local ground-contact sample points for ``body_name``'s foot/hand,
    one pair per capsule collision geom attached to it -- each capsule's two
    endpoints, offset by its own radius along the body's local -z ("down").

    This reaches the actual sole/palm surface instead of the ankle/wrist joint
    origin: G1 models a foot's sole as several parallel capsules
    (``left_footN_collision``) and a hand as one (``left_hand_collision``);
    the joint origin sits well above where any of them actually touch the
    ground. A capsule's length axis is its local +z (MuJoCo's ``fromto``
    compiles to this convention), which is why extracting the endpoints needs
    ``geom_quat``, not just ``geom_pos``/``geom_size``.
    """
    body_id = model.body(body_name).id
    offsets = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        pos = model.geom_pos[g].copy()
        quat_xyzw = model.geom_quat[g][[1, 2, 3, 0]]
        radius, half_length = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        axis = Rotation.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
        for sign in (-1.0, 1.0):
            endpoint = pos + sign * half_length * axis
            endpoint = endpoint.copy()
            endpoint[2] -= radius
            offsets.append(endpoint)
    if not offsets:
        raise ValueError(f"no capsule collision geoms found on body {body_name!r}")
    return np.array(offsets)


def capsule_endpoints(model: mujoco.MjModel, body_name: str) -> list[tuple[np.ndarray, float]]:
    """``(body-local endpoint, radius)`` for both ends of every capsule collision
    geom on ``body_name``. Empty list if it has none.

    Distinct from ``sole_sample_offsets``, which bakes the radius into an offset
    along the body's local **-z**: that encodes "this body has a sole facing
    down", which is true of a foot and meaningless for a knee or elbow. Here the
    raw endpoints are returned so the caller can find the capsule's lowest point
    in the **world** frame, whatever the body's current orientation.
    """
    body_id = model.body(body_name).id
    endpoints: list[tuple[np.ndarray, float]] = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        pos = model.geom_pos[g].copy()
        axis = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        radius, half_length = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        for sign in (-1.0, 1.0):
            endpoints.append((pos + sign * half_length * axis, radius))
    return endpoints


def min_capsule_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    cfg: KeyframeIKCfg = KeyframeIKCfg(),
) -> float:
    """Smallest gap between ``body_name``'s collision capsules and the terrain
    beneath them, given its CURRENT pose (already forwarded into ``data``).
    Negative means penetrating.

    A capsule's lowest point is ``min(endpoint_z) - radius``, so each endpoint is
    tested at its own xy. Returns ``inf`` when the body carries no capsules or no
    ray hits terrain -- "unmeasurable, therefore nothing to correct" -- which lets
    callers query any ``mid_body`` without special-casing.
    """
    endpoints = capsule_endpoints(model, body_name)
    if not endpoints:
        return math.inf

    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    rot = Rotation.from_quat(data.xquat[body_id][[1, 2, 3, 0]])
    terrain_geomgroup = terrain_only_geomgroup()

    clearance = math.inf
    for local, radius in endpoints:
        world = pos + rot.apply(local)
        hit = query_terrain_height_normal(
            model, data, world[0], world[1], geomgroup=terrain_geomgroup, cfg=cfg
        )
        if hit is None:
            continue
        terrain_z, _normal = hit
        clearance = min(clearance, (world[2] - radius) - terrain_z)
    return clearance


def rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> Rotation:
    """The shortest rotation that maps unit-ish vector ``a`` onto ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return Rotation.identity()
    if dot < -1.0 + 1e-8:
        orth = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, orth)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    return Rotation.from_rotvec(axis * angle)


def cap_rotation_angle(rot: Rotation, max_angle: float) -> Rotation:
    """``rot`` unchanged if its angle is within ``max_angle``, else scaled down
    to exactly ``max_angle`` (same axis)."""
    angle = rot.magnitude()
    if angle <= max_angle or angle < 1e-9:
        return rot
    return Rotation.from_rotvec(rot.as_rotvec() * (max_angle / angle))


def solve_orientation_align(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    orient_joint_names: tuple[str, ...],
    body_name: str,
    base_qpos: np.ndarray,
    initial_guess: np.ndarray,
    target_world_quat: np.ndarray,
) -> np.ndarray:
    """Numerically solve ``orient_joint_names`` (e.g. ankle_pitch/roll, or
    wrist_roll/pitch/yaw) so ``body_name``'s world orientation matches
    ``target_world_quat`` (wxyz) -- the same mujoco-FK-as-ground-truth approach
    as ``solve_chain_joint_angles``, but for orientation instead of position.
    """
    qpos_addrs = [model.joint(name).qposadr[0] for name in orient_joint_names]
    body_id = model.body(body_name).id
    lower = np.array([model.joint(name).range[0] for name in orient_joint_names])
    upper = np.array([model.joint(name).range[1] for name in orient_joint_names])
    initial_guess = np.clip(initial_guess, lower, upper)
    target_rot = Rotation.from_quat(target_world_quat[[1, 2, 3, 0]])

    def residual(angles: np.ndarray) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[qpos_addrs] = angles
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        current_quat = data.xquat[body_id]
        current_rot = Rotation.from_quat(current_quat[[1, 2, 3, 0]])
        error_rot = target_rot * current_rot.inv()
        return error_rot.as_rotvec()

    result = scipy.optimize.least_squares(
        residual, initial_guess, bounds=(lower, upper), method="trf"
    )
    return result.x


def sole_contact_correction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    end_body_name: str,
    clearance: float,
    cfg: KeyframeIKCfg = KeyframeIKCfg(),
) -> float:
    """Given ``end_body_name``'s CURRENT solved pose (already forwarded into
    ``data``), sample its sole/palm points (``sole_sample_offsets``), raycast
    each against the terrain, and return the extra vertical lift needed to
    clear the worst (most-penetrating) sample by ``clearance`` -- 0.0 if none
    penetrate (never negative: a floating sample needs no correction).

    Runs after orientation alignment because rotating a foot moves its
    sole points -- a single ankle-origin raycast can't see that.
    """
    offsets = sole_sample_offsets(model, end_body_name)
    body_id = model.body(end_body_name).id
    pos = data.xpos[body_id].copy()
    quat_xyzw = data.xquat[body_id][[1, 2, 3, 0]]
    rot = Rotation.from_quat(quat_xyzw)
    world_samples = pos + rot.apply(offsets)

    terrain_geomgroup = terrain_only_geomgroup()
    max_needed = 0.0
    for sample in world_samples:
        hit = query_terrain_height_normal(
            model, data, sample[0], sample[1], geomgroup=terrain_geomgroup, cfg=cfg
        )
        if hit is None:
            continue
        terrain_z, _normal = hit
        needed = (terrain_z + clearance) - sample[2]
        if needed > max_needed:
            max_needed = needed
    return max_needed


@dataclass(frozen=True)
class TwoBoneIKResult:
    desired_knee: np.ndarray  # (3,) world position
    clamped_target: np.ndarray  # (3,) world position, target after reach-clamping
    hip_to_knee_dir: np.ndarray  # (3,) unit vector
    knee_to_target_dir: np.ndarray  # (3,) unit vector


def _reach_from_bend(upper_len: float, lower_len: float, bend: float) -> float:
    """Hip-to-foot distance when the two bones meet at angle ``bend``."""
    return math.sqrt(upper_len**2 + lower_len**2 + 2 * upper_len * lower_len * math.cos(bend))


def _rest_pole(hip: np.ndarray, source_knee: np.ndarray, source_foot: np.ndarray) -> np.ndarray:
    """The source pose's own knee-bend direction, projected orthogonal to the
    source hip-to-foot chain direction.
    """
    rest_chain = source_foot - hip
    rest_chain_norm = np.linalg.norm(rest_chain)
    if rest_chain_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    rest_chain = rest_chain / rest_chain_norm

    knee_rel = source_knee - hip
    pole = knee_rel - rest_chain * np.dot(knee_rel, rest_chain)
    pole_norm = np.linalg.norm(pole)
    if pole_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return pole / pole_norm


def solve_two_bone_ik(
    hip: np.ndarray,
    source_knee: np.ndarray,
    source_foot: np.ndarray,
    target: np.ndarray,
    min_bend: float = DEFAULT_MIN_KNEE_BEND,
    max_bend: float = DEFAULT_MAX_KNEE_BEND,
    swivel: float = 0.0,
) -> TwoBoneIKResult:
    """Analytic two-bone IK: where should the knee/elbow go so the chain end
    reaches (a reach-clamped version of) ``target``?

    Bone lengths are measured from the source pose (``hip``/``source_knee``/
    ``source_foot``), not stored rest lengths. The knee-bend plane (pole) is
    taken from the source pose's own bend direction -- there is no "previous
    frame" pole to fall back on since keyframes are solved independently.

    ``swivel`` rotates that bend plane about the hip->target axis by the given
    angle (right-hand rule), which moves the knee/elbow while leaving BOTH
    endpoints exactly where they are -- with hip and end fixed, it is the
    chain's only remaining degree of freedom. That is what makes it usable to
    lift a penetrating mid-link without disturbing the terrain contact the
    caller already solved for (see ``_swivel_mid_clear``). ``swivel=0.0`` is
    bit-identical to omitting it.
    """
    upper_len = max(1e-4, float(np.linalg.norm(source_knee - hip)))
    lower_len = max(1e-4, float(np.linalg.norm(source_foot - source_knee)))

    hip_to_target = target - hip
    target_distance = float(np.linalg.norm(hip_to_target))
    if target_distance < 1e-4:
        raise ValueError("target coincides with hip -- degenerate two-bone IK query")

    max_reach = _reach_from_bend(upper_len, lower_len, min_bend)
    min_reach = _reach_from_bend(upper_len, lower_len, max_bend)
    clamped_distance = min(max(target_distance, min_reach), max_reach)
    clamped_target = hip + hip_to_target * (clamped_distance / target_distance)

    chain_dir = (clamped_target - hip) / clamped_distance
    pole = _rest_pole(hip, source_knee, source_foot)
    # Remove any component of pole along chain_dir (pole must be orthogonal to
    # the chain to act as a bend-plane normal).
    pole = pole - chain_dir * np.dot(pole, chain_dir)
    pole_norm = np.linalg.norm(pole)
    if pole_norm < 1e-6:
        pole = np.array([0.0, 0.0, 1.0]) - chain_dir * chain_dir[2]
        pole_norm = np.linalg.norm(pole)
        if pole_norm == 0.0:
            # World Z is parallel to a vertical chain. Use a deterministic
            # Cartesian axis least aligned with it to define a bend plane.
            axis = np.zeros(3)
            axis[np.argmin(np.abs(chain_dir))] = 1.0
            pole = axis - chain_dir * np.dot(axis, chain_dir)
            pole_norm = np.linalg.norm(pole)
    pole = pole / pole_norm

    if swivel != 0.0:
        pole = Rotation.from_rotvec(chain_dir * swivel).apply(pole)

    along = (upper_len**2 - lower_len**2 + clamped_distance**2) / (2 * clamped_distance)
    height = math.sqrt(max(0.0, upper_len**2 - along**2))
    desired_knee = hip + chain_dir * along + pole * height

    hip_to_knee_dir = (desired_knee - hip) / upper_len
    knee_to_target_dir = (clamped_target - desired_knee) / lower_len

    return TwoBoneIKResult(
        desired_knee=desired_knee,
        clamped_target=clamped_target,
        hip_to_knee_dir=hip_to_knee_dir,
        knee_to_target_dir=knee_to_target_dir,
    )


def aggregate_root_height_offset(active_chain_offsets: np.ndarray) -> float:
    """Return the active chain offset with greatest magnitude, preserving its sign."""
    if active_chain_offsets.size == 0:
        return 0.0
    idx = int(np.argmax(np.abs(active_chain_offsets)))
    return float(active_chain_offsets[idx])


@dataclass(frozen=True)
class ChainSpec:
    hip_body: str  # chain root: hip (legs) or shoulder (arms) body
    mid_body: str  # knee or elbow body
    end_body: str  # the EE itself (ankle_roll/wrist_yaw link)
    chain_joint_names: tuple[str, ...]  # 4 DOF: hip/shoulder x3 + knee/elbow x1
    # DOF beyond chain_joint_names used to align end_body's orientation to the
    # terrain normal (ankle_pitch/roll for feet). Empty for hands: a hand is a
    # rigid capsule extending past the wrist, not a flat sole -- there's no
    # orientation that makes it "lie flat", so it only gets the sole-sample
    # position correction (solve_ik_for_clip skips alignment when this is empty).
    orient_joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class IKClipResult:
    frame_indices: np.ndarray  # (K,) source frame indices that were processed
    joint_pos: np.ndarray  # (K, dof) adjusted joint angles, dof order == joint_names
    root_pos_w: np.ndarray  # (K, 3) adjusted root position (terrain-placed frame)


def _select_frames(
    contact_mask: np.ndarray, frame_selection: Literal["sparse", "dense"]
) -> np.ndarray:
    if frame_selection == "dense":
        return np.arange(contact_mask.shape[0])
    return np.nonzero(contact_mask.any(axis=1))[0]


def _solve_ee_to_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: ChainSpec,
    adjusted_qpos: np.ndarray,
    source_hip: np.ndarray,
    source_mid: np.ndarray,
    source_end: np.ndarray,
    target_pos: np.ndarray,
    posture_target: np.ndarray,
    swivel: float = 0.0,
) -> None:
    """Solve ``spec.chain_joint_names`` so ``spec.end_body`` reaches
    ``target_pos``. ``source_hip``/``source_mid``/``source_end`` are the bone-
    length/pole-vector reference for ``solve_two_bone_ik`` -- always the
    frame's true source pose (fixed once by the caller), never re-derived from
    an intermediate solved state (see ``solve_ik_for_clip``'s docstring on why
    that drifts). ``posture_target`` (the same frame's fixed source chain
    angles) is the branch-disambiguation anchor for ``solve_chain_joint_angles``
    -- deliberately NOT re-derived from ``adjusted_qpos`` either, for the same
    reason: re-anchoring to a value a previous step may have already knocked
    onto the wrong branch would just reinforce it instead of pulling back to
    the source's own branch. Mutates ``adjusted_qpos`` and ``data`` in place.
    """
    two_bone = solve_two_bone_ik(source_hip, source_mid, source_end, target_pos, swivel=swivel)
    initial_guess = np.array(
        [adjusted_qpos[model.joint(n).qposadr[0]] for n in spec.chain_joint_names]
    )
    solved = solve_chain_joint_angles(
        model,
        data,
        chain_joint_names=spec.chain_joint_names,
        mid_body_name=spec.mid_body,
        end_body_name=spec.end_body,
        base_qpos=adjusted_qpos,
        initial_guess=initial_guess,
        target_mid_pos=two_bone.desired_knee,
        target_end_pos=two_bone.clamped_target,
        posture_target=posture_target,
    )
    for name, angle in zip(spec.chain_joint_names, solved):
        adjusted_qpos[model.joint(name).qposadr[0]] = angle
    data.qpos[:] = adjusted_qpos
    mujoco.mj_forward(model, data)


def _swivel_mid_clear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: ChainSpec,
    adjusted_qpos: np.ndarray,
    source_hip: np.ndarray,
    source_mid: np.ndarray,
    source_end: np.ndarray,
    target_pos: np.ndarray,
    posture_target: np.ndarray,
    cfg: KeyframeIKCfg,
    max_ee_drift: float,
) -> None:
    """Lift a penetrating knee/elbow clear of the terrain by rotating the chain's
    bend plane about the hip->EE axis, leaving both endpoints untouched.

    Runs after the position/orientation/sole passes have fixed the end-effector.
    Swivel is the chain's only remaining freedom once hip and end are pinned, so
    this is the one correction that cannot trade away the terrain contact those
    passes just solved for (see ``solve_two_bone_ik``'s ``swivel``).

    Two stages, because each real attempt costs a ``least_squares``: candidates
    are ranked analytically by the z gain of ``desired_knee`` (cheap, no solver),
    but acceptance is always the true re-measured capsule clearance -- the
    analytic knee is only the body *origin*, while penetration also depends on
    the body's solved orientation. If no attempt clears, the un-swivelled solve
    is restored: a half-rotated pose that still penetrates has paid posture
    distortion for nothing.
    """
    clearance = min_capsule_clearance(model, data, spec.mid_body, cfg)
    if clearance >= cfg.mid_link_clearance - cfg.mid_link_tolerance:
        return
    needed = cfg.mid_link_clearance - clearance

    baseline_qpos = adjusted_qpos.copy()
    baseline_end = data.xpos[model.body(spec.end_body).id].copy()
    baseline_end_lift = sole_contact_correction(model, data, spec.end_body, 0.0, cfg)
    baseline_knee_z = solve_two_bone_ik(
        source_hip, source_mid, source_end, target_pos
    ).desired_knee[2]

    magnitudes = np.linspace(0.0, cfg.max_swivel_angle, cfg.swivel_search_steps + 1)[1:]
    candidates = [a for m in magnitudes for a in (m, -m)]
    promising = [
        angle
        for angle in candidates
        if solve_two_bone_ik(
            source_hip, source_mid, source_end, target_pos, swivel=angle
        ).desired_knee[2]
        - baseline_knee_z
        >= needed
    ]

    for angle in promising[: cfg.max_swivel_solves]:
        _solve_ee_to_position(
            model,
            data,
            spec,
            adjusted_qpos,
            source_hip,
            source_mid,
            source_end,
            target_pos,
            posture_target,
            swivel=angle,
        )
        cleared = min_capsule_clearance(model, data, spec.mid_body, cfg) >= (
            cfg.mid_link_clearance - cfg.mid_link_tolerance
        )
        drift = float(np.linalg.norm(data.xpos[model.body(spec.end_body).id] - baseline_end))
        # Moving a free limb's EE is allowed, driving it into the ground is not.
        end_lift = sole_contact_correction(model, data, spec.end_body, 0.0, cfg)
        if cleared and drift <= max_ee_drift and end_lift <= baseline_end_lift + 1e-6:
            return

    adjusted_qpos[:] = baseline_qpos
    data.qpos[:] = adjusted_qpos
    mujoco.mj_forward(model, data)


def _align_ee_orientation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: ChainSpec,
    adjusted_qpos: np.ndarray,
    target_xy: np.ndarray,
    terrain_geomgroup: np.ndarray,
    cfg: KeyframeIKCfg,
) -> None:
    """Feet only (no-op if ``spec.orient_joint_names`` is empty, e.g. hands):
    tilt ``spec.end_body`` to match the terrain normal at ``target_xy``,
    capped at ``cfg.max_tilt_angle``. Mutates ``adjusted_qpos``/``data``."""
    if not spec.orient_joint_names:
        return
    hit = query_terrain_height_normal(
        model, data, target_xy[0], target_xy[1], geomgroup=terrain_geomgroup, cfg=cfg
    )
    if hit is None:
        return
    _terrain_z, normal = hit
    if normal[2] < cfg.ground_align_min_z:
        return
    body_id = model.body(spec.end_body).id
    current_quat = data.xquat[body_id].copy()
    current_rot = Rotation.from_quat(current_quat[[1, 2, 3, 0]])
    current_up = current_rot.apply([0.0, 0.0, 1.0])
    align_delta = cap_rotation_angle(
        rotation_between_vectors(current_up, normal), cfg.max_tilt_angle
    )
    target_rot = align_delta * current_rot
    target_quat = target_rot.as_quat()[[3, 0, 1, 2]]
    orient_initial_guess = np.array(
        [adjusted_qpos[model.joint(n).qposadr[0]] for n in spec.orient_joint_names]
    )
    solved_orient = solve_orientation_align(
        model,
        data,
        spec.orient_joint_names,
        spec.end_body,
        adjusted_qpos,
        orient_initial_guess,
        target_quat,
    )
    for name, angle in zip(spec.orient_joint_names, solved_orient):
        adjusted_qpos[model.joint(name).qposadr[0]] = angle
    data.qpos[:] = adjusted_qpos
    mujoco.mj_forward(model, data)


def solve_ik_for_clip(
    model: mujoco.MjModel,
    dx: float,
    dy: float,
    dyaw: float,
    joint_names: list[str],
    joint_pos_seq: np.ndarray,
    root_pos_w_seq: np.ndarray,
    root_quat_w_seq: np.ndarray,
    contact_mask: np.ndarray,
    ee_names: list[str],
    chain_specs: dict[str, ChainSpec],
    frame_selection: Literal["sparse", "dense"] = "sparse",
    cfg: KeyframeIKCfg = KeyframeIKCfg(),
) -> IKClipResult:
    """Terrain-adapt the selected frames of a flat-source clip.

    ``dx, dy``: rigid xy translation placing the whole clip onto the terrain
    patch. IK proceeds directly in this placed frame, so raycasts hit the terrain
    wherever it actually is; the caller is responsible for shifting the output
    back to a scene-offset-free frame for storage, if desired.
    ``dyaw``: reserved for a future rotated placement; **must be 0.0 for now**
    -- there's no established convention yet for what point a clip should
    rotate about, and guessing one isn't worth the risk. Raises if nonzero.

    Reuses the supplied compiled model and creates local simulation data.
    Each selected frame uses MuJoCo FK and per-end-effector terrain raycasts;
    callers can reuse the model across clips without loading XML here.
    """
    if dyaw != 0.0:
        raise NotImplementedError(
            "dyaw rotation is not implemented yet -- no established convention "
            "for the rotation pivot; pass dyaw=0.0"
        )

    data = mujoco.MjData(model)
    terrain_geomgroup = terrain_only_geomgroup()
    joint_qpos_addr = np.array([model.joint(name).qposadr[0] for name in joint_names])
    offset_xy = np.array([dx, dy])

    frame_indices = _select_frames(contact_mask, frame_selection)
    out_joint_pos = np.zeros((len(frame_indices), joint_pos_seq.shape[1]), dtype=np.float64)
    out_root_pos_w = np.zeros((len(frame_indices), 3), dtype=np.float64)

    for out_idx, frame in enumerate(frame_indices):
        qpos = np.zeros(model.nq)
        qpos[0:2] = root_pos_w_seq[frame, 0:2] + offset_xy
        qpos[2] = root_pos_w_seq[frame, 2]
        qpos[3:7] = root_quat_w_seq[frame]  # wxyz, never adjusted
        qpos[joint_qpos_addr] = joint_pos_seq[frame]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        active = contact_mask[frame]
        adjusted_qpos = qpos.copy()

        # Pass 1: each active EE's target (raycast at its SOURCE (x, y), before
        # any root shift) and how much root height it alone would need.
        # Root height must be resolved *before* solving any chain -- solving a
        # chain first and only then shifting the root would rigidly drag that
        # already-solved foot away from the terrain by the shift amount.
        ee_targets: dict[str, np.ndarray] = {}
        needed_offsets = []
        for col, ee_name in enumerate(ee_names):
            if not active[col]:
                continue
            end = data.xpos[model.body(chain_specs[ee_name].end_body).id].copy()
            hit = query_terrain_height_normal(
                model, data, end[0], end[1], geomgroup=terrain_geomgroup, cfg=cfg
            )
            if hit is None:
                continue
            terrain_z, _normal = hit
            # Preserve the source end-effector's clearance above flat ground.
            # A joint origin can sit above its sole/palm contact surface;
            # targeting terrain_z alone would remove that clearance and alter
            # the limb pose even when adapting to a flat surface at z=0.
            ee_targets[ee_name] = np.array([end[0], end[1], terrain_z + end[2]])
            # Root height offset, by contrast, IS just terrain_z (no `+ end[2]`
            # term): this is a uniform whole-body shift, not a per-EE target,
            # so it should reflect "how far this contact point's local ground
            # differs from the z=0 ground the flat motion assumes" -- adding
            # `end[2]` here would double-count the clearance already handled
            # by ee_targets above.
            needed_offsets.append(terrain_z)

        adjusted_qpos[2] += aggregate_root_height_offset(np.array(needed_offsets))
        data.qpos[:] = adjusted_qpos
        mujoco.mj_forward(model, data)  # chain joints still source-valued; root is now placed

        # Pass 2: solve each active EE's chain against the *shifted* root.
        # Chains are kinematically independent (each attaches directly to the
        # pelvis/torso, none shares joints with another), so solving one
        # doesn't move another's hip/shoulder -- no re-forward needed between
        # them here.
        for ee_name, target in ee_targets.items():
            spec = chain_specs[ee_name]

            # Fixed ONCE per EE per frame, from the true (post-root-shift, but
            # pre-any-IK-for-this-chain) source pose. solve_two_bone_ik uses
            # these only to measure bone length and the knee/elbow bend-plane
            # (pole) direction -- re-deriving them from an already-adjusted
            # intermediate state (as an earlier version of this code did, by
            # re-reading "current" mid/end on every re-solve) lets the pole
            # drift across the position -> orient -> sole-correction passes,
            # which is numerically unstable for a near-extended leg (a small
            # perturbation can flip which side the knee bends to) and was
            # observed to make the knee visibly jump between frames.
            source_hip = data.xpos[model.body(spec.hip_body).id].copy()
            source_mid = data.xpos[model.body(spec.mid_body).id].copy()
            source_end = data.xpos[model.body(spec.end_body).id].copy()
            # Branch-disambiguation anchor for solve_chain_joint_angles (see
            # _solve_ee_to_position's docstring): fixed ONCE per EE per frame
            # from the true source pose, same rationale as source_hip/mid/end.
            source_chain_angles = np.array(
                [adjusted_qpos[model.joint(n).qposadr[0]] for n in spec.chain_joint_names]
            )

            # Step 1: position (ankle/wrist origin -> terrain height at its own xy).
            _solve_ee_to_position(
                model,
                data,
                spec,
                adjusted_qpos,
                source_hip,
                source_mid,
                source_end,
                target,
                source_chain_angles,
            )

            # Step 2: align orientation to the terrain normal.
            _align_ee_orientation(model, data, spec, adjusted_qpos, target, terrain_geomgroup, cfg)

            # Step 3: sole/palm-sample correction -- orientation (and, for
            # non-flat sole geometry, position too) can leave the actual
            # contact surface still penetrating or floating relative to the
            # single ankle/wrist-origin target above; re-solve position
            # against the worst sample.
            lift = sole_contact_correction(
                model, data, spec.end_body, cfg.sole_correction_clearance, cfg
            )
            if lift > 0.0:
                _solve_ee_to_position(
                    model,
                    data,
                    spec,
                    adjusted_qpos,
                    source_hip,
                    source_mid,
                    source_end,
                    target + np.array([0.0, 0.0, lift]),
                    source_chain_angles,
                )
                # Re-solving the hip/knee chain for this z-correction can
                # perturb the ankle's WORLD orientation (it depends on the
                # whole chain, not just the ankle's own joints), undoing some
                # of step 2's alignment -- re-lock it now that position is final.
                _align_ee_orientation(
                    model, data, spec, adjusted_qpos, target, terrain_geomgroup, cfg
                )

            # Step 4: lift the knee/elbow out of the terrain if the chain solve
            # above buried it. Endpoint-preserving, so it runs last and cannot
            # undo steps 1-3.
            if cfg.enable_mid_link_correction:
                _swivel_mid_clear(
                    model,
                    data,
                    spec,
                    adjusted_qpos,
                    source_hip,
                    source_mid,
                    source_end,
                    target,
                    source_chain_angles,
                    cfg,
                    max_ee_drift=cfg.max_swivel_ee_drift,
                )

        if cfg.enable_penetration_correction:
            data.qpos[:] = adjusted_qpos
            mujoco.mj_forward(model, data)
            for col, ee_name in enumerate(ee_names):
                if active[col]:
                    continue
                spec = chain_specs[ee_name]
                # Sole-sample based, like the active-EE correction above (not a
                # single ankle/wrist-origin check) -- a frozen limb can
                # penetrate at its toe/heel even when its origin is clear.
                lift = sole_contact_correction(
                    model, data, spec.end_body, cfg.penetration_clearance, cfg
                )
                frozen_hip = data.xpos[model.body(spec.hip_body).id].copy()
                frozen_mid = data.xpos[model.body(spec.mid_body).id].copy()
                frozen_end = data.xpos[model.body(spec.end_body).id].copy()
                frozen_chain_angles = np.array(
                    [adjusted_qpos[model.joint(n).qposadr[0]] for n in spec.chain_joint_names]
                )
                frozen_target = frozen_end + np.array([0.0, 0.0, max(lift, 0.0)])
                if lift > 0.0:
                    _solve_ee_to_position(
                        model,
                        data,
                        spec,
                        adjusted_qpos,
                        frozen_hip,
                        frozen_mid,
                        frozen_end,
                        frozen_target,
                        frozen_chain_angles,
                    )
                # A swinging limb's knee/elbow can be buried even when its own
                # sole is clear (the check above only sees the end body), so this
                # runs whether or not the lift above fired.
                if cfg.enable_mid_link_correction:
                    _swivel_mid_clear(
                        model,
                        data,
                        spec,
                        adjusted_qpos,
                        frozen_hip,
                        frozen_mid,
                        frozen_end,
                        frozen_target,
                        frozen_chain_angles,
                        cfg,
                        max_ee_drift=cfg.max_swivel_ee_drift_free,
                    )

        out_joint_pos[out_idx] = adjusted_qpos[joint_qpos_addr]
        out_root_pos_w[out_idx] = adjusted_qpos[0:3]

    return IKClipResult(
        frame_indices=frame_indices, joint_pos=out_joint_pos, root_pos_w=out_root_pos_w
    )
