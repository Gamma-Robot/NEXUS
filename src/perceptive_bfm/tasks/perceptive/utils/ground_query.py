"""Runtime terrain-height query under arbitrary world (x, y) points.

Needed because the perceptive task's reference contact schedule and its dense
foot/hand clearance rewards are both *ground-relative* quantities on rough
terrain -- an absolute-z threshold silently misreads any EE standing on a raised
step (see ``docs/superpowers/specs/2026-07-31-ee-contact-reward-design.md``).

This is the runtime counterpart of ``utils/terrain_motion/keyframe_ik.py``'s
offline ``query_terrain_height_normal``. It cannot reuse that one (single-env
``mujoco.mj_ray`` on a plain ``MjModel``) nor the ``GroupedRayCaster`` sensor
(one fixed attachment body + a fixed relative ray pattern, whereas the four EEs
move independently), so it calls ``mujoco_warp.rays`` directly.

Two conventions differ from the offline path and are easy to get wrong:

* ``rays``' ``geomgroup`` is a ``vec6`` where **-1 means include and 0 means
  exclude** -- the inverse of the ``mj_ray`` numpy mask that
  ``terrain_only_geomgroup()`` builds.
* Which group holds the terrain is **not** assumed to be 0 (that is only true of
  G1's hand-written scene MJCFs). At runtime the terrain is emitted by mjlab's
  terrain generator, so the groups are resolved by elimination against the
  robot's own geoms -- see ``resolve_terrain_geom_groups``.

Group filtering -- not ``bodyexclude`` -- is what keeps the robot out of the
raycast: ``bodyexclude`` drops geoms on one *single* body id, not the whole
kinematic tree, a trap already hit once offline (``keyframe_ik.py:49``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import warp as wp
from mujoco_warp import rays

# mujoco_warp's vec6 geomgroup mask: -1 = include this group, 0 = exclude it.
_GEOMGROUP_INCLUDE = -1
_GEOMGROUP_EXCLUDE = 0


def resolve_ground_z(
    origin_z: float,
    dist: torch.Tensor,
    normal_z: torch.Tensor,
    prev: torch.Tensor,
    ground_normal_min_z: float = 0.5,
) -> torch.Tensor:
    """Turn raw downward-raycast output into a usable ground height per ray.

    Args:
      origin_z: world z every ray was fired from, straight down.
      dist: (B, nray) hit distance; negative means the ray hit nothing.
      normal_z: (B, nray) z component of the hit surface normal.
      prev: (B, nray) last known good ground height, held on a rejected ray.
      ground_normal_min_z: reject hits flatter than this in z -- a stair riser is
        a wall, not the ground under the foot. Matches the offline
        ``KeyframeIKCfg.ground_normal_min_z`` semantics.

    Returns:
      (B, nray) ground height, falling back to ``prev`` wherever the ray missed
      or landed on a wall. Holding the previous value (rather than substituting
      0.0) keeps a momentary miss from injecting a fake ground plane at the
      world origin, which would read as a huge sudden clearance.
    """
    valid = (dist >= 0.0) & (normal_z >= ground_normal_min_z)
    return torch.where(valid, origin_z - dist, prev)


def resolve_terrain_geom_groups(mj_model, robot_geom_ids: Iterable) -> tuple[int, ...]:
    """Geom groups that contain terrain, resolved by elimination rather than assumed.

    Everything that is not one of the robot's own geoms is treated as terrain
    (the perceptive scene holds only the robot and the terrain). Returns the
    sorted set of groups those geoms live in.

    ``robot_geom_ids`` is coerced element-wise to ``int`` on purpose: mjlab hands
    back ``robot.indexing.geom_ids`` as a CUDA int32 *tensor*, and testing a plain
    Python geom id for membership in a set of 0-dim tensors silently never matches
    -- which classifies every robot geom as terrain and lets the "terrain" raycast
    hit the robot's own legs.

    Raises:
      ValueError: if no non-robot geom exists, which would silently produce a
        raycast that can never hit anything.
    """
    robot_geoms = {int(gid) for gid in robot_geom_ids}
    groups = {
        int(mj_model.geom_group[gid]) for gid in range(mj_model.ngeom) if gid not in robot_geoms
    }
    groups = {g for g in groups if 0 <= g <= 5}
    if not groups:
        raise ValueError(
            "No non-robot geom found in the scene, so terrain raycasts could never "
            "hit anything. Check that the scene actually has terrain geoms and that "
            "robot_geom_ids is the robot's own geom id list."
        )
    return tuple(sorted(groups))


def _geom_groups_to_vec6(groups: tuple[int, ...]):
    mask = [_GEOMGROUP_EXCLUDE] * 6
    for group in groups:
        if 0 <= group <= 5:
            mask[group] = _GEOMGROUP_INCLUDE
    return wp.types.vector(length=6, dtype=wp.int32)(*mask)


class EEGroundRaycaster:
    """Fixed-width batched downward raycaster for a handful of moving world points.

    Buffers are allocated once at exactly ``(num_envs, num_rays)`` so they stay
    contiguous -- ``wp.from_torch`` on a non-contiguous slice would silently
    hand the kernel a *copy* and the results would never reach the caller.

    ``num_rays`` is the only free dimension: ``mujoco_warp.rays`` launches with
    ``dim=(d.nworld, pnt.shape[1])``, so every call necessarily covers all envs.
    """

    def __init__(
        self,
        num_envs: int,
        num_rays: int,
        device: str,
        terrain_geom_groups: tuple[int, ...],
        origin_z: float = 10.0,
        ground_normal_min_z: float = 0.5,
    ):
        self.num_envs = num_envs
        self.num_rays = num_rays
        self.origin_z = origin_z
        self.ground_normal_min_z = ground_normal_min_z
        self._geomgroup = _geom_groups_to_vec6(terrain_geom_groups)
        # Warp launches on its *current* device, not on the device of the arrays it is
        # handed, so every launch must be scoped -- see query().
        self._wp_device = wp.get_device(device)

        self._pnt = torch.zeros(num_envs, num_rays, 3, device=device)
        self._pnt[..., 2] = origin_z
        self._vec = torch.zeros(num_envs, num_rays, 3, device=device)
        self._vec[..., 2] = -1.0
        self._dist = torch.zeros(num_envs, num_rays, device=device)
        self._geomid = torch.zeros(num_envs, num_rays, dtype=torch.int32, device=device)
        self._normal = torch.zeros(num_envs, num_rays, 3, device=device)
        self._bodyexclude = torch.full((num_rays,), -1, dtype=torch.int32, device=device)

    def query(self, wp_model, wp_data, xy: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Ground height under each ``xy`` point. ``xy``: (num_envs, num_rays, 2).

        The ``wp.ScopedDevice`` is load-bearing, not defensive: ``rays`` launches on
        warp's *current* device rather than the device of the arrays passed in. With a
        single GPU the two coincide (``CUDA_VISIBLE_DEVICES`` makes the env device
        ``cuda:0``, which is also warp's default) and an unscoped call happens to work.
        On rank 1 of a multi-GPU run the env is on ``cuda:1`` while warp still defaults
        to ``cuda:0``, so the kernel launches in the wrong context against ``cuda:1``
        memory and dies with "CUDA error 700: an illegal memory access". Rank 1 then
        never joins the process group and rank 0 times out ~10 min later, surfacing as
        a torchrunx ``AgentFailedError`` far from the real fault.

        Same pattern as every other warp launch in mjlab (``sim.py``,
        ``sensor_context.py``) and in this repo (``grouped_ray_caster_camera.py``).
        """
        self._pnt[..., :2] = xy
        self._dist.fill_(-1.0)
        self._geomid.fill_(-1)
        self._normal.zero_()

        with wp.ScopedDevice(self._wp_device):
            rays(
                m=wp_model,
                d=wp_data,
                pnt=wp.from_torch(self._pnt, dtype=wp.vec3),
                vec=wp.from_torch(self._vec, dtype=wp.vec3),
                geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
                flg_static=True,
                bodyexclude=wp.from_torch(self._bodyexclude, dtype=wp.int32),
                dist=wp.from_torch(self._dist, dtype=wp.float32),
                geomid=wp.from_torch(self._geomid, dtype=wp.int32),
                normal=wp.from_torch(self._normal, dtype=wp.vec3),
            )

        return resolve_ground_z(
            origin_z=self.origin_z,
            dist=self._dist,
            normal_z=self._normal[..., 2],
            prev=prev,
            ground_normal_min_z=self.ground_normal_min_z,
        )
