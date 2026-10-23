"""Batched FK and finite-difference velocities from minimal motion windows.

Uses a depth-grouped kinematic tree and five-tap velocity smoothing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from mjlab.utils.lab_api.math import (
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

from perceptive_bfm.utils.motion import MotionData, MotionMinimalData


def _bcast_quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """``quat_mul`` with numpy-style broadcasting -- mjlab's version requires exact
    shape matches, so every broadcasting call site here
    (root-frame quats combined with per-body tree quats) must expand first."""
    shape = torch.broadcast_shapes(q1.shape, q2.shape)
    return quat_mul(q1.expand(shape), q2.expand(shape))


def _bcast_quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``quat_apply`` with numpy-style broadcasting -- see ``_bcast_quat_mul``."""
    batch_shape = torch.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    quat = quat.expand(batch_shape + quat.shape[-1:])
    vec = vec.expand(batch_shape + vec.shape[-1:])
    return quat_apply(quat, vec)


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _as_torch(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    import warp as wp  # type: ignore

    if isinstance(value, wp.array):  # type: ignore[arg-type]
        return wp.to_torch(value).to(device=device)
    return torch.as_tensor(value, device=device)


def _as_scalar_1d(value: Any, *, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 2:
        out = out[0]
    if out.ndim != 1:
        raise ValueError(f"Expected scalar field [N] or [E,N], got {tuple(out.shape)}")
    return out


def _as_vec_field(value: Any, *, dim: int, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[-1] != dim:
        raise ValueError(f"Expected vector field [N,{dim}] or [E,N,{dim}], got {tuple(out.shape)}")
    return out


def finite_diff_torch(x: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    x_t = x.movedim(dim, 0)
    vel = torch.zeros_like(x_t)
    if fps <= 0 or x_t.shape[0] < 2:
        return vel.movedim(0, dim)
    vel[1:-1] = (x_t[2:] - x_t[:-2]) * (fps / 2.0)
    vel[0] = (x_t[1] - x_t[0]) * fps
    vel[-1] = (x_t[-1] - x_t[-2]) * fps
    return vel.movedim(0, dim)


def angvel_from_quat_wxyz_torch(quat_wxyz: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    quat_t = normalize(quat_wxyz.movedim(dim, 0))
    if fps <= 0 or quat_t.shape[0] < 2:
        return torch.zeros(
            quat_t.shape[:-1] + (3,), dtype=quat_t.dtype, device=quat_t.device
        ).movedim(0, dim)

    flat = quat_t.reshape(quat_t.shape[0], -1, 4)
    dots = (flat[1:] * flat[:-1]).sum(dim=-1)
    signs = torch.where(dots < 0.0, -torch.ones_like(dots), torch.ones_like(dots))
    signs = torch.cat([torch.ones_like(signs[:1]), signs], dim=0)
    flat = flat * torch.cumprod(signs, dim=0).unsqueeze(-1)

    qdot = torch.zeros_like(flat)
    qdot[1:-1] = (flat[2:] - flat[:-2]) * (fps / 2.0)
    qdot[0] = (flat[1] - flat[0]) * fps
    qdot[-1] = (flat[-1] - flat[-2]) * fps

    omega = 2.0 * quat_mul(qdot, quat_conjugate(flat))[..., 1:]
    return omega.reshape(quat_t.shape[:-1] + (3,)).movedim(0, dim)


def smooth_avg5_torch(x: torch.Tensor, dim: int) -> torch.Tensor:
    """5-tap box-filter (uniform moving average) along ``dim``, with edge
    replication -- equivalent to ``F.conv1d`` against a ``ones(5)/5`` kernel over
    ``F.pad(..., mode="replicate")``, but done as plain slicing/addition instead.
    The packed backend calls this from a cache refill whose batch size (however
    many envs need a refill this step) changes almost every call, so the old
    ``F.conv1d`` version fed cuDNN a new input shape almost every time; with
    mjlab's ``configure_torch_backends`` enabling ``cudnn.benchmark=True``, every
    such shape triggered a fresh synchronous algorithm search -- profiling showed
    ``aten::cudnn_convolution`` alone accounting for ~90% of total training CPU
    time despite needing <1% of the actual GPU time. Elementwise ops have no such
    shape-keyed cache to miss."""
    x_t = x.movedim(dim, 0)
    if x_t.shape[0] < 2:
        return x
    padded = torch.cat([x_t[:1], x_t[:1], x_t, x_t[-1:], x_t[-1:]], dim=0)
    smooth = (padded[0:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
    return smooth.movedim(0, dim)


class MotionFKHelper:
    """Batched FK from ``joint_pos`` to per-body pose, scoped to a body subset."""

    def __init__(
        self,
        *,
        device: torch.device,
        base_body_id: int,
        tree_body_ids: torch.Tensor,
        parent_local_idx: torch.Tensor,
        body_pos0: torch.Tensor,
        body_quat0: torch.Tensor,
        joint_types: torch.Tensor,
        joint_pos_local: torch.Tensor,
        joint_axis_local: torch.Tensor,
        joint_dataset_idx: torch.Tensor,
        output_local_idx: torch.Tensor,
        output_body_names: list[str],
        output_body_ids: torch.Tensor,
    ):
        self.device = device
        self.base_body_id = int(base_body_id)
        self.tree_body_ids = tree_body_ids
        self.parent_local_idx = parent_local_idx
        self.joint_types = joint_types
        self.joint_dataset_idx = joint_dataset_idx
        self.output_local_idx = output_local_idx
        self.output_body_names = output_body_names
        self.output_body_ids = output_body_ids
        self.base_local_idx = int(
            (self.tree_body_ids == self.base_body_id).nonzero(as_tuple=False)[0].item()
        )
        self.body_pos0 = body_pos0.to(dtype=torch.float32, device=self.device)
        self.body_quat0 = normalize(body_quat0.to(dtype=torch.float32, device=self.device))
        self.joint_pos_local = joint_pos_local.to(dtype=torch.float32, device=self.device)
        self.joint_axis_local = normalize(
            joint_axis_local.to(dtype=torch.float32, device=self.device)
        )
        self._tree_size = int(self.tree_body_ids.numel())
        self._body_count = len(self.output_body_names)
        self._parent_local_idx_cpu = self.parent_local_idx.detach().cpu().tolist()
        self._joint_types_cpu = self.joint_types.detach().cpu().tolist()
        self._valid_output_idx = (self.output_local_idx >= 0).nonzero(as_tuple=False).squeeze(-1)
        self._valid_output_local_idx = self.output_local_idx[self._valid_output_idx]
        self._depth_groups = self._build_depth_groups()

    def _make_group(self, local_ids: list[int]):
        if len(local_ids) == 0:
            return None
        local_idx = torch.tensor(local_ids, device=self.device, dtype=torch.long)
        return {
            "local_idx": local_idx,
            "parent_idx": self.parent_local_idx.index_select(0, local_idx),
            "pos0": self.body_pos0.index_select(0, local_idx),
            "quat0": self.body_quat0.index_select(0, local_idx),
            "joint_dataset_idx": self.joint_dataset_idx.index_select(0, local_idx),
            "joint_pos_local": self.joint_pos_local.index_select(0, local_idx),
            "joint_axis_local": self.joint_axis_local.index_select(0, local_idx),
        }

    def _build_depth_groups(self):
        depths = [0] * self._tree_size
        max_depth = 0
        for local_idx in range(self._tree_size):
            parent_idx = self._parent_local_idx_cpu[local_idx]
            if parent_idx >= 0:
                depths[local_idx] = depths[parent_idx] + 1
                max_depth = max(max_depth, depths[local_idx])

        groups = []
        for depth in range(1, max_depth + 1):
            fixed_ids, slide_ids, hinge_ids = [], [], []
            for local_idx, node_depth in enumerate(depths):
                if node_depth != depth:
                    continue
                joint_type = self._joint_types_cpu[local_idx]
                if joint_type < 0:
                    fixed_ids.append(local_idx)
                elif joint_type == 2:
                    slide_ids.append(local_idx)
                elif joint_type == 3:
                    hinge_ids.append(local_idx)
                else:
                    raise RuntimeError(f"Unsupported joint type {joint_type} in FK depth grouping.")
            groups.append(
                {
                    "fixed": self._make_group(fixed_ids),
                    "slide": self._make_group(slide_ids),
                    "hinge": self._make_group(hinge_ids),
                }
            )
        return groups

    @classmethod
    def from_mjlab_asset(
        cls,
        *,
        asset: Any,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str],
    ) -> MotionFKHelper:
        """Build from a live mjlab ``Entity`` (used by ``MotionCommand`` at train time)."""
        device = torch.device(asset.data.device)
        model = asset.data.model

        body_name_to_id = {}
        for i, name in enumerate(asset.body_names):
            gid = int(asset.indexing.body_ids[i].item())
            body_name_to_id[_basename(name)] = gid

        joint_id_to_name = {}
        for i, name in enumerate(asset.joint_names):
            gid = int(asset.indexing.joint_ids[i].item())
            joint_id_to_name[gid] = _basename(name)

        base_body_name = _basename(asset.body_names[0])

        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            output_body_names=list(output_body_names),
            base_body_name=base_body_name,
            device=device,
        )

    @classmethod
    def from_mjcf_path(
        cls,
        *,
        xml_path: str | Path,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str] | None = None,
        base_body_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> MotionFKHelper:
        """Build directly from an MJCF file without constructing a simulation.

        Used by the FK correctness test to get a ground-truth-comparable helper
        without paying for a full mjlab ``Scene``/``Simulation`` build.
        """
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        return cls.from_mujoco_model(
            model=model,
            dataset_joint_names=dataset_joint_names,
            output_body_names=output_body_names,
            base_body_name=base_body_name,
            device=device,
        )

    @classmethod
    def from_mujoco_model(
        cls,
        *,
        model: Any,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str] | None = None,
        base_body_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> MotionFKHelper:
        import mujoco

        torch_device = torch.device(device)

        body_name_to_id: dict[str, int] = {}
        ordered_body_names: list[str] = []
        for body_id in range(1, int(model.nbody)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                raise ValueError(f"Unnamed body id={body_id}")
            short_name = _basename(name)
            body_name_to_id[short_name] = body_id
            ordered_body_names.append(short_name)

        joint_id_to_name: dict[int, str] = {}
        free_base_body_id: int | None = None
        for joint_id in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                joint_id_to_name[joint_id] = _basename(name)
            elif int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"Unnamed actuated joint id={joint_id}")

            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                free_base_body_id = int(model.jnt_bodyid[joint_id])

        if base_body_name is None:
            if free_base_body_id is not None and free_base_body_id > 0:
                base_body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, free_base_body_id
                )
            elif ordered_body_names:
                base_body_name = ordered_body_names[0]
        if base_body_name is None:
            raise ValueError("Could not infer a base body from the MuJoCo model")
        base_body_name = _basename(base_body_name)
        if base_body_name not in body_name_to_id:
            raise ValueError(f"Base body '{base_body_name}' not found in model")
        if output_body_names is None:
            # MuJoCo orders parents before children. The scene can also contain
            # terrain or other bodies that do not descend from this robot base.
            subtree = {body_name_to_id[base_body_name]}
            for body_id in range(1, int(model.nbody)):
                if int(model.body_parentid[body_id]) in subtree:
                    subtree.add(body_id)
            output_body_names = [
                name for name in ordered_body_names if body_name_to_id[name] in subtree
            ]

        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            output_body_names=list(output_body_names),
            base_body_name=base_body_name,
            device=torch_device,
        )

    @classmethod
    def _build(
        cls,
        *,
        model: Any,
        body_name_to_id: dict[str, int],
        joint_id_to_name: dict[int, str],
        dataset_joint_names: Sequence[str],
        output_body_names: list[str],
        base_body_name: str,
        device: torch.device,
    ) -> MotionFKHelper:
        body_parentid = _as_scalar_1d(model.body_parentid, device=device).to(torch.long)
        body_jntnum = _as_scalar_1d(model.body_jntnum, device=device).to(torch.long)
        body_jntadr = _as_scalar_1d(model.body_jntadr, device=device).to(torch.long)
        body_pos_all = _as_vec_field(model.body_pos, dim=3, device=device)
        body_quat_all = _as_vec_field(model.body_quat, dim=4, device=device)
        jnt_type = _as_scalar_1d(model.jnt_type, device=device).to(torch.long)
        jnt_pos = _as_vec_field(model.jnt_pos, dim=3, device=device)
        jnt_axis = _as_vec_field(model.jnt_axis, dim=3, device=device)

        if base_body_name not in body_name_to_id:
            raise ValueError(f"Base body '{base_body_name}' not found in model")
        base_body_id = int(body_name_to_id[base_body_name])

        requested_ids: list[int] = []
        for body_name in output_body_names:
            if body_name not in body_name_to_id:
                raise ValueError(f"Output body '{body_name}' not found in model")
            requested_ids.append(int(body_name_to_id[body_name]))

        selected: set[int] = {base_body_id}
        for body_id in requested_ids:
            cur = body_id
            while True:
                selected.add(cur)
                if cur == base_body_id:
                    break
                cur = int(body_parentid[cur].item())
                if cur <= 0:
                    raise ValueError(
                        f"Output body id={body_id} is outside base '{base_body_name}' subtree"
                    )

        children_by_gid: dict[int, list[int]] = {body_id: [] for body_id in selected}
        for body_id in selected:
            if body_id == base_body_id:
                continue
            parent_id = int(body_parentid[body_id].item())
            if parent_id in selected:
                children_by_gid[parent_id].append(body_id)
        for children in children_by_gid.values():
            children.sort()

        order_gid: list[int] = []

        def _dfs(body_id: int):
            order_gid.append(body_id)
            for child_id in children_by_gid.get(body_id, []):
                _dfs(child_id)

        _dfs(base_body_id)
        gid_to_local = {gid: idx for idx, gid in enumerate(order_gid)}
        joint_name_to_dataset_idx = {name: idx for idx, name in enumerate(dataset_joint_names)}
        output_name_to_index = {name: idx for idx, name in enumerate(output_body_names)}
        id_to_body_name = {value: key for key, value in body_name_to_id.items()}

        tree_body_ids = torch.tensor(order_gid, device=device, dtype=torch.long)
        parent_local_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        body_pos0 = torch.empty((len(order_gid), 3), device=device, dtype=body_pos_all.dtype)
        body_quat0 = torch.empty((len(order_gid), 4), device=device, dtype=body_quat_all.dtype)
        joint_types = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        joint_pos_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_pos.dtype)
        joint_axis_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_axis.dtype)
        joint_dataset_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        output_local_idx = torch.full(
            (len(output_body_names),), -1, device=device, dtype=torch.long
        )
        output_body_ids = torch.full((len(output_body_names),), -1, device=device, dtype=torch.long)

        for local_idx, body_id in enumerate(order_gid):
            parent_id = int(body_parentid[body_id].item())
            parent_local_idx[local_idx] = (
                gid_to_local[parent_id] if parent_id in gid_to_local else -1
            )
            body_pos0[local_idx] = body_pos_all[body_id]
            body_quat0[local_idx] = body_quat_all[body_id]

            body_name = id_to_body_name[body_id]
            if body_name in output_name_to_index:
                out_idx = output_name_to_index[body_name]
                output_local_idx[out_idx] = local_idx
                output_body_ids[out_idx] = body_id

            joint_count = int(body_jntnum[body_id].item())
            if joint_count > 1:
                raise NotImplementedError(
                    f"Body '{body_name}' has {joint_count} joints; only <=1 joint/body is supported."
                )
            if joint_count == 0:
                continue

            joint_id = int(body_jntadr[body_id].item())
            joint_type = int(jnt_type[joint_id].item())
            if body_id == base_body_id and joint_type == 0:
                continue
            if joint_type not in (2, 3):
                raise NotImplementedError(
                    f"Joint '{joint_id_to_name[joint_id]}' type={joint_type} unsupported."
                )

            joint_name = joint_id_to_name[joint_id]
            if joint_name not in joint_name_to_dataset_idx:
                raise ValueError(f"Joint '{joint_name}' missing from dataset_joint_names")
            joint_types[local_idx] = joint_type
            joint_pos_local[local_idx] = jnt_pos[joint_id]
            joint_axis_local[local_idx] = jnt_axis[joint_id]
            joint_dataset_idx[local_idx] = joint_name_to_dataset_idx[joint_name]

        if (output_local_idx < 0).any():
            missing = [
                output_body_names[i]
                for i in (output_local_idx < 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            ]
            raise ValueError(f"Failed to resolve requested output bodies: {missing}")

        return cls(
            device=device,
            base_body_id=base_body_id,
            tree_body_ids=tree_body_ids,
            parent_local_idx=parent_local_idx,
            body_pos0=body_pos0,
            body_quat0=body_quat0,
            joint_types=joint_types,
            joint_pos_local=joint_pos_local,
            joint_axis_local=joint_axis_local,
            joint_dataset_idx=joint_dataset_idx,
            output_local_idx=output_local_idx,
            output_body_names=output_body_names,
            output_body_ids=output_body_ids,
        )

    def body_pose(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Root-relative body pose from ``joint_pos``. Returns (body_pos_b, body_quat_b)."""
        if joint_pos.dtype != torch.float32:
            raise RuntimeError("FK helper expects float32 joint_pos input")

        prefix = joint_pos.shape[:-1]
        flat_count = math.prod(prefix) if len(prefix) > 0 else 1
        joint_pos_f = joint_pos.reshape(flat_count, joint_pos.shape[-1])

        tree_pos_b = torch.zeros(
            (flat_count, self._tree_size, 3), device=self.device, dtype=torch.float32
        )
        tree_quat_b = torch.zeros(
            (flat_count, self._tree_size, 4), device=self.device, dtype=torch.float32
        )
        tree_quat_b[:, self.base_local_idx, 0] = 1.0

        for depth_group in self._depth_groups:
            fixed_group = depth_group["fixed"]
            if fixed_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, fixed_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, fixed_group["parent_idx"])
                rel_quat = fixed_group["quat0"].unsqueeze(0)
                rel_pos = fixed_group["pos0"].unsqueeze(0)
                tree_quat_b.index_copy_(
                    1, fixed_group["local_idx"], normalize(_bcast_quat_mul(parent_quat_b, rel_quat))
                )
                tree_pos_b.index_copy_(
                    1,
                    fixed_group["local_idx"],
                    parent_pos_b + _bcast_quat_apply(parent_quat_b, rel_pos),
                )

            slide_group = depth_group["slide"]
            if slide_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, slide_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, slide_group["parent_idx"])
                quat0 = slide_group["quat0"].unsqueeze(0)
                pos0 = slide_group["pos0"].unsqueeze(0)
                axis_local = slide_group["joint_axis_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, slide_group["joint_dataset_idx"])
                axis_parent = _bcast_quat_apply(quat0, axis_local)
                rel_quat = quat0
                rel_pos = pos0 + axis_parent * joint_value.unsqueeze(-1)
                tree_quat_b.index_copy_(
                    1, slide_group["local_idx"], normalize(_bcast_quat_mul(parent_quat_b, rel_quat))
                )
                tree_pos_b.index_copy_(
                    1,
                    slide_group["local_idx"],
                    parent_pos_b + _bcast_quat_apply(parent_quat_b, rel_pos),
                )

            hinge_group = depth_group["hinge"]
            if hinge_group is not None:
                parent_pos_b = tree_pos_b.index_select(1, hinge_group["parent_idx"])
                parent_quat_b = tree_quat_b.index_select(1, hinge_group["parent_idx"])
                quat0 = hinge_group["quat0"].unsqueeze(0)
                pos0 = hinge_group["pos0"].unsqueeze(0)
                axis_local = hinge_group["joint_axis_local"].unsqueeze(0)
                anchor_local = hinge_group["joint_pos_local"].unsqueeze(0)
                joint_value = joint_pos_f.index_select(1, hinge_group["joint_dataset_idx"])
                joint_quat = quat_from_angle_axis(joint_value, axis_local)
                rel_quat = _bcast_quat_mul(quat0, joint_quat)
                rel_pos = pos0 + _bcast_quat_apply(
                    quat0, anchor_local - _bcast_quat_apply(joint_quat, anchor_local)
                )
                tree_quat_b.index_copy_(
                    1, hinge_group["local_idx"], normalize(_bcast_quat_mul(parent_quat_b, rel_quat))
                )
                tree_pos_b.index_copy_(
                    1,
                    hinge_group["local_idx"],
                    parent_pos_b + _bcast_quat_apply(parent_quat_b, rel_pos),
                )

        body_pos_b = torch.zeros(
            (flat_count, self._body_count, 3), device=self.device, dtype=torch.float32
        )
        body_quat_b = torch.zeros(
            (flat_count, self._body_count, 4), device=self.device, dtype=torch.float32
        )
        body_quat_b[..., 0] = 1.0

        if self._valid_output_idx.numel() > 0:
            body_pos_b.index_copy_(
                1, self._valid_output_idx, tree_pos_b.index_select(1, self._valid_output_local_idx)
            )
            body_quat_b.index_copy_(
                1, self._valid_output_idx, tree_quat_b.index_select(1, self._valid_output_local_idx)
            )

        body_pos_b = body_pos_b.reshape(prefix + (self._body_count, 3))
        body_quat_b = body_quat_b.reshape(prefix + (self._body_count, 4))
        return body_pos_b, body_quat_b

    def expand_minimal_motion(self, motion: MotionMinimalData, fps: float) -> MotionData:
        """FK-expand a ``(..., T, ...)`` minimal window into full ``MotionData``.

        ``fps`` sets the finite-difference ``dt``; velocities are computed along
        dim=-2 (the frame/time axis, one before the trailing feature axis) and
        smoothed with a 5-tap average.
        """
        if (
            motion.root_pos_w.dtype != torch.float32
            or motion.root_quat_w.dtype != torch.float32
            or motion.joint_pos.dtype != torch.float32
        ):
            raise RuntimeError("FK helper expects MotionMinimalData float32 tensors")
        root_pos_w = motion.root_pos_w
        root_quat_w = normalize(motion.root_quat_w)
        joint_pos = motion.joint_pos
        time_dim = root_pos_w.ndim - 2

        root_lin_vel_w = smooth_avg5_torch(
            finite_diff_torch(root_pos_w, fps, dim=time_dim), dim=time_dim
        )
        root_ang_vel_w = smooth_avg5_torch(
            angvel_from_quat_wxyz_torch(root_quat_w, fps, dim=time_dim), dim=time_dim
        )
        joint_vel = smooth_avg5_torch(finite_diff_torch(joint_pos, fps, dim=time_dim), dim=time_dim)
        body_pos_b, body_quat_b = self.body_pose(joint_pos)
        body_pos_w = _bcast_quat_apply(
            root_quat_w.unsqueeze(-2), body_pos_b
        ) + root_pos_w.unsqueeze(-2)
        body_quat_w = normalize(_bcast_quat_mul(root_quat_w.unsqueeze(-2), body_quat_b))

        root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_lin_vel_b = smooth_avg5_torch(
            finite_diff_torch(body_pos_b, fps, dim=time_dim)
            + torch.cross(root_ang_vel_b.unsqueeze(-2), body_pos_b, dim=-1),
            dim=time_dim,
        )
        body_ang_vel_b = smooth_avg5_torch(
            angvel_from_quat_wxyz_torch(body_quat_b, fps, dim=time_dim), dim=time_dim
        )
        body_lin_vel_w = _bcast_quat_apply(
            root_quat_w.unsqueeze(-2), body_lin_vel_b
        ) + root_lin_vel_w.unsqueeze(-2)
        body_ang_vel_w = _bcast_quat_apply(
            root_quat_w.unsqueeze(-2), body_ang_vel_b
        ) + root_ang_vel_w.unsqueeze(-2)

        return MotionData(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_pos_b=body_pos_b,
            body_quat_w=body_quat_w,
            body_quat_b=body_quat_b,
            body_lin_vel_w=body_lin_vel_w,
            body_lin_vel_b=body_lin_vel_b,
            body_ang_vel_w=body_ang_vel_w,
            body_ang_vel_b=body_ang_vel_b,
            batch_size=list(motion.batch_size),
            device=self.device,
        )
