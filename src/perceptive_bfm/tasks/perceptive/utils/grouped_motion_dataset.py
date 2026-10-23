"""Sparse per-terrain minimal motion data, sharded by flat motion identity.

Body state and velocities are reconstructed from windows at runtime.
"""

from __future__ import annotations

import os.path as osp
from collections.abc import Sequence
from functools import cached_property

import torch
from loguru import logger

from perceptive_bfm.utils.motion import (
    GlobalMotionDataset,
    MotionMinimalData,
)


class GroupedMotionDataset:
    def __init__(
        self,
        motion_file: str,
        group_names: list[str],
        *,
        joint_names: Sequence[str],
        device: str = "cpu",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size and world_size >= 1")
        if len(set(group_names)) != len(group_names):
            raise ValueError("group_names must be unique")
        if not group_names:
            raise ValueError("GroupedMotionDataset requires a non-empty `group_names`.")
        if "flat" not in group_names:
            raise ValueError(
                f"GroupedMotionDataset requires a 'flat' group, got groups: {group_names}"
            )
        self.motion_file = motion_file
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.group_names = list(group_names)
        self._flat_group_index = self.group_names.index("flat")

        # 1. Load every requested group fully resident (unsharded). A group whose
        #    directory is entirely absent is treated as "no motions for this group"
        #    (see pack_motions.py, which skips packing empty terrain
        #    groups) -- its row in start_lookup/len_lookup below simply stays all
        #    zero. A directory that exists but isn't a valid pack (e.g. an
        #    interrupted packing run) still fails loudly via GlobalMotionDataset's
        #    own check, since that's more likely a mistake than an intentional skip.
        empty_groups: list[str] = []
        group_datasets: dict[str, GlobalMotionDataset] = {}
        for group_name in self.group_names:
            group_dir = osp.join(motion_file, group_name)
            if not osp.isdir(group_dir):
                empty_groups.append(group_name)
                continue
            group_datasets[group_name] = GlobalMotionDataset(
                group_dir, joint_names=joint_names, device=device
            )

        if "flat" in empty_groups:
            raise ValueError(
                "GroupedMotionDataset requires a packed 'flat' group (the union of "
                f"every other group's motions), but no directory was found at "
                f"{osp.join(motion_file, 'flat')}."
            )
        if empty_groups:
            logger.warning(
                f"[Rank {rank}] {len(empty_groups)}/{len(self.group_names)} declared "
                f"terrain group(s) have no packed motion data, treated as empty: "
                f"{empty_groups}"
            )

        joint_dims = {gd.joint_dim for gd in group_datasets.values()}
        fps_values = {gd.fps for gd in group_datasets.values()}
        if len(joint_dims) > 1:
            raise ValueError(f"joint_dim mismatch across groups: {joint_dims}")
        if len(fps_values) > 1:
            raise ValueError(f"fps mismatch across groups: {fps_values}")
        self.joint_names = group_datasets["flat"].joint_names
        self.joint_dim = joint_dims.pop()
        self.fps = fps_values.pop()

        # Every motion that exists on any group must also exist on flat -- both the
        # rank-sharding below and command_mode='source' key off flat's motion-name
        # space, so a motion missing from flat would otherwise silently vanish
        # (invisible to the sampler) instead of failing loudly.
        flat_names_set = set(group_datasets["flat"].source_paths)
        for group_name, gd in group_datasets.items():
            if group_name == "flat":
                continue
            missing = sorted(set(gd.source_paths) - flat_names_set)
            if missing:
                raise ValueError(
                    f"Group '{group_name}' has {len(missing)} motion(s) not present on "
                    f"the flat group: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                    f"The flat group must be a superset of every other group's motions "
                    f"-- re-pack so flat includes them."
                )

        flat_lengths = dict(
            zip(group_datasets["flat"].source_paths, group_datasets["flat"].lengths.tolist())
        )
        for group_name, gd in group_datasets.items():
            for name, length in zip(gd.source_paths, gd.lengths.tolist()):
                if length != flat_lengths[name]:
                    raise ValueError(
                        f"Paired frame count mismatch for {group_name}/{name}: {length} vs flat {flat_lengths[name]}"
                    )

        # 2. Rank-shard the *flat* group's sorted motion names -- the canonical global
        #    motion id space every other group is looked up against.
        flat_names_all = sorted(group_datasets["flat"].source_paths)
        shard_names = flat_names_all[rank::world_size]
        if not shard_names:
            raise ValueError(
                f"Rank {rank} has no motions assigned "
                f"(num_motions={len(flat_names_all)}, world_size={world_size})."
            )
        num_motions = len(shard_names)
        num_groups = len(self.group_names)

        # 3. For every (group, shard-local motion) pair present in that group's pack,
        #    slice out its rows and build a flat depot + sparse (start, len) lookup.
        start_lookup = torch.zeros((num_groups, num_motions), dtype=torch.long)
        len_lookup = torch.zeros((num_groups, num_motions), dtype=torch.long)
        chunks: list[MotionMinimalData] = []
        cursor = 0
        for gi, group_name in enumerate(self.group_names):
            gd = group_datasets.get(group_name)
            if gd is None:
                continue  # empty group (see step 1): row stays all-zero.
            name_to_local = {name: i for i, name in enumerate(gd.source_paths)}
            for mj, name in enumerate(shard_names):
                local_idx = name_to_local.get(name)
                if local_idx is None:
                    continue  # sparse: this group has no clip for this motion.
                s = int(gd.starts[local_idx].item())
                e = int(gd.ends[local_idx].item())
                chunks.append(gd.data[s:e])
                start_lookup[gi, mj] = cursor
                len_lookup[gi, mj] = e - s
                cursor += e - s

        if cursor == 0:
            raise ValueError(f"Rank {rank}: no motion frames were loaded across any group.")

        self.data: MotionMinimalData = (
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        ).to(device=device, dtype=torch.float32)
        self.start_lookup = start_lookup.to(device)
        self.len_lookup = len_lookup.to(device)
        self._num_groups = num_groups
        self._num_motions = num_motions

        logger.info(
            f"[Rank {self.rank}] GroupedMotionDataset ready: {num_groups} groups x "
            f"{num_motions} motions (shard of {len(flat_names_all)}), {cursor} frames "
            f"resident on {self.device}."
        )

    @cached_property
    def num_groups(self) -> int:
        return self._num_groups

    @cached_property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_lengths(self) -> torch.Tensor:
        return self.len_lookup[self._flat_group_index, :]

    def get_slice(
        self,
        group_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> MotionMinimalData:
        """Fetch a `(len(group_ids), frame_offsets.shape[-1])` window per query.

        `frame_offsets` is broadcast against `group_ids`/`motion_ids` (typically
        `fill_start[:, None] + arange(window)[None, :]`, already relative to each
        clip's own frame 0). Out-of-range indices are clamped to the clip's first/last
        valid frame.
        """
        group_ids = group_ids.to(device=self.device, dtype=torch.long)
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        frame_offsets = frame_offsets.to(device=self.device, dtype=torch.long)
        clip_starts = self.start_lookup[group_ids, motion_ids]
        clip_lens = self.len_lookup[group_ids, motion_ids]
        if not bool((clip_lens > 0).all()):
            raise ValueError(
                "get_slice queried at least one (group, motion) pair with no data "
                "(len_lookup == 0); check group_ids/motion_ids against len_lookup first."
            )
        if frame_offsets.ndim == 1:
            idx = clip_starts.unsqueeze(1) + frame_offsets.unsqueeze(0)
        else:
            idx = clip_starts.unsqueeze(-1) + frame_offsets
        idx = torch.max(idx, clip_starts.unsqueeze(-1))
        idx = torch.min(idx, (clip_starts + clip_lens - 1).unsqueeze(-1))
        return self.data[idx]
