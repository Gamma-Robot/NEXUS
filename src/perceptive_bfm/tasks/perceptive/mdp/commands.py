from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from perceptive_bfm.tasks.perceptive.utils.ground_query import (
    EEGroundRaycaster,
    resolve_terrain_geom_groups,
)
from perceptive_bfm.tasks.perceptive.utils.grouped_motion_dataset import GroupedMotionDataset
from perceptive_bfm.utils.contact_phase import EEContactCfg, ee_contact_hysteresis_step
from perceptive_bfm.utils.contact_viz import EE_SPHERE_RADIUS, ee_sphere_specs
from perceptive_bfm.utils.fk_helper import MotionFKHelper

if TYPE_CHECKING:
    import viser
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _group_raw_proportions(
    sub_terrain_names: Sequence[str],
    proportions: dict[str, float],
    num_levels: int,
) -> list[float]:
    """Per-group raw ``proportion`` values, in the same order ``MotionCommand``
    builds ``group_names`` (see its ``__init__``): flat contributes one entry, every
    other terrain type contributes one entry per level, all sharing that type's
    single ``proportion`` value."""
    raw: list[float] = []
    for i, name in enumerate(sub_terrain_names):
        if i == 0:
            raw.append(proportions[name])
            continue
        raw.extend([proportions[name]] * num_levels)
    return raw


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        ##
        # Robot setup
        ##
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        # Reference EE contact schedule (ee_contact_active): local indices of the four
        # end-effectors within the tracked body list (cfg.body_names), used by the
        # reference air_time rewards. Feet columns come first, hands after -- that
        # order is relied on by the feet_standing/hands_standing slice properties, by
        # the rewards' per-group column resolution, and by the ee_contact_state
        # observations. Both name tuples must be subsets of body_names; empty disables.
        self.ee_body_names: tuple[str, ...] = tuple(self.cfg.foot_body_names) + tuple(
            self.cfg.hand_body_names
        )
        self._num_feet = len(self.cfg.foot_body_names)
        self.motion_ee_body_indexes = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.ee_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.ee_contact_active = torch.zeros(
            self.num_envs,
            len(self.ee_body_names),
            dtype=torch.bool,
            device=self.device,
        )
        # Terrain height under each EE, refreshed by command updates. Dense rewards
        # also refresh the robot heights after physics, before the next command update.
        # Two separate sets of query points: the *target* motion's EE xy drives the
        # reference contact schedule, while the *robot's own* EE xy drives the dense
        # rewards' clearance shaping. They diverge exactly at step edges, which is
        # where getting it right matters most. Both are filled by one raycast.
        self.target_ee_ground_z = torch.zeros(
            self.num_envs, len(self.ee_body_names), device=self.device
        )
        self.robot_ee_ground_z = torch.zeros(
            self.num_envs, len(self.ee_body_names), device=self.device
        )
        self.target_anchor_ground_z = torch.zeros(self.num_envs, device=self.device)
        self.robot_anchor_ground_z = torch.zeros(self.num_envs, device=self.device)
        # Reference clearance: how high the TARGET motion holds each EE above the
        # terrain under it. This is what the dense rewards shape toward, instead of
        # hand-tuned constants -- see rewards.py's _ee_air_time_reference_dense.
        self.target_ee_clearance = torch.zeros(
            self.num_envs, len(self.ee_body_names), device=self.device
        )
        self.robot_ee_body_indexes = torch.tensor(
            (
                self.robot.find_bodies(list(self.ee_body_names), preserve_order=True)[0]
                if self.ee_body_names
                else []
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._ground_raycaster: EEGroundRaycaster | None = None
        self._robot_ground_raycaster: EEGroundRaycaster | None = None
        self._robot_ground_refresh_step = -1
        self._anchor_ground_raycaster: EEGroundRaycaster | None = None

        # Live per-EE contact state for debug visualization only (see
        # update_ee_contact_viz/_debug_vis_impl) -- not consumed by any reward,
        # kept fully independent of feet_standing above.
        self.ee_contact_viz_body_indexes = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.cfg.ee_contact_viz_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.ee_contact_viz_standing = torch.zeros(
            self.num_envs,
            len(self.cfg.ee_contact_viz_body_names),
            dtype=torch.bool,
            device=self.device,
        )

        ##
        # Environment setup
        ##
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.group_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # External resets resample before the manager's command update. The new command
        # has not participated in simulation yet, so its first same-step visit must be
        # skipped once (see `_prepare_adaptive_resample`).
        self._adaptive_skip_visit_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Motion end normally resamples silently without a done signal. An explicit
        # counter lets evaluators detect completion even when the next sampled pair
        # has the same group and motion ids (for example a one-motion group).
        self.natural_completion_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Buffer origin: the motion-relative frame that buffer index 0 corresponds to.
        # Doubles as scratch space for the sampled start phase between
        # `_sample_and_load`'s steps 3 and 4 (see that method) -- its final value after
        # step 4 is always the FK cache's own origin, `time_steps + cache_fill_range[0]`.
        self.start_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Explicit sparse offset list (reference, may include negative/past offsets)
        # overrides the synthetic linear/quadratic/exp schedule below.
        self.future_indexes = torch.zeros(0, dtype=torch.long, device=self.device)
        if self.cfg.future_offsets is not None:
            self.future_indexes = torch.tensor(
                self.cfg.future_offsets, dtype=torch.long, device=self.device
            )
        elif self.cfg.future_window_size > 0:
            self.future_indexes = self._compute_future_indexes(
                self.cfg.future_window_size,
                self.cfg.lookahead_mode,
                self.cfg.lookahead_stride,
                self.device,
            )
        # Teleop ("policy") horizon: a separate, typically much shorter offset list for
        # the deployable policy's observations. Distinct from the "teacher" horizon
        # above (`priv`'s future_offsets) -- both horizons read the same underlying
        # motion buffer, just different offsets into it.
        self.future_indexes_teleop = torch.tensor(
            self.cfg.future_offsets_teleop, dtype=torch.long, device=self.device
        )
        future_max = max(
            (int(self.future_indexes.max().item()) if self.future_indexes.numel() > 0 else 0),
            (
                int(self.future_indexes_teleop.max().item())
                if self.future_indexes_teleop.numel() > 0
                else 0
            ),
        )
        past_max = max(
            (
                int((-self.future_indexes.min()).clamp(min=0).item())
                if self.future_indexes.numel() > 0
                else 0
            ),
            (
                int((-self.future_indexes_teleop.min()).clamp(min=0).item())
                if self.future_indexes_teleop.numel() > 0
                else 0
            ),
        )

        ##
        # Data setup
        ##
        group_names = []
        self._terrain = self._env.scene.terrain
        if self._terrain is not None and self._terrain.cfg.terrain_type != "plane":
            generator = self._terrain.cfg.terrain_generator
            assert generator is not None
            sub_terrain_names = tuple(generator.sub_terrains.keys())
            num_types = len(sub_terrain_names)
            num_levels = int(generator.num_rows)
            # Bidirectional mapping between terrain (type, level) and motion group id,
            # used for selecting terrain-specific motion clips.
            self.terrain_to_group_id = torch.full(
                (num_types, num_levels), -1, dtype=torch.long, device=self.device
            )
            for terrain_id, terrain_name in enumerate(sub_terrain_names):
                if terrain_id == 0:
                    assert (
                        terrain_name == "flat"
                    ), f"Expected the first terrain type to be 'flat', but got '{terrain_name}'"
                    # All levels of the flat terrain share the same data.
                    group_names.append(terrain_name)
                    self.terrain_to_group_id[terrain_id, :] = 0
                    continue
                for level in range(num_levels):
                    group_names.append(f"{terrain_name}_lv{level}")
                    self.terrain_to_group_id[terrain_id, level] = len(group_names) - 1

            # Per-group terrain-`proportion` weight for the adaptive sampler (see
            # `_recompute_bin_probabilities`), normalized to mean 1 across groups so
            # equal configured proportions are an exact no-op regardless of how many
            # levels/motions any group has.
            raw_group_proportions = torch.tensor(
                _group_raw_proportions(
                    sub_terrain_names,
                    {name: cfg.proportion for name, cfg in generator.sub_terrains.items()},
                    num_levels,
                ),
                dtype=torch.float,
                device=self.device,
            )
            self.group_proportion_weight = raw_group_proportions / raw_group_proportions.mean()

        # Load motion data: one packed (tracking-format, minimal-field) dataset per
        # terrain group -- see `perceptive_bfm.scripts.data_process.
        # pack_motions` for the packer and `GroupedMotionDataset` for the
        # loader. Full per-body pose/velocity is reconstructed at train time via FK
        # (`MotionFKHelper`) from a small sliding-window cache
        # (`_refill_group_cache`/`_ensure_group_cache`) instead of being resident on
        # disk or in VRAM.
        self.motion = GroupedMotionDataset(
            motion_file=self.cfg.motion_file,
            group_names=group_names,
            joint_names=list(self.robot.joint_names),
            device=self.device,
            rank=self.cfg.rank,
            world_size=self.cfg.world_size,
        )
        # One motion frame advances per control step. Allow scalar rounding only;
        # otherwise position targets and FPS-derived velocities disagree.
        control_frequency = 1.0 / env.step_dt
        if not math.isclose(self.motion.fps, control_frequency, rel_tol=1e-6, abs_tol=0.0):
            raise ValueError(
                f"Packed motion FPS ({self.motion.fps:g}) must match the environment "
                f"control frequency ({control_frequency:g} Hz; step_dt={env.step_dt:g} s). "
                f"Prepare and repack motions at {control_frequency:g} FPS; "
                "packing does not resample motions."
            )
        self._fk_helper = MotionFKHelper.from_mjlab_asset(
            asset=self.robot,
            dataset_joint_names=list(self.robot.joint_names),
            output_body_names=list(self.cfg.body_names),
        )
        # Source (reference) group used when flat-source buffers are enabled. By
        # convention the flat group is index 0 in `group_names`; GroupedMotionDataset
        # guarantees it is a superset of every other group's motions.
        self.source_group_index = 0

        # Cache window sizing: the configured [start, end) pair (cache_request_range,
        # cache_fill_range), not a derived margin/formula. cache_request_range is the
        # range ever actually queried (must cover [-past_max, future_max+1));
        # cache_fill_range is the range actually fetched/FK-expanded and cached, and
        # must contain cache_request_range. The gap between the two on the forward
        # side is pure amortization slack (a refill stays valid for fill_end -
        # request_end extra steps of forward phase progress); reference puts no slack on
        # the backward side (fill_start == request_start), and neither do we by
        # default, so cache refills can be amortized over forward phase progress.
        request_start, request_end = self.cfg.cache_request_range
        fill_start_offset, fill_end_offset = self.cfg.cache_fill_range
        if not (request_start < request_end and fill_start_offset < fill_end_offset):
            raise ValueError(
                "cache_request_range and cache_fill_range must each satisfy start < end."
            )
        if not (fill_start_offset <= request_start <= 0 < request_end <= fill_end_offset):
            raise ValueError(
                "cache_fill_range must cover cache_request_range, and "
                "cache_request_range must include 0."
            )
        if request_start > -past_max:
            raise ValueError(
                f"future_offsets reads {past_max} frames into the past, but "
                f"cache_request_range starts at {request_start}. Widen "
                "cache_request_range (and cache_fill_range) to cover it."
            )
        if request_end < future_max + 1:
            raise ValueError(
                f"future_offsets reads {future_max} frames into the future, but "
                f"cache_request_range ends at {request_end}. Widen cache_request_range "
                "(and cache_fill_range) to cover it."
            )
        self._cache_request_start = request_start
        self._cache_request_end = request_end
        self._cache_fill_start_offset = fill_start_offset
        self.motion_buffer_length: int = fill_end_offset - fill_start_offset
        self._init_motion_buffers()

        ##
        # Adaptive sampling setup (per-bin per-bin failure-rate curriculum).
        #
        # The sampling space is a flat list of *clips*: every valid (group, motion)
        # pair (i.e. a motion that has terrain data on that group) is split into
        # fixed-size phase bins. A bin is the atomic unit of sampling -- drawing one
        # bin simultaneously fixes the terrain group, the motion clip, and the start
        # phase. Per-bin failure/episode counters then implement a fine-grained
        # curriculum over (terrain, motion, phase), decoupling group difficulty from
        # motion difficulty. Different groups may hold different numbers of motions.
        ##
        # Built unconditionally (not just for adaptive): uniform/start also need it,
        # to re-place an env physically when `_resolve_group_from_terrain` redraws
        # it off an empty terrain group (see `_resolve_group_from_terrain`).
        self._build_group_terrain_cells()
        if self.cfg.sampling_mode == "adaptive":
            self._build_adaptive_bins()
        else:
            # uniform / start: restrict motion choice to clips valid on each env's
            # group, so a sparse (union) pack never loads an empty (group, motion).
            self._build_group_valid_motions()

        ##
        # Metrics setup
        ##
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.sampling_mode == "adaptive":
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_effective_num_bins"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_prob_max_over_uniform"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_num_concentrated_bins"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_evidence_min"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_evidence_max"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_evidence_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_evidence_median"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_failure_rate_min"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_failure_rate_max"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_failure_rate_mean"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["sampling_1st_percentile_avg_failure_rate"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["mastery_ratio"] = torch.zeros(self.num_envs, device=self.device)
            # Recent-performance mastery signal: based on the diagnostic EMA failure
            # rate rather than the cumulative one, so it responds within ~10 episodes
            # instead of only fading out once enough new episodes dilute old failures.
            self.metrics["mastery_ratio_recent"] = torch.zeros(self.num_envs, device=self.device)

        # Ghost model created lazily on first visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
        self._src_ghost_model: mujoco.MjModel | None = None
        self._src_ghost_color = np.array(cfg.viz.src_ghost_color, dtype=np.float32)
        # Reset-time baseline for the flat-reference ghost: its position tracks the
        # flat clip's own delta since the env's last reset, not the (possibly
        # terrain-climbing) current terrain-target height. Only populated when
        # show_src_ghost is on. reset_quat_w is the (yaw-only) anchor heading at
        # reset, used to rotate the body-frame src_ghost_offset_b into world frame.
        self._src_ghost_reset_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._src_ghost_reset_flat_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._src_ghost_reset_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._src_ghost_reset_quat_w[:, 0] = 1.0

    @staticmethod
    def _compute_future_indexes(
        window_size: int,
        mode: Literal["linear", "quadratic", "exp"] = "linear",
        stride: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Args:
          window_size (int): Number of future frames.
          mode (Literal["linear", "quadratic", "exp"]): Lookahead mode. "linear" for fixed stride, "quadratic" for n^2, "exp" for 2^n. Default is "linear".
          stride (int): Stride for linear lookahead mode. Only used if mode is "linear". Default is 1.
          device (str): Device to place the tensor on.
        Returns:
          torch.Tensor: Tensor of shape (window_size, )
        """
        assert window_size > 0, "Lookahead steps must be greater than 0."
        if mode == "linear":
            # linear increment with fixed stride: 1+n*stride. e.g., for stride=2 -> 1, 3, 5, 7, ...
            future_indexes = stride * torch.arange(0, window_size, device=device) + 1
        if mode == "quadratic":
            # quadratic increment: n^2. e.g., 1, 4, 9, 16, ...
            future_indexes = torch.arange(1, window_size + 1, device=device) ** 2
        if mode == "exp":
            # exponential increment: 2^n. e.g., 1, 2, 4, 8, ...
            future_indexes = 2 ** torch.arange(0, window_size, device=device)
        return future_indexes  # only future frames

    def _init_motion_buffers(self):
        body_dim = len(self.cfg.body_names)
        self.joint_pos_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, self.motion.joint_dim),
            device=self.device,
        )
        self.joint_vel_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, self.motion.joint_dim),
            device=self.device,
        )
        self.body_pos_w_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, body_dim, 3),
            device=self.device,
        )
        self.body_quat_w_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, body_dim, 4),
            device=self.device,
        )
        self.body_quat_w_buf[:, :, :, 0] = 1.0  # initialize quaternions to (1, 0, 0, 0)
        self.body_lin_vel_w_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, body_dim, 3),
            device=self.device,
        )
        self.body_ang_vel_w_buf = torch.zeros(
            (self.num_envs, self.motion_buffer_length, body_dim, 3),
            device=self.device,
        )
        if self.cfg.flat_motion_enabled:
            self.flat_joint_pos_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, self.motion.joint_dim),
                device=self.device,
            )
            self.flat_joint_vel_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, self.motion.joint_dim),
                device=self.device,
            )
            self.flat_body_pos_w_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, body_dim, 3),
                device=self.device,
            )
            self.flat_body_quat_w_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, body_dim, 4),
                device=self.device,
            )
            self.flat_body_quat_w_buf[:, :, :, 0] = 1.0
            self.flat_body_lin_vel_w_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, body_dim, 3),
                device=self.device,
            )
            self.flat_body_ang_vel_w_buf = torch.zeros(
                (self.num_envs, self.motion_buffer_length, body_dim, 3),
                device=self.device,
            )

        # Configure the scene terrain up-front (the per-env group id is resolved
        # from the terrain layout inside `_resample_command`).
        if (
            self._terrain is not None
            and self._terrain.cfg.terrain_type != "plane"
            and self.cfg.fixed_terrain is not None
        ):
            terrain_type, terrain_level = self.cfg.fixed_terrain
            self._terrain.terrain_types[:] = terrain_type
            self._terrain.terrain_levels[:] = terrain_level
            self._env.scene.env_origins[:] = self._terrain.terrain_origins[
                terrain_level, terrain_type
            ]

    def _refill_group_cache(self, env_ids: torch.Tensor):
        """Gather packed terrain/flat windows over cache_fill_range, expand FK and fill the selected environment buffers."""
        if env_ids.numel() == 0:
            return
        fill_start = self.time_steps[env_ids] + self._cache_fill_start_offset
        frame_offsets = fill_start.unsqueeze(1) + torch.arange(
            self.motion_buffer_length, device=self.device
        ).unsqueeze(0)
        minimal = self.motion.get_slice(
            self.group_ids[env_ids], self.motion_ids[env_ids], frame_offsets
        )
        full = self._fk_helper.expand_minimal_motion(minimal, fps=float(self.motion.fps))

        self.joint_pos_buf[env_ids] = full.joint_pos
        self.joint_vel_buf[env_ids] = full.joint_vel
        self.body_pos_w_buf[env_ids] = full.body_pos_w
        self.body_quat_w_buf[env_ids] = full.body_quat_w
        self.body_lin_vel_w_buf[env_ids] = full.body_lin_vel_w
        self.body_ang_vel_w_buf[env_ids] = full.body_ang_vel_w
        self.start_time_steps[env_ids] = fill_start

        if self.cfg.flat_motion_enabled:
            flat_minimal = self.motion.get_slice(
                torch.full_like(self.group_ids[env_ids], self.source_group_index),
                self.motion_ids[env_ids],
                frame_offsets,
            )
            flat_full = self._fk_helper.expand_minimal_motion(
                flat_minimal, fps=float(self.motion.fps)
            )
            self.flat_joint_pos_buf[env_ids] = flat_full.joint_pos
            self.flat_joint_vel_buf[env_ids] = flat_full.joint_vel
            self.flat_body_pos_w_buf[env_ids] = flat_full.body_pos_w
            self.flat_body_quat_w_buf[env_ids] = flat_full.body_quat_w
            self.flat_body_lin_vel_w_buf[env_ids] = flat_full.body_lin_vel_w
            self.flat_body_ang_vel_w_buf[env_ids] = flat_full.body_ang_vel_w

    def _ensure_group_cache(self):
        """Refill windows whose requested phase range falls outside the cached fill range."""
        cache_start = self.start_time_steps
        cache_end = self.start_time_steps + self.motion_buffer_length
        req_low = self.time_steps + self._cache_request_start
        req_high = self.time_steps + self._cache_request_end
        hit = (req_low >= cache_start) & (req_high <= cache_end)
        miss_env_ids = (~hit).nonzero(as_tuple=False).squeeze(-1)
        self._refill_group_cache(miss_env_ids)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def buffer_indexes(self) -> torch.Tensor:
        return torch.clamp(
            self.time_steps - self.start_time_steps,
            min=0,
            max=self.motion_buffer_length - 1,
        )

    @property
    def env_origins_xy(self) -> torch.Tensor:
        return torch.cat(
            [
                self._env.scene.env_origins[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ],
            dim=1,
        )

    @property
    def terrain_types(self) -> torch.Tensor:
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        return self._terrain.terrain_types

    @property
    def terrain_levels(self) -> torch.Tensor:
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        return self._terrain.terrain_levels

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.joint_pos_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.joint_vel_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device), self.buffer_indexes
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device),
                self.buffer_indexes,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def buffer_indexes_future(self) -> torch.Tensor:
        if self.cfg.future_window_size <= 0:
            raise ValueError("Future window size must be greater than 0 to access future frames.")
        return torch.clamp(
            self.buffer_indexes[:, None] + self.future_indexes[None, :],
            min=0,
            max=self.motion_buffer_length - 1,
        )

    @property
    def joint_pos_future(self) -> torch.Tensor:
        return self.joint_pos_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def joint_vel_future(self) -> torch.Tensor:
        return self.joint_vel_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def anchor_pos_w_future(self) -> torch.Tensor:
        return (
            self.body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def anchor_quat_w_future(self) -> torch.Tensor:
        return self.body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def anchor_lin_vel_w_future(self) -> torch.Tensor:
        return self.body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def anchor_ang_vel_w_future(self) -> torch.Tensor:
        return self.body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def body_pos_w_future(self) -> torch.Tensor:
        return (
            self.body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future,
            ]
            + self.env_origins_xy[:, None, None, :]
        )

    @property
    def body_quat_w_future(self) -> torch.Tensor:
        return self.body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def body_lin_vel_w_future(self) -> torch.Tensor:
        return self.body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def body_ang_vel_w_future(self) -> torch.Tensor:
        return self.body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    ##
    # Teleop ("policy") horizon -- same buffers, different (typically much shorter)
    # offset list. See `future_indexes_teleop` in __init__.
    ##

    @property
    def buffer_indexes_future_teleop(self) -> torch.Tensor:
        if self.future_indexes_teleop.numel() == 0:
            raise ValueError(
                "future_offsets_teleop must be non-empty to access teleop future frames."
            )
        return torch.clamp(
            self.buffer_indexes[:, None] + self.future_indexes_teleop[None, :],
            min=0,
            max=self.motion_buffer_length - 1,
        )

    @property
    def joint_pos_future_teleop(self) -> torch.Tensor:
        return self.joint_pos_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future_teleop,
        ]

    @property
    def anchor_pos_w_future_teleop(self) -> torch.Tensor:
        return (
            self.body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future_teleop,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def anchor_quat_w_future_teleop(self) -> torch.Tensor:
        return self.body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future_teleop,
            self.motion_anchor_body_index,
        ]

    ##
    # Flat-source buffer/property layer -- exact parallel of the terrain-side
    # properties above, backed by flat_*_buf instead of *_buf. Observations
    # always imitate the flat clip (see perceptive/mdp/observations.py); reward/
    # termination keep reading the terrain-side properties above, which still
    # reflect the env's actual terrain-conditioned target.
    ##

    @property
    def flat_joint_pos(self) -> torch.Tensor:
        return self.flat_joint_pos_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def flat_joint_vel(self) -> torch.Tensor:
        return self.flat_joint_vel_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def flat_body_pos_w(self) -> torch.Tensor:
        return (
            self.flat_body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device), self.buffer_indexes
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def flat_body_quat_w(self) -> torch.Tensor:
        return self.flat_body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def flat_body_lin_vel_w(self) -> torch.Tensor:
        return self.flat_body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def flat_body_ang_vel_w(self) -> torch.Tensor:
        return self.flat_body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device), self.buffer_indexes
        ]

    @property
    def flat_anchor_pos_w(self) -> torch.Tensor:
        return (
            self.flat_body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device),
                self.buffer_indexes,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy
        )

    @property
    def flat_anchor_quat_w(self) -> torch.Tensor:
        return self.flat_body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.flat_body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.flat_body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device),
            self.buffer_indexes,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_joint_pos_future(self) -> torch.Tensor:
        return self.flat_joint_pos_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def flat_joint_vel_future(self) -> torch.Tensor:
        return self.flat_joint_vel_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def flat_anchor_pos_w_future(self) -> torch.Tensor:
        return (
            self.flat_body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def flat_anchor_quat_w_future(self) -> torch.Tensor:
        return self.flat_body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_anchor_lin_vel_w_future(self) -> torch.Tensor:
        return self.flat_body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_anchor_ang_vel_w_future(self) -> torch.Tensor:
        return self.flat_body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
            self.motion_anchor_body_index,
        ]

    @property
    def flat_body_pos_w_future(self) -> torch.Tensor:
        return (
            self.flat_body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future,
            ]
            + self.env_origins_xy[:, None, None, :]
        )

    @property
    def flat_body_quat_w_future(self) -> torch.Tensor:
        return self.flat_body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def flat_body_lin_vel_w_future(self) -> torch.Tensor:
        return self.flat_body_lin_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    @property
    def flat_body_ang_vel_w_future(self) -> torch.Tensor:
        return self.flat_body_ang_vel_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future,
        ]

    ##
    # obs_* -- the motion source the priv observation group's imitation terms read,
    # selected by cfg.obs_motion_source. One decision point, so observations.py needs
    # no branching. Only these eight pairs are switchable; everything else is pinned
    # (the whole policy/teleop group to flat for deployment, and a few priv terms to
    # terrain because they are local deltas against the robot's own subterrain). See
    # docs/superpowers/specs/2026-08-01-priv-obs-motion-source-design.md
    ##

    @property
    def obs_anchor_pos_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_anchor_pos_w_future if flat else self.anchor_pos_w_future

    @property
    def obs_anchor_quat_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_anchor_quat_w_future if flat else self.anchor_quat_w_future

    @property
    def obs_anchor_lin_vel_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_anchor_lin_vel_w_future if flat else self.anchor_lin_vel_w_future

    @property
    def obs_anchor_ang_vel_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_anchor_ang_vel_w_future if flat else self.anchor_ang_vel_w_future

    @property
    def obs_joint_pos_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_joint_pos_future if flat else self.joint_pos_future

    @property
    def obs_joint_vel_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_joint_vel_future if flat else self.joint_vel_future

    @property
    def obs_body_pos_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_body_pos_w_future if flat else self.body_pos_w_future

    @property
    def obs_body_quat_w_future(self) -> torch.Tensor:
        flat = self.cfg.obs_motion_source == "flat"
        return self.flat_body_quat_w_future if flat else self.body_quat_w_future

    @property
    def flat_joint_pos_future_teleop(self) -> torch.Tensor:
        return self.flat_joint_pos_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future_teleop,
        ]

    @property
    def flat_anchor_pos_w_future_teleop(self) -> torch.Tensor:
        return (
            self.flat_body_pos_w_buf[
                torch.arange(self.num_envs, device=self.device)[:, None],
                self.buffer_indexes_future_teleop,
                self.motion_anchor_body_index,
            ]
            + self.env_origins_xy[:, None, :]
        )

    @property
    def flat_anchor_quat_w_future_teleop(self) -> torch.Tensor:
        return self.flat_body_quat_w_buf[
            torch.arange(self.num_envs, device=self.device)[:, None],
            self.buffer_indexes_future_teleop,
            self.motion_anchor_body_index,
        ]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_anchor_pos_metrics(self):
        """Record total, horizontal and vertical anchor position errors."""
        error = self.anchor_pos_w - self.robot_anchor_pos_w
        self.metrics["error_anchor_pos"] = torch.norm(error, dim=-1)
        self.metrics["error_anchor_pos_xy"] = torch.norm(error[:, :2], dim=-1)
        self.metrics["error_anchor_pos_z"] = error[:, 2].abs()

    def _update_metrics(self):
        self._update_anchor_pos_metrics()
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _build_group_terrain_cells(self) -> None:
        """Invert ``terrain_to_group_id[type, level]`` into, for each motion group,
        the list of terrain ``(type, level)`` cells that map to it.

        This is the mapping that lets the adaptive sampler place an env on a terrain
        patch matching the group it just drew. The flat group spans every level row
        (all flat tiles share the same data), other groups own a single cell. Stored
        as padded ``(num_groups, max_cells)`` tensors plus a per-group cell count.
        Set to ``None`` for plane terrain (nothing to assign).
        """
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            self.group_num_cells = None
            return
        ttg = self.terrain_to_group_id  # (num_types, num_levels), -1 where unused
        num_groups = self.motion.num_groups
        num_types, num_levels = ttg.shape
        ttg_cpu = ttg.cpu()
        cells_per_group: list[list[tuple[int, int]]] = [[] for _ in range(num_groups)]
        for t in range(num_types):
            for lv in range(num_levels):
                g = int(ttg_cpu[t, lv].item())
                if g >= 0:
                    cells_per_group[g].append((t, lv))
        max_cells = max((len(c) for c in cells_per_group), default=1)
        self.group_cell_type = torch.zeros(
            (num_groups, max_cells), dtype=torch.long, device=self.device
        )
        self.group_cell_level = torch.zeros(
            (num_groups, max_cells), dtype=torch.long, device=self.device
        )
        self.group_num_cells = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        for g, cells in enumerate(cells_per_group):
            self.group_num_cells[g] = len(cells)
            for k, (t, lv) in enumerate(cells):
                self.group_cell_type[g, k] = t
                self.group_cell_level[g, k] = lv

    def _build_adaptive_bins(self) -> None:
        """Build fixed-size phase bins for each available terrain-group and motion pair."""
        device = self.device
        bin_size = self.cfg.adaptive_bin_size
        G = self.motion.num_groups
        M = self.motion.num_motions
        lengths_cpu = self.motion.len_lookup.detach().cpu()  # (G, M)

        # pair_frame_starts[g, m] -> offset of clip (g, m) in the flattened global
        # frame index space; -1 for empty clips. frame_to_bin maps each global frame
        # to the global bin index it belongs to.
        self.pair_frame_starts = torch.full((G, M), -1, dtype=torch.long, device=device)

        bin_group_ids_list = []
        bin_motion_ids_list = []
        bin_starts_list = []
        bin_ends_list = []
        num_peer_bins_list = []
        frame_to_bin_chunks = []
        cur_bin = 0
        cur_frame = 0
        for g in range(G):
            for m in range(M):
                n = int(lengths_cpu[g, m].item())
                if n <= 0:
                    continue
                self.pair_frame_starts[g, m] = cur_frame
                starts = torch.arange(0, n, bin_size, device=device, dtype=torch.long)
                ends = torch.clamp(starts + bin_size, max=n)
                nb = len(starts)
                bin_group_ids_list.append(torch.full((nb,), g, dtype=torch.long, device=device))
                bin_motion_ids_list.append(torch.full((nb,), m, dtype=torch.long, device=device))
                bin_starts_list.append(starts)
                bin_ends_list.append(ends)
                num_peer_bins_list.append(torch.full((nb,), nb, dtype=torch.long, device=device))
                # Map each frame of this clip to its global bin index.
                bin_ids = torch.zeros(n, dtype=torch.long, device=device)
                bin_ids[starts[1:]] = 1
                bin_ids = bin_ids.cumsum(0) + cur_bin
                frame_to_bin_chunks.append(bin_ids)
                cur_bin += nb
                cur_frame += n

        assert bin_group_ids_list, "No valid (group, motion) clips found for adaptive sampling."
        self.adp_total_frames = cur_frame
        self.frame_to_bin = torch.cat(frame_to_bin_chunks)
        self.bin_group_ids = torch.cat(bin_group_ids_list)
        self.bin_group_weight = self.group_proportion_weight[self.bin_group_ids]
        self.bin_motion_ids = torch.cat(bin_motion_ids_list)
        self.bin_starts = torch.cat(bin_starts_list)
        self.bin_ends = torch.cat(bin_ends_list)
        self.bin_lengths = (self.bin_ends - self.bin_starts).float()
        num_peer_bins = torch.cat(num_peer_bins_list).float()
        self.num_bins = self.bin_group_ids.numel()

        # Per-bin weight: longer bins get proportionally more probability, then
        # (optionally) divide by the number of peer bins so every *clip* is covered
        # equally regardless of its length / bin count.
        self.bin_weights = self.bin_lengths / self.bin_lengths.mean()
        if self.cfg.adaptive_sequence_length_agnostic:
            self.bin_weights = self.bin_weights / num_peer_bins

        # Cumulative failure / episode counters. Initialized to the same value so the
        # initial failure rate is 1 for every bin (optimistic, uniform start). They
        # are not decayed: as the policy succeeds, episode counts grow and the failure
        # rate naturally falls -- this is the forgetting mechanism (per-bin behavior).
        init = float(self.cfg.adaptive_init_num_failures)
        self.bin_num_failures = torch.full((self.num_bins,), init, dtype=torch.float, device=device)
        self.bin_num_episodes = torch.full((self.num_bins,), init, dtype=torch.float, device=device)
        self.bin_sampling_prob = (
            torch.ones(self.num_bins, dtype=torch.float, device=device) / self.num_bins
        )
        # Diagnostic-only EMA of each bin's recent failure rate (see
        # `_update_bin_ema_failure_rate`); starts at the same pessimistic prior as the
        # cumulative rate above but decays independently and is never read by the
        # sampler itself.
        self.bin_ema_failure_rate = torch.full(
            (self.num_bins,), 1.0, dtype=torch.float, device=device
        )

    def _adaptive_global_frames(
        self,
        group_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Map per-env ``(group, motion, time_step)`` tuples to global frame indices,
        clamped to each clip's valid range so they can index ``frame_to_bin``."""
        lengths = self.motion.len_lookup[group_ids, motion_ids]
        ts = torch.clamp(time_steps, min=0)
        ts = torch.minimum(ts, lengths - 1)
        return self.pair_frame_starts[group_ids, motion_ids] + ts

    def _accumulate_adaptive_visits(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        consume_skips: bool = True,
    ) -> None:
        """Increment visits for executed commands, excluding just-resampled commands."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        skip = self._adaptive_skip_visit_once[env_ids]
        if consume_skips:
            self._adaptive_skip_visit_once[env_ids] = False
        env_ids = env_ids[~skip]
        if env_ids.numel() == 0:
            return

        frames = self._adaptive_global_frames(
            self.group_ids[env_ids],
            self.motion_ids[env_ids],
            self.time_steps[env_ids],
        )
        bins = self.frame_to_bin[frames]
        counts = torch.bincount(bins, minlength=self.num_bins).float()
        self.bin_num_episodes += counts / self.bin_lengths

    def _prepare_adaptive_resample(
        self,
        env_ids: torch.Tensor,
        *,
        visits_already_recorded: bool,
    ) -> None:
        """Record the old command's outcome before a resample overwrites its bin."""
        has_old_command = self.motion_lengths[env_ids] > 0
        old_env_ids = env_ids[has_old_command & ~self._adaptive_skip_visit_once[env_ids]]
        if old_env_ids.numel() > 0:
            if not visits_already_recorded:
                self._accumulate_adaptive_visits(old_env_ids)
            self._record_adaptive_failures(old_env_ids)
            self._update_bin_ema_failure_rate(old_env_ids)

        if not visits_already_recorded:
            self._adaptive_skip_visit_once[env_ids] = True

    def _record_adaptive_failures(self, env_ids: torch.Tensor) -> None:
        """Attribute failures to the bin where termination occurred.

        Must run *before* the resetting envs' ``group_ids`` / ``motion_ids`` /
        ``time_steps`` are overwritten. ``terminated`` excludes timeouts/motion-end,
        so only genuine failures contribute."""
        if len(env_ids) == 0:
            return
        terminated = self._env.termination_manager.terminated[env_ids]
        if not torch.any(terminated):
            return
        failed_env_ids = env_ids[terminated]
        frames = self._adaptive_global_frames(
            self.group_ids[failed_env_ids],
            self.motion_ids[failed_env_ids],
            self.time_steps[failed_env_ids],
        )
        bins = self.frame_to_bin[frames]
        counts = torch.bincount(bins, minlength=self.num_bins).float()
        self.bin_num_failures += counts * self.cfg.adaptive_failure_counts_multiplier

    def _update_bin_ema_failure_rate(self, env_ids: torch.Tensor) -> None:
        """Diagnostic-only EMA of each bin's *recent* failure rate.

        Unlike `bin_num_failures` / `bin_num_episodes` (cumulative for the whole run,
        the basis for `bin_sampling_prob`), this decays with `adaptive_ema_alpha` so it
        reflects roughly the last ``1/alpha`` episodes and can recover quickly once a
        bin is mastered, instead of only fading slowly as new episodes dilute old
        failures. It never feeds back into the sampling distribution.

        Every env in `env_ids` is an episode-end (termination, timeout, or motion-end),
        so unlike `_record_adaptive_failures` this uses the *whole* batch, not just the
        failed subset -- non-failures are the "success" observations the rate needs.
        Bins with several episode-ends in the same step are blended into one batch
        observation (`failures_this_step / events_this_step`) before the EMA update."""
        if len(env_ids) == 0:
            return
        terminated = self._env.termination_manager.terminated[env_ids].float()
        frames = self._adaptive_global_frames(
            self.group_ids[env_ids], self.motion_ids[env_ids], self.time_steps[env_ids]
        )
        bins = self.frame_to_bin[frames]
        event_counts = torch.bincount(bins, minlength=self.num_bins).float()
        failure_counts = torch.bincount(bins, weights=terminated, minlength=self.num_bins)
        touched = event_counts > 0
        batch_failure_rate = failure_counts[touched] / event_counts[touched]
        alpha = self.cfg.adaptive_ema_alpha
        self.bin_ema_failure_rate[touched] = (
            alpha * batch_failure_rate + (1 - alpha) * self.bin_ema_failure_rate[touched]
        )

    def _recompute_bin_probabilities(self) -> None:
        """Recompute the per-bin sampling distribution from current failure rates.

        Clip the failure rate to ``failure_rate_max_over_mean`` x mean (kills
        impossible-clip outliers), normalize, blend with a uniform baseline
        (``adaptive_uniform_ratio``), then apply the (length-agnostic) bin weights and
        each bin's terrain-group ``proportion`` weight (``bin_group_weight``, mean 1
        across groups -- a no-op when every sub-terrain's configured ``proportion`` is
        equal)."""
        failure_rate = self.bin_num_failures / self.bin_num_episodes
        upper = failure_rate.mean() * self.cfg.adaptive_failure_rate_max_over_mean
        fr_clipped = torch.clamp(failure_rate, min=0.0, max=upper.item())
        failure_based = fr_clipped / (fr_clipped.sum() + 1e-12)
        uniform = torch.ones_like(failure_based) / self.num_bins
        prob = (
            failure_based * (1 - self.cfg.adaptive_uniform_ratio)
            + uniform * self.cfg.adaptive_uniform_ratio
        )
        prob = prob * self.bin_weights
        prob = prob * self.bin_group_weight
        self.bin_sampling_prob = prob / prob.sum()

        # Metrics (broadcast scalars across envs for logging).
        p = self.bin_sampling_prob
        H = -(p * (p + 1e-12).log()).sum()
        H_norm = H / math.log(self.num_bins) if self.num_bins > 1 else H.new_tensor(1.0)
        uniform_prob = 1.0 / self.num_bins
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_effective_num_bins"][:] = 1.0 / (p * p).sum()
        self.metrics["sampling_top1_prob"][:] = p.max()
        self.metrics["sampling_prob_max_over_uniform"][:] = p.max() / uniform_prob
        # Bins whose probability exceeds 10x uniform ("significantly concentrated";
        # matches the convention used for this diagnostic elsewhere).
        self.metrics["sampling_num_concentrated_bins"][:] = (p > 10.0 * uniform_prob).sum().float()
        evidence = self.bin_num_episodes - self.cfg.adaptive_init_num_failures
        self.metrics["sampling_evidence_min"][:] = evidence.min()
        self.metrics["sampling_evidence_max"][:] = evidence.max()
        self.metrics["sampling_evidence_mean"][:] = evidence.mean()
        self.metrics["sampling_evidence_median"][:] = evidence.median()
        self.metrics["sampling_failure_rate_min"][:] = failure_rate.min()
        self.metrics["sampling_failure_rate_max"][:] = failure_rate.max()
        self.metrics["sampling_failure_rate_mean"][:] = failure_rate.mean()
        self.metrics["sampling_1st_percentile_avg_failure_rate"][:] = torch.topk(
            failure_rate,
            k=max(1, int(0.01 * self.num_bins)),
            largest=True,
            sorted=False,
        )[0].mean()
        self.metrics["mastery_ratio"][:] = (failure_rate < 0.01).float().mean()
        self.metrics["mastery_ratio_recent"][:] = (
            (self.bin_ema_failure_rate < self.cfg.adaptive_ema_mastery_threshold).float().mean()
        )

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        """Sample a new ``(group, motion, start time_step)`` for ``env_ids`` from the
        per-bin failure-rate distribution: one multinomial over bins selects the
        terrain group, the motion, and the phase, then a uniform frame is drawn
        within the chosen bin. Terrain assignment from the sampled group is done by
        the caller (``_sample_and_load``)."""
        self._recompute_bin_probabilities()
        n = len(env_ids)
        sampled_bins = torch.multinomial(self.bin_sampling_prob, num_samples=n, replacement=True)
        self.group_ids[env_ids] = self.bin_group_ids[sampled_bins]
        self.motion_ids[env_ids] = self.bin_motion_ids[sampled_bins]
        bin_starts = self.bin_starts[sampled_bins]
        bin_ends = self.bin_ends[sampled_bins]
        time_steps = (
            torch.rand(n, device=self.device) * (bin_ends - bin_starts).float()
        ).floor().long() + bin_starts
        # Optionally start a little before the difficult segment so the policy
        # practices entering it (the configured pre_failure_sample_window).
        window = self.cfg.adaptive_pre_failure_sample_window
        if window > 0:
            offset = torch.randint(window, (n,), device=self.device)
            time_steps = (time_steps - offset).clamp_min(0)
        self.start_time_steps[env_ids] = time_steps

    def _assign_terrain_from_group(self, env_ids: torch.Tensor) -> None:
        """Place each env on a terrain patch whose ``(type, level)`` maps to the
        group the env just sampled, and update its origin. No-op on plane terrain."""
        if getattr(self, "group_num_cells", None) is None:
            return
        groups = self.group_ids[env_ids]
        num_cells = self.group_num_cells[groups]  # (n,)
        k = (torch.rand(len(env_ids), device=self.device) * num_cells.float()).long()
        k = torch.minimum(k, (num_cells - 1).clamp(min=0))
        types = self.group_cell_type[groups, k]
        levels = self.group_cell_level[groups, k]
        self._terrain.terrain_types[env_ids] = types
        self._terrain.terrain_levels[env_ids] = levels
        self._env.scene.env_origins[env_ids] = self._terrain.terrain_origins[levels, types]

    def _build_group_valid_motions(self) -> None:
        """For each group, the motion ids that have a non-empty clip on it.

        Sparse (union) packs leave some ``(group, motion)`` pairs empty, so the
        non-adaptive samplers must restrict their motion choice to the clips that
        actually exist on an env's (externally fixed) group. Stored as a padded
        ``(num_groups, max_valid)`` table plus a per-group valid count.
        """
        valid = self.motion.len_lookup > 0  # (G, M)
        counts = valid.sum(dim=1)  # (G,)
        max_valid = max(1, int(counts.max().item()))
        table = torch.zeros(
            (self.motion.num_groups, max_valid), dtype=torch.long, device=self.device
        )
        for g in range(self.motion.num_groups):
            ids = torch.nonzero(valid[g], as_tuple=False).flatten()
            table[g, : ids.numel()] = ids
        self.group_valid_motions = table
        self.group_num_valid_motions = counts
        # Groups with at least one valid clip -- the redraw pool `_resolve_group_
        # from_terrain` uses when an env lands on an empty group. Always non-empty:
        # flat is guaranteed a full clip set (see `GroupedMotionDataset`).
        self._valid_group_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()

    def set_fixed_terrain(
        self, env_ids: torch.Tensor, terrain_type: int, terrain_level: int
    ) -> None:
        """Place ``env_ids`` on one generated terrain cell before resetting them."""
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            raise ValueError("Terrain switching requires a generated terrain scene.")

        num_levels, num_types = self._terrain.terrain_origins.shape[:2]
        if not 0 <= terrain_type < num_types:
            raise ValueError(f"terrain type must be in [0, {num_types - 1}], got {terrain_type}.")
        if not 0 <= terrain_level < num_levels:
            raise ValueError(
                f"terrain level must be in [0, {num_levels - 1}], got {terrain_level}."
            )

        group = int(self.terrain_to_group_id[terrain_type, terrain_level].item())
        if group < 0 or self.group_num_valid_motions[group] == 0:
            raise ValueError(
                f"fixed_terrain=(type={terrain_type}, level={terrain_level}) maps "
                f"to terrain group {group}, which has no packed motion data."
            )

        self._terrain.terrain_types[env_ids] = terrain_type
        self._terrain.terrain_levels[env_ids] = terrain_level
        self._env.scene.env_origins[env_ids] = self._terrain.terrain_origins[
            terrain_level, terrain_type
        ]

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Create live terrain type/level controls in the Viser viewer."""
        del get_env_idx, on_change
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            return

        num_levels, num_types = self._terrain.terrain_origins.shape[:2]
        initial_type, initial_level = self.cfg.fixed_terrain or (0, 0)
        # The CLI value has already positioned every env during construction. From
        # here on, each env's terrain tensor is authoritative so GUI changes can be
        # applied to the selected env without affecting later resets of the others.
        self.cfg.fixed_terrain = None

        with server.gui.add_folder(name.capitalize()):
            terrain_type = server.gui.add_number(
                "Terrain type",
                min=0,
                max=num_types - 1,
                step=1,
                initial_value=initial_type,
            )
            terrain_level = server.gui.add_number(
                "Terrain level",
                min=0,
                max=num_levels - 1,
                step=1,
                initial_value=initial_level,
            )
            all_envs = server.gui.add_checkbox("All envs", initial_value=False)

            def queue_switch(_) -> None:
                if request_action is not None:
                    request_action(
                        "CUSTOM",
                        {
                            "type": "terrain_switch",
                            "command": name,
                            "terrain_type": int(terrain_type.value),
                            "terrain_level": int(terrain_level.value),
                            "all_envs": all_envs.value,
                        },
                    )

            terrain_type.on_update(queue_switch)
            terrain_level.on_update(queue_switch)

    def _resolve_group_from_terrain(self, env_ids: torch.Tensor) -> None:
        """Set ``group_ids`` from each env's (externally assigned) terrain patch,
        honoring ``fixed_terrain``. No-op on plane terrain (group stays 0).

        An env that lands on a terrain patch whose group has no packed motion data
        is redrawn onto a random group that does have data, and re-placed (via
        ``_assign_terrain_from_group``) on a terrain patch belonging to that group
        instead -- terrain-group parity may be looser than the sampled group, but
        there's simply nothing to track on the original patch. If ``fixed_terrain``
        was set explicitly, an empty group is a hard error instead of a silent
        redirect, since the caller asked for that exact patch on purpose.
        """
        if self._terrain is None or self._terrain.cfg.terrain_type == "plane":
            return
        if self.cfg.fixed_terrain is not None:
            terrain_type, terrain_level = self.cfg.fixed_terrain
            group = int(self.terrain_to_group_id[terrain_type, terrain_level].item())
            if self.group_num_valid_motions[group] == 0:
                raise ValueError(
                    f"fixed_terrain=(type={terrain_type}, level={terrain_level}) maps "
                    f"to terrain group {group}, which has no packed motion data."
                )
            self._terrain.terrain_types[env_ids] = terrain_type
            self._terrain.terrain_levels[env_ids] = terrain_level
            self._env.scene.env_origins[env_ids] = self._terrain.terrain_origins[
                terrain_level, terrain_type
            ]
            self.group_ids[env_ids] = group
            return

        self.group_ids[env_ids] = self.terrain_to_group_id[
            self._terrain.terrain_types[env_ids],
            self._terrain.terrain_levels[env_ids],
        ]
        invalid_mask = self.group_num_valid_motions[self.group_ids[env_ids]] == 0
        if not bool(invalid_mask.any()):
            return
        redraw_ids = env_ids[invalid_mask]
        k = torch.randint(len(self._valid_group_ids), (len(redraw_ids),), device=self.device)
        self.group_ids[redraw_ids] = self._valid_group_ids[k]
        self._assign_terrain_from_group(redraw_ids)

    def _uniform_sampling(self, env_ids: torch.Tensor):
        """Sample a motion uniformly among the clips valid on each env's group
        (``group_ids`` must already be resolved), then a uniform start frame."""
        groups = self.group_ids[env_ids]
        nvalid = self.group_num_valid_motions[groups]  # (n,)
        k = (torch.rand(len(env_ids), device=self.device) * nvalid.float()).long()
        k = torch.minimum(k, (nvalid - 1).clamp(min=0))
        self.motion_ids[env_ids] = self.group_valid_motions[groups, k]
        lengths = self.motion.len_lookup[groups, self.motion_ids[env_ids]]
        max_start_time = (lengths - self._env.max_episode_length).clamp(
            min=0
        )  # ensure enough frames for the episode
        phases = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)  # in [0, 1)
        self.start_time_steps[env_ids] = (phases * max_start_time).long()

    def _start_sampling(self, env_ids: torch.Tensor):
        """Cycle each env through the clips valid on its (fixed) group, one per
        reset (useful in play mode). ``motion_lengths[env_id] == 0`` means this
        env has never been resampled before (it's still the zero-init value, only
        ever overwritten below in `_resample_command`), so it lands on the
        group's first valid motion instead of advancing past it."""
        for env_id in env_ids.tolist():
            g = int(self.group_ids[env_id])
            nvalid = int(self.group_num_valid_motions[g])
            if nvalid == 0:
                # Should be unreachable: `_resolve_group_from_terrain` (called
                # before this) redraws env_ids off any empty group, so `group_ids`
                # never points at an nvalid==0 group here. Kept as a defensive
                # fallback -- leave on the padded default (0), matching the
                # nvalid==0 fallthrough in `_uniform_sampling`. Slicing [:0] below
                # would make valid[0] raise.
                self.motion_ids[env_id] = 0
                continue
            valid = self.group_valid_motions[g, :nvalid]
            if int(self.motion_lengths[env_id]) == 0:
                next_pos = 0
            else:
                pos = (valid == int(self.motion_ids[env_id])).nonzero(as_tuple=False)
                next_pos = (int(pos[0]) + 1) % nvalid if pos.numel() > 0 else 0
            self.motion_ids[env_id] = int(valid[next_pos])
        self.start_time_steps[env_ids] = 0

    def _sample_and_load(self, env_ids: torch.Tensor):
        """Sample a fresh motion + start for `env_ids` and refill its FK cache
        (this is the data-movement step)."""
        # 1-2. Sample the new (group, motion, start) and set the terrain patch.
        #
        # Adaptive: the bin draw fixes the group, so the group is authoritative and
        # *drives* the terrain/origin assignment (the fine-grained terrain
        # resampling the bin curriculum needs). It also consumes the just-ended
        # episode's motion_ids/time_steps for its failure stats before overwriting.
        #
        # Non-adaptive (uniform/start): the terrain (hence group) is external (mjlab
        # init or fixed_terrain), so resolve the group *first*, then sample a motion
        # that actually has data on that group (sparse-safe).
        if env_ids.numel() > 0:
            if self.cfg.sampling_mode == "adaptive":
                self._adaptive_sampling(env_ids)
                self._assign_terrain_from_group(env_ids)
            else:
                self._resolve_group_from_terrain(env_ids)
                if self.cfg.sampling_mode == "start":
                    self._start_sampling(env_ids)
                else:
                    assert self.cfg.sampling_mode == "uniform"
                    self._uniform_sampling(env_ids)

        # 3. Commit episode timing for the new episode. The clip length is read per
        #    (group, motion) so a decoupled group advances over the right clip.
        #    `start_time_steps` briefly holds the sampled phase here -- step 4 below
        #    overwrites it with the FK cache's own origin.
        self.time_steps[env_ids] = self.start_time_steps[env_ids]
        self.motion_lengths[env_ids] = self.motion.len_lookup[
            self.group_ids[env_ids], self.motion_ids[env_ids]
        ]

        # 4. Refill the sliding-window FK cache around the new phase.
        self._refill_group_cache(env_ids)

    def _advance_time_steps(self) -> torch.Tensor:
        """Advance phases, count natural completions, and resample completed clips."""
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_lengths)[0]
        if env_ids.numel() > 0:
            self.natural_completion_counts[env_ids] += 1
            self._resample_command(env_ids, visits_already_recorded=True)
            # Command updates run after the environment's forward pass. Refresh
            # the new pose before relative targets, ground queries and cameras read it.
            self._env.sim.forward()
        return env_ids

    def _resample_command(
        self,
        env_ids: torch.Tensor,
        *,
        visits_already_recorded: bool = False,
    ):
        if self.cfg.sampling_mode == "adaptive":
            self._prepare_adaptive_resample(
                env_ids,
                visits_already_recorded=visits_already_recorded,
            )
        self._resample_command_impl(env_ids)

    def _resample_command_impl(self, env_ids: torch.Tensor) -> None:
        """Sample the replacement command and reset simulator state."""
        # Sample a new clip and refill its window from the FK cache. Every reset
        # always draws fresh (no buffered-clip reuse) -- refilling is a cheap
        # on-device gather + FK now, so there's no data-movement cost to amortize.
        self._sample_and_load(env_ids)

        # Reset-time baseline for the flat-reference debug ghost (see
        # _debug_vis_impl) -- gated on show_src_ghost so non-viz runs pay nothing.
        if self.cfg.viz.show_src_ghost:
            self._src_ghost_reset_pos_w[env_ids] = self.anchor_pos_w[env_ids]
            self._src_ghost_reset_flat_pos_w[env_ids] = self.flat_anchor_pos_w[env_ids]
            self._src_ghost_reset_quat_w[env_ids] = yaw_quat(self.anchor_quat_w[env_ids])

        # Clear the reference contact schedule so its hysteresis does not carry over
        # from the previous episode; update_ee_contact_active() re-seeds it this step.
        # The cached ground heights go with it: a resampled env is on a different
        # terrain patch entirely, so the old values are not a valid fallback for a
        # missed ray any more.
        self.ee_contact_active[env_ids] = False
        self.target_ee_ground_z[env_ids] = 0.0
        self.robot_ee_ground_z[env_ids] = 0.0
        self._robot_ground_refresh_step = -1
        self.target_ee_clearance[env_ids] = 0.0
        self.ee_contact_viz_standing[env_ids] = False

        # Robot state reset for all resetting envs (reads from the now-valid buffer).
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,  # type: ignore
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)

        root_state = torch.cat(
            [
                root_pos[env_ids],
                root_ori[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.robot.reset(env_ids=env_ids)

    def _update_adaptive_sampling(self) -> None:
        """Account for this manager update, advance phases, then consume reset skips."""
        self._accumulate_adaptive_visits(consume_skips=False)
        self._advance_time_steps()
        self._adaptive_skip_visit_once[:] = False

    def _update_command(self):
        if self.cfg.sampling_mode == "adaptive":
            self._update_adaptive_sampling()
        else:
            self._advance_time_steps()

        # Refill the sliding-window cache for every env whose phase (or its
        # future-lookahead window) has walked outside the cached range. No-op for
        # envs just resampled above (freshly filled, so this always hits for them).
        self._ensure_group_cache()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        self.update_ee_contact_active()
        self.update_ee_contact_viz()

    @property
    def feet_standing(self) -> torch.Tensor:
        """Feet columns of the reference EE contact schedule."""
        return self.ee_contact_active[:, : self._num_feet]

    @property
    def hands_standing(self) -> torch.Tensor:
        """Hand columns of the reference EE contact schedule."""
        return self.ee_contact_active[:, self._num_feet :]

    @property
    def robot_foot_ground_z(self) -> torch.Tensor:
        return self.robot_ee_ground_z[:, : self._num_feet]

    @property
    def robot_hand_ground_z(self) -> torch.Tensor:
        return self.robot_ee_ground_z[:, self._num_feet :]

    @property
    def target_foot_clearance(self) -> torch.Tensor:
        return self.target_ee_clearance[:, : self._num_feet]

    @property
    def target_hand_clearance(self) -> torch.Tensor:
        return self.target_ee_clearance[:, self._num_feet :]

    def _ensure_ground_raycaster(self) -> EEGroundRaycaster:
        """Built lazily: the sim's model/geom ids are not available in __init__."""
        if self._ground_raycaster is None:
            self._ground_raycaster = EEGroundRaycaster(
                num_envs=self.num_envs,
                # One ray per target EE plus one per robot EE, packed into one call.
                num_rays=2 * len(self.ee_body_names),
                device=str(self.device),
                terrain_geom_groups=resolve_terrain_geom_groups(
                    self._env.sim.mj_model, list(self.robot.indexing.geom_ids)
                ),
                origin_z=self.cfg.ground_raycast_origin_z,
            )
        return self._ground_raycaster

    def _ensure_anchor_ground_raycaster(self) -> EEGroundRaycaster:
        """Build the pre-termination two-anchor raycaster lazily."""
        if self._anchor_ground_raycaster is None:
            self._anchor_ground_raycaster = EEGroundRaycaster(
                num_envs=self.num_envs,
                num_rays=2,
                device=str(self.device),
                terrain_geom_groups=resolve_terrain_geom_groups(
                    self._env.sim.mj_model, list(self.robot.indexing.geom_ids)
                ),
                origin_z=self.cfg.ground_raycast_origin_z,
            )
        return self._anchor_ground_raycaster

    def refresh_anchor_ground_z(self):
        """Query terrain under target and robot anchors at termination time."""
        raycaster = self._ensure_anchor_ground_raycaster()
        ground_z = raycaster.query(
            self._env.sim.wp_model,
            self._env.sim.wp_data,
            torch.stack([self.anchor_pos_w[:, :2], self.robot_anchor_pos_w[:, :2]], dim=1),
            torch.stack([self.target_anchor_ground_z, self.robot_anchor_ground_z], dim=1),
        )
        self.target_anchor_ground_z = ground_z[:, 0]
        self.robot_anchor_ground_z = ground_z[:, 1]

    def update_ee_contact_active(self):
        """Update the reference EE-contact schedule from the TERRAIN-ADAPTED target
        motion.

        Historically this read the *flat* source clip instead, because the old
        deformation-energy retargeter's terrain output had contact timing too
        unreliable to reward against, so the flat clip's cadence was used as a
        stand-in. The keyframe-IK pipeline (``export_smoothed_ik_motion.py``) now
        plants each foothold against the real terrain, so the target motion's own
        timing is the physically correct thing to track -- see
        ``docs/superpowers/specs/2026-07-31-ee-contact-reward-design.md``.

        Per-EE boolean hysteresis: an EE counts as "in contact" when it is low and
        slow relative to the root speed, and leaves that state when it rises or
        speeds up. Height is measured against the terrain directly beneath that EE,
        not absolute world z -- otherwise an EE resting on a raised step never
        registers as contacting. Read by the air_time rewards AND
        motion_ee_contact_state (observations.py). No-op if no EEs are configured.
        """
        if self.motion_ee_body_indexes.numel() == 0:
            return
        self.refresh_ee_ground_z()
        self.update_ee_contact_from_ground_z()

    def refresh_ee_ground_z(self):
        """Raycast terrain under target and robot EEs after command updates.

        One call with ``nray = 2 * n_ee``: ``mujoco_warp.rays`` always launches over
        every env (``dim=(d.nworld, pnt.shape[1])``), so ``nray`` is the only
        dimension worth batching along. Query order is target EEs, robot EEs.
        """
        raycaster = self._ensure_ground_raycaster()
        n_ee = len(self.ee_body_names)
        target_xy = self.body_pos_w[:, self.motion_ee_body_indexes, :2]
        robot_xy = self.robot.data.body_link_pos_w[:, self.robot_ee_body_indexes, :2]
        ground_z = raycaster.query(
            self._env.sim.wp_model,
            self._env.sim.wp_data,
            torch.cat([target_xy, robot_xy], dim=1),
            torch.cat([self.target_ee_ground_z, self.robot_ee_ground_z], dim=1),
        )
        self.target_ee_ground_z = ground_z[:, :n_ee]
        self.robot_ee_ground_z = ground_z[:, n_ee:]

    def refresh_robot_ee_ground_z(self):
        """Refresh current robot clearance once per reward step, keeping reference state intact."""
        step = self._env.common_step_counter
        if self._robot_ground_refresh_step == step or not self.ee_body_names:
            return
        if self._robot_ground_raycaster is None:
            self._robot_ground_raycaster = EEGroundRaycaster(
                num_envs=self.num_envs,
                num_rays=len(self.ee_body_names),
                device=str(self.device),
                terrain_geom_groups=resolve_terrain_geom_groups(
                    self._env.sim.mj_model, list(self.robot.indexing.geom_ids)
                ),
                origin_z=self.cfg.ground_raycast_origin_z,
            )
        self.robot_ee_ground_z = self._robot_ground_raycaster.query(
            self._env.sim.wp_model,
            self._env.sim.wp_data,
            self.robot.data.body_link_pos_w[:, self.robot_ee_body_indexes, :2],
            self.robot_ee_ground_z,
        )
        self._robot_ground_refresh_step = step

    def update_ee_contact_from_ground_z(self):
        """The hysteresis half of update_ee_contact_active, split out so it can be
        exercised without a live sim (``refresh_ee_ground_z`` needs one)."""
        target_pos_w = self.body_pos_w[:, self.motion_ee_body_indexes]
        ee_vel_w = self.body_lin_vel_w[:, self.motion_ee_body_indexes]
        # Both the hysteresis input and the dense rewards' shaping target.
        self.target_ee_clearance = target_pos_w[..., 2] - self.target_ee_ground_z
        # Normalize velocity thresholds by root planar speed (min 1.0 m/s).
        root_vxy = self.body_lin_vel_w[:, 0, :2].norm(dim=-1, keepdim=True).clamp_min(1.0)
        self.ee_contact_active = ee_contact_hysteresis_step(
            self.target_ee_clearance,
            ee_vel_w[..., :2].norm(dim=-1),
            ee_vel_w[..., 2].abs(),
            root_vxy,
            self.ee_contact_active,
            self.cfg.ee_contact_cfg,
        )

    def update_ee_contact_viz(self):
        """Live per-EE contact state for debug visualization only (not consumed by
        any reward). Deliberately still reads the FLAT reference buffers with an
        absolute-z threshold, unlike update_ee_contact_active: this exists to show
        what the *source* clip was doing, so it is not affected by that method's
        switch to the terrain-adapted target. Gated on show_ee_contact so normal
        training pays nothing."""
        if not self.cfg.viz.show_ee_contact or self.ee_contact_viz_body_indexes.numel() == 0:
            return
        pos_w = self.flat_body_pos_w[:, self.ee_contact_viz_body_indexes]
        vel_w = self.flat_body_lin_vel_w[:, self.ee_contact_viz_body_indexes]
        root_vxy = self.flat_body_lin_vel_w[:, 0, :2].norm(dim=-1, keepdim=True).clamp_min(1.0)
        z = pos_w[..., 2]
        vxy = vel_w[..., :2].norm(dim=-1)
        vz = vel_w[..., 2].abs()
        self.ee_contact_viz_standing = ee_contact_hysteresis_step(
            z, vxy, vz, root_vxy, self.ee_contact_viz_standing, self.cfg.ee_contact_viz_cfg
        )

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot or frames based on visualization mode."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self._ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
                qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

                # Optionally visualize the flat reference motion, offset from the
                # primary ghost to avoid overlap. Skipped for envs currently sampling
                # the flat/source group itself -- there's no terrain deviation to
                # compare against.
                if (
                    self.cfg.viz.show_src_ghost
                    and int(self.group_ids[batch]) != self.source_group_index
                ):
                    if self._src_ghost_model is None:
                        self._src_ghost_model = copy.deepcopy(self._env.sim.mj_model)
                        self._src_ghost_model.geom_rgba[:] = self._src_ghost_color

                    # Position: reset-time terrain anchor + a fixed offset (given in
                    # the anchor's reset-time heading frame, so it stays e.g. "to the
                    # robot's left" regardless of which way the env is facing) + how
                    # far the flat clip has moved since that reset -- so height stays
                    # pinned to the reset height (ramps/stairs don't drag it up) while
                    # x/y still follow the flat clip's own walking motion.
                    offset_b = torch.tensor(
                        self.cfg.viz.src_ghost_offset_b, device=self.device, dtype=torch.float32
                    )
                    offset_w = quat_apply(self._src_ghost_reset_quat_w[batch], offset_b)
                    flat_delta = (
                        self.flat_anchor_pos_w[batch] - self._src_ghost_reset_flat_pos_w[batch]
                    )
                    ghost_pos = (
                        (self._src_ghost_reset_pos_w[batch] + offset_w + flat_delta).cpu().numpy()
                    )

                    qpos_src = np.zeros(self._env.sim.mj_model.nq)
                    qpos_src[free_joint_q_adr[0:3]] = ghost_pos
                    # use flat root orientation from the flat buffer
                    qpos_src[free_joint_q_adr[3:7]] = self.flat_anchor_quat_w[batch].cpu().numpy()
                    # flat joint positions are in flat_joint_pos
                    qpos_src[joint_q_adr] = self.flat_joint_pos[batch].cpu().numpy()

                    visualizer.add_ghost_mesh(
                        qpos_src,
                        model=self._src_ghost_model,
                        label=f"src_ghost_{batch}",
                    )

        elif self.cfg.viz.mode == "frames":
            for batch in env_indices:
                desired_body_pos = self.body_pos_w[batch].cpu().numpy()
                desired_body_quat = self.body_quat_w[batch]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
                current_body_quat = self.robot_body_quat_w[batch]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.cfg.body_names):
                    visualizer.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_{batch}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    visualizer.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_{batch}",
                    )

                desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
                desired_anchor_quat = self.anchor_quat_w[batch]
                desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=desired_anchor_pos,
                    rotation_matrix=desired_rotation_matrix,
                    scale=0.1,
                    label=f"desired_anchor_{batch}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )

                current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
                current_anchor_quat = self.robot_anchor_quat_w[batch]
                current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=current_anchor_pos,
                    rotation_matrix=current_rotation_matrix,
                    scale=0.15,
                    label=f"current_anchor_{batch}",
                )

        # EE contact markers: independent of mode (ghost/frames), same pattern as
        # show_src_ghost being an additive flag layered on top of `mode`. Position
        # uses body_pos_w (the rendered, terrain-conditioned ghost's actual world
        # position) even though the contact boolean itself is computed from the
        # flat reference buffers in update_ee_contact_viz.
        if self.cfg.viz.show_ee_contact:
            ee_body_indexes = self.ee_contact_viz_body_indexes.tolist()
            for batch in env_indices:
                for pos, color in ee_sphere_specs(
                    self.body_pos_w[batch].cpu().numpy(),
                    self.ee_contact_viz_standing[batch].cpu().numpy(),
                    ee_body_indexes,
                ):
                    visualizer.add_sphere(pos, EE_SPHERE_RADIUS, color)


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    motion_file: str
    anchor_body_name: str
    body_names: tuple[str, ...]
    entity_name: str
    # Reference EE-contact schedule (ee_contact_active) for the reference air_time
    # rewards. Both name tuples must be subsets of body_names; empty disables that
    # group. Feet columns always precede hand columns in the resulting schedule.
    # ee_contact_cfg is the same hysteresis dataclass the offline pipeline uses
    # (utils/contact_phase.py); its defaults are the reference G1 profile.
    # Which motion the *priv* observation group's imitation terms describe: the flat
    # source clip, or the terrain-adapted target. Does NOT change what is optimized --
    # rewards/terminations always read the terrain-side target regardless (there is no
    # `flat_` anywhere in rewards.py/terminations.py). It only changes what the policy
    # is shown.
    #
    # Defaults to "terrain" so the privileged side sees the same motion it is scored
    # against; "flat" reproduces the older behaviour, in which priv was judged on the
    # terrain target while only being shown the flat reference and had to infer the
    # difference.
    #
    # The `policy` group is deliberately NOT covered and must stay flat: it is what a
    # deployed policy consumes, and on the real robot there is no terrain-adapted
    # motion at all -- the teleoperator's reference is flat. See
    # docs/superpowers/specs/2026-08-01-priv-obs-motion-source-design.md
    obs_motion_source: Literal["flat", "terrain"] = "terrain"
    # Allocate and FK-expand the flat-source cache. Disable only when every active
    # observation/debug consumer is terrain-only (the standalone teacher training
    # configs do this); deployable policy observations always require the flat cache.
    flat_motion_enabled: bool = True
    foot_body_names: tuple[str, ...] = ()
    hand_body_names: tuple[str, ...] = ()
    ee_contact_cfg: EEContactCfg = field(default_factory=EEContactCfg)
    # World z that every EE ground raycast is fired downward from. Must clear the
    # tallest terrain in the scene.
    ground_raycast_origin_z: float = 10.0
    # Live per-EE contact state for debug visualization only (VizCfg.show_ee_contact) --
    # independent of ee_contact_active above, no reward reads this. Deliberately still
    # sourced from the FLAT clip (see update_ee_contact_viz). Must be a subset of
    # body_names; empty disables it.
    ee_contact_viz_body_names: tuple[str, ...] = ()
    ee_contact_viz_cfg: EEContactCfg = field(default_factory=EEContactCfg)
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
    # --- per-bin per-bin adaptive sampling over (group, motion, phase) ---
    # Every valid (group, motion) clip is split into fixed-size bins; each bin tracks
    # a cumulative failure/episode ratio. Drawing a bin fixes the terrain group, the
    # motion, and the start phase at once, and harder bins are sampled more often.
    adaptive_bin_size: int = 50  # frames per bin
    adaptive_uniform_ratio: float = 0.1  # fraction blended toward a uniform baseline
    adaptive_failure_rate_max_over_mean: float = 200.0  # clip failure rate to N x mean
    adaptive_init_num_failures: float = 1.0  # optimistic init: failure rate starts at 1
    adaptive_sequence_length_agnostic: bool = True  # sample every clip equally often
    adaptive_failure_counts_multiplier: float = 1.0  # weight of each failure event
    adaptive_pre_failure_sample_window: int = 200  # start up to N frames before the bin
    # Diagnostic-only EMA of each bin's *recent* failure rate (does not feed back into
    # bin_sampling_prob -- the sampler stays on the cumulative failure rate above).
    # alpha=0.1 gives an effective window of ~1/alpha=10 episodes.
    adaptive_ema_alpha: float = 0.1
    # A bin counts as "recently mastered" once its EMA failure rate drops below this
    # (default: <=~1 failure in the last ~10 episodes, i.e. >=90% recent success).
    adaptive_ema_mastery_threshold: float = 0.1
    future_window_size: int = 20
    lookahead_mode: Literal["linear", "quadratic", "exp"] = "linear"
    lookahead_stride: int = 5
    # Explicit sparse future/past offset list (frames from the current phase; negative
    # = past frames of the same clip). Overrides future_window_size/lookahead_mode/
    # lookahead_stride when set (that linear/quadratic/exp schedule is now only
    # reachable by explicitly passing future_offsets=None). Defaults to the configured G1
    # teacher future_steps, same default as tasks.tracking's
    # MotionCommandCfg.future_offsets -- previously this defaulted to None and
    # silently fell back to the far denser/longer synthetic
    # future_window_size=20/stride=5 schedule (max offset 96 instead of the reference 20).
    # Teacher horizon offset list. Paired with teleop `future_offsets_teleop` below.
    # Negative offsets must fit within cache_request_range's backward end in the
    # packed backend (validated at init).
    future_offsets: tuple[int, ...] | None = (0, 1, 2, 4, 8, 12, 16, 20, -1, -2, -4, -8)
    # Teleop ("policy") horizon offset list -- a separate, much shorter window for the
    # deployable policy's observations (named after the real-robot teleop deployment
    # pipeline; reference calls this `horizon: student`). the reference G1 profile sets a shorter,
    # more past-heavy list than the teacher future_steps above -- teacher looks 20
    # frames ahead, student only 6 (and further back: -16 vs -8).
    future_offsets_teleop: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
    rank: int = 0
    world_size: int = 1
    # --- Packed (per-group directory) backend only: sliding-window FK cache sizing ---
    # Literally the configured ProgressiveMultiMotionDataset [start, end) pair (see
    # active_adaptation/utils/multimotion.py), not a derived margin/formula. Mirrors
    # tasks.tracking's MotionCommandCfg.cache_request_range/cache_fill_range:
    #
    # cache_request_range: the range of offsets, relative to the current phase, that
    # is ever actually queried. Must cover [-past_max, future_max+1) where past_max/
    # future_max come from future_offsets -- validated at init (raises if it doesn't).
    # Must include 0 and satisfy start < end.
    #
    # cache_fill_range: the range that actually gets fetched/FK-expanded and cached
    # on a refill (the buffer is cache_fill_range[1] - cache_fill_range[0] frames
    # long). Must contain cache_request_range -- validated at init. The gap on the
    # forward side (cache_fill_range[1] - cache_request_range[1]) is pure
    # amortization slack: a refill stays valid for that many extra steps of forward
    # phase progress before the next miss. the reference G1 defaults below put all of that
    # slack forward and none backward (fill_start == request_start exactly) -- there's
    # no amortization benefit to backward slack, since nothing ever walks backward
    # across resets/steps.
    #
    # To reproduce whole-clip residency (no phase-drift-driven refills within an
    # episode) instead of a small sliding window, widen cache_fill_range's forward end
    # to >= max_episode_length + future_window_size.
    cache_request_range: tuple[int, int] = (-20, 24)
    cache_fill_range: tuple[int, int] = (-20, 100)
    fixed_terrain: tuple[int, int] | None = (
        None  # if set, override terrain type and level for all envs
    )

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
        show_src_ghost: bool = True
        src_ghost_color: tuple[float, float, float, float] = (0.2, 0.8, 0.8, 0.5)
        # Fixed offset applied to the flat-reference ghost so it doesn't overlap the
        # primary (terrain-target) ghost, given in the anchor's reset-time heading
        # frame (yaw-only -- pitch/roll ignored) rather than the world frame, so it
        # stays e.g. "to the robot's left" regardless of which way the env faces.
        src_ghost_offset_b: tuple[float, float, float] = (0.0, 0.5, 0.0)
        # Independent of mode -- draws a colored sphere per ee_contact_viz_body_names
        # EE (green=contact, red=swing) via update_ee_contact_viz.
        show_ee_contact: bool = False

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
