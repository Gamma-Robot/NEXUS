"""Reconstruct dense joint and root-height corrections from sparse IK keyframes.

Root XY follows rigid patch placement; orientation remains unchanged."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from terrain_motion_adaptation.optimization.keyframe_ik import IKClipResult


def infill_dense_trajectory(
    sparse: IKClipResult,
    source_joint_pos: np.ndarray,
    source_root_pos_w: np.ndarray,
    offset_xy: np.ndarray,
    num_frames: int,
) -> IKClipResult:
    """Blend ``sparse``'s IK corrections into a full ``num_frames``-long dense
    trajectory, keeping the source motion's own shape everywhere but at the
    sparse keyframes themselves.

    For ``joint_pos`` (every DOF) and ``root_pos_w``'s z component: the
    correction ``sparse - source`` at each keyframe is PCHIP-interpolated
    (shape-preserving, no overshoot between widely-spaced keyframes) with
    virtual anchors at frame 0 and frame ``num_frames - 1`` holding the first /
    last keyframe's own correction -- this keeps the correction constant before
    the first and after the last keyframe rather than extrapolating it -- then
    added back onto the source. A keyframe's un-corrected DOF already has ~zero
    offset there, so no per-DOF "was this touched" bookkeeping is needed.

    ``root_pos_w``'s x, y are NOT part of that blend: they are
    ``solve_ik_for_clip``'s rigid terrain-patch placement (``source + offset_xy``),
    constant for every frame regardless of which are keyframes. Interpolating
    or fading this placement would move frames off their terrain patch.
    Root orientation is unchanged by IK and is not part of this function's
    output. The pipeline preserves the source quaternion in the output qpos.
    """
    frame_indices = np.asarray(sparse.frame_indices)
    joint_offset = sparse.joint_pos - source_joint_pos[frame_indices]
    root_z_offset = sparse.root_pos_w[:, 2] - source_root_pos_w[frame_indices, 2]

    # Hold the nearest keyframe correction outside the keyframe span.
    # Zero boundary corrections would pull elevated terrain motions back toward
    # their flat source and introduce abrupt height changes at the clip edges.
    # Do not duplicate an anchor where a real keyframe already fixes the value.
    # With no keyframes, zero anchors preserve the source plus XY placement.
    anchor_frames = [f for f in (0, num_frames - 1) if f not in frame_indices]
    anchor_joint_offset = np.zeros((len(anchor_frames), source_joint_pos.shape[1]))
    anchor_root_z_offset = np.zeros(len(anchor_frames))
    if len(frame_indices) > 0:
        order_by_frame = np.argsort(frame_indices)
        nearest = {0: order_by_frame[0], num_frames - 1: order_by_frame[-1]}
        for i, f in enumerate(anchor_frames):
            anchor_joint_offset[i] = joint_offset[nearest[f]]
            anchor_root_z_offset[i] = root_z_offset[nearest[f]]

    all_frames = np.concatenate([anchor_frames, frame_indices]).astype(np.int64)
    all_joint_offset = np.concatenate([anchor_joint_offset, joint_offset], axis=0)
    all_root_z_offset = np.concatenate([anchor_root_z_offset, root_z_offset])

    order = np.argsort(all_frames)
    all_frames = all_frames[order]
    all_joint_offset = all_joint_offset[order]
    all_root_z_offset = all_root_z_offset[order]

    query_frames = np.arange(num_frames)
    joint_offset_dense = PchipInterpolator(all_frames, all_joint_offset, axis=0)(query_frames)
    root_z_offset_dense = PchipInterpolator(all_frames, all_root_z_offset)(query_frames)

    dense_joint_pos = source_joint_pos + joint_offset_dense
    dense_root_pos_w = source_root_pos_w.copy()
    dense_root_pos_w[:, 0:2] = source_root_pos_w[:, 0:2] + offset_xy
    dense_root_pos_w[:, 2] = source_root_pos_w[:, 2] + root_z_offset_dense

    return IKClipResult(
        frame_indices=query_frames, joint_pos=dense_joint_pos, root_pos_w=dense_root_pos_w
    )
