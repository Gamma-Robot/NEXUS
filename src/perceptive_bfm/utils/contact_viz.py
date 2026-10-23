"""Per-frame (position, color) sphere specs for visualizing ``detect_ee_contact``'s
output (or the live ``update_ee_contact_viz`` state) during motion replay/playback.
Pure function -- no mjlab dependency, so callers (``scripts/data_process/
replay_npz.py``'s ``--viz_ee_contact`` offline path, and ``tasks/{tracking,
perceptive}/mdp/commands.py``'s live ``_debug_vis_impl``) each own how the spec is
actually drawn (via ``DebugVisualizer.add_sphere``) and where the frame's data
comes from -- an offline whole-clip array indexed by frame, or a live env's
current-step tensor.
"""

from __future__ import annotations

import numpy as np

Color = tuple[float, float, float, float]

CONTACT_COLOR: Color = (0.0, 1.0, 0.0, 1.0)
SWING_COLOR: Color = (1.0, 0.0, 0.0, 1.0)

# Shared default sphere radius (meters) for both call sites above.
EE_SPHERE_RADIUS = 0.05


def ee_sphere_specs(
    body_pos_w_frame: np.ndarray,
    contact_mask_frame: np.ndarray,
    ee_body_indexes: list[int],
    offset_xy: np.ndarray | None = None,
    contact_color: Color = CONTACT_COLOR,
    swing_color: Color = SWING_COLOR,
) -> list[tuple[np.ndarray, Color]]:
    """One (world position, color) pair per EE, for an already-selected frame/env.

    Args:
      body_pos_w_frame: (B, 3) world-space body positions for the frame/env to draw.
      contact_mask_frame: (n_ee,) bool, in the same order as ``ee_body_indexes``.
      ee_body_indexes: index into ``body_pos_w_frame``'s body axis, one per EE, in
        the same order as ``contact_mask_frame``.
      offset_xy: (2,) world xy offset to add (e.g. a terrain patch placement);
        None applies no offset. Only x/y are shifted, z is left as-is.
      contact_color: color when ``contact_mask_frame`` is True for that EE.
      swing_color: color when ``contact_mask_frame`` is False.
    """
    specs: list[tuple[np.ndarray, Color]] = []
    for col, body_idx in enumerate(ee_body_indexes):
        pos = body_pos_w_frame[body_idx].copy()
        if offset_xy is not None:
            pos[:2] += offset_xy
        color = contact_color if contact_mask_frame[col] else swing_color
        specs.append((pos, color))
    return specs
