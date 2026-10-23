"""End-effector contact/phase detection: flat motion -> per-frame per-EE stance mask.

Generalizes the enter/exit hysteresis rule used by ``feet_standing``
(``tasks/{tracking,perceptive}/mdp/commands.py``) from the two feet to an
arbitrary configurable set of end-effector bodies. Shared by (a) the offline
flat->terrain motion pipeline (see
``docs/superpowers/specs/2026-07-30-ee-contact-phase-detection-design.md``) via
``detect_ee_contact``, and (b) both tasks' live debug visualization
(``update_ee_contact_viz`` in ``commands.py``, see
``docs/superpowers/specs/2026-07-30-ee-contact-viz-design.md``) via
``ee_contact_hysteresis_step``. Pure/framework-agnostic: no terrain/mjlab
dependency, and ``ee_contact_hysteresis_step`` works identically for numpy
arrays or torch tensors (only comparison/boolean ops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

ArrayLike = TypeVar("ArrayLike")


@dataclass(frozen=True)
class EEContactCfg:
    z_enter: float = 0.18
    z_exit: float = 0.25
    vxy_enter: float = 0.2
    vxy_exit: float = 0.3
    vz_enter: float = 0.15
    vz_exit: float = 0.25


def ee_contact_hysteresis_step(
    z: ArrayLike,
    vxy: ArrayLike,
    vz: ArrayLike,
    root_vxy: ArrayLike,
    prev_state: ArrayLike,
    cfg: EEContactCfg,
) -> ArrayLike:
    """One hysteresis transition: ``new_state = (prev_state & ~exit) | enter``.

    Works for both ``numpy.ndarray`` and ``torch.Tensor`` inputs (only uses
    comparison and boolean operators, no library-specific calls) -- this is the
    one core shared between the offline whole-clip scan (``detect_ee_contact``)
    and the live per-step persistent-state update in ``commands.py``.
    """
    enter = (z < cfg.z_enter) & (vxy < cfg.vxy_enter * root_vxy) & (vz < cfg.vz_enter * root_vxy)
    exit_ = (z > cfg.z_exit) | (vxy > cfg.vxy_exit * root_vxy) | (vz > cfg.vz_exit * root_vxy)
    return (prev_state & ~exit_) | enter


def detect_ee_contact(
    body_pos_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_names: list[str],
    ee_names: list[str],
    cfg: EEContactCfg | dict[str, EEContactCfg] = EEContactCfg(),
    ground_z: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame, per-EE stance/contact mask via enter/exit height+velocity hysteresis.

    Args:
      body_pos_w: (T, B, 3) world-space body positions.
      body_lin_vel_w: (T, B, 3) world-space body linear velocities.
      body_names: length-B index into ``body_pos_w``/``body_lin_vel_w``.
        ``body_names[0]`` is treated as the root (pelvis) for velocity normalization.
      ee_names: candidate end-effector body names (subset of ``body_names``).
      cfg: shared ``EEContactCfg``, or a per-EE dict keyed exactly by ``ee_names``.
      ground_z: (T, len(ee_names)) terrain height directly under each EE, at
        that EE's own (x, y) each frame -- the height check is relative to
        this, not absolute world z. None (default) means flat ground at z=0,
        matching a source clip that hasn't been placed on terrain yet. Each EE
        gets its own column because two EEs at the same frame can have
        different (x, y) and therefore different terrain heights underneath
        them (e.g. one foot on a step, the other still on the floor).

    Returns:
      bool (T, len(ee_names)) contact mask, column order matching ``ee_names``.
    """
    unknown = [name for name in ee_names if name not in body_names]
    if unknown:
        raise ValueError(f"ee_names not found in body_names: {unknown}")

    if isinstance(cfg, dict):
        if set(cfg.keys()) != set(ee_names):
            raise ValueError(
                f"cfg keys {sorted(cfg.keys())} must match ee_names {sorted(ee_names)} exactly"
            )
        cfg_per_ee = [cfg[name] for name in ee_names]
    else:
        cfg_per_ee = [cfg] * len(ee_names)

    root_vxy = np.linalg.norm(body_lin_vel_w[:, 0, :2], axis=-1)
    root_vxy = np.clip(root_vxy, 1.0, None)  # (T,)

    T = body_pos_w.shape[0]
    if ground_z is None:
        ground_z = np.zeros((T, len(ee_names)))

    mask = np.zeros((T, len(ee_names)), dtype=bool)
    for col, (name, ee_cfg) in enumerate(zip(ee_names, cfg_per_ee)):
        idx = body_names.index(name)
        z = body_pos_w[:, idx, 2] - ground_z[:, col]
        vxy = np.linalg.norm(body_lin_vel_w[:, idx, :2], axis=-1)
        vz = np.abs(body_lin_vel_w[:, idx, 2])

        state = np.zeros(T, dtype=bool)
        prev = np.array(False)
        for t in range(T):
            prev = ee_contact_hysteresis_step(z[t], vxy[t], vz[t], root_vxy[t], prev, ee_cfg)
            state[t] = bool(prev)
        mask[:, col] = state

    return mask
