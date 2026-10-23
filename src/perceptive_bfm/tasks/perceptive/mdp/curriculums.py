"""Linearly ramp regularization reward weights as training progresses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["modify_reward_scale"]


def _scale_factor(step: int, total: int, factor_range: tuple[float, float]) -> float:
    """Linear ramp factor for the current training step.

    ``progress = clamp(step / total, 0, 1)``; the factor linearly interpolates
    ``factor_range`` over that progress. A non-positive ``total`` (unset/invalid) is
    treated as "fully trained" and returns ``factor_range[1]``.
    """
    f0, f1 = factor_range
    if total <= 0:
        return f1
    progress = min(max(step / total, 0.0), 1.0)
    return f0 + progress * (f1 - f0)


def modify_reward_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    num_steps_fallback: int,
    task_reward_prefix: str = "motion",
    factor_range: tuple[float, float] = (0.5, 1.0),
) -> float:
    """Ramp the regularization reward terms' weights with training progress.

    The regularization terms are selected by convention: every reward term whose name
    does **not** start with ``task_reward_prefix`` (task/tracking rewards are named
    ``motion_*``).

    Total training steps come from ``env.curriculum_total_steps`` when the runner has
    injected the runtime value; otherwise ``num_steps_fallback``.

    Returns the current factor, logged by the curriculum manager as
    ``Curriculum/reward_scale``.
    """
    del env_ids  # Global factor.
    store: dict[str, float] = env.__dict__.setdefault("_reward_curriculum_base_weights", {})
    total = getattr(env, "curriculum_total_steps", None) or num_steps_fallback
    factor = _scale_factor(env.common_step_counter, total, factor_range)
    for name in env.reward_manager.active_terms:
        if name.startswith(task_reward_prefix):
            continue  # Task/tracking reward — not part of the curriculum.
        term_cfg = env.reward_manager.get_term_cfg(name)
        if name not in store:
            store[name] = term_cfg.weight
        term_cfg.weight = store[name] * factor
    return factor
