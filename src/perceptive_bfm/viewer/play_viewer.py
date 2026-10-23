"""mjlab ViserPlayViewer + the raycaster depth-image panel.

mjlab's viser viewer builds camera feeds only for ``CameraSensor`` sensors, which
leaves the perceptive task's raycaster depth camera with no image panel (see
viser_depth_panel). This subclass adds one, reusing mjlab's own update gating so
the feed refreshes continuously while running and on demand while paused.

Safe to use for every task: with no raycaster camera in the scene the panel
builds no widgets and this behaves exactly like ``ViserPlayViewer``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
from mjlab.sim import Simulation
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction

from perceptive_bfm.viewer.viser_depth_panel import ViserRaycasterDepthPanel

_REWARD_BAR_MAX_TERMS = 32


class MyViserPlayViewer(ViserPlayViewer):
    """``ViserPlayViewer`` with depth-image panels for raycaster depth cameras."""

    def setup(self) -> None:
        super().setup()
        self._configure_reward_bar_panel()
        self._depth_panel = ViserRaycasterDepthPanel(self._server, self.env.unwrapped)
        self._depth_panel.setup()

    def _configure_reward_bar_panel(self) -> None:
        """Expand mjlab's reward bar panel without moving it out of its tab."""
        term_overlays = getattr(self, "_term_overlays", None)
        panel = getattr(term_overlays, "reward_bar_panel", None)
        if panel is None:
            return

        reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
        if reward_manager is None:
            return
        term_names = [
            name for name, _ in reward_manager.get_active_iterable_terms(self._scene.env_idx)
        ][:_REWARD_BAR_MAX_TERMS]

        # RewardBarPanel creates these buffers from its default 20-term slice.
        # Rebuild them in place so the existing Rewards tab and HTML handle stay intact.
        panel._term_names = term_names
        panel._histories = {name: deque(maxlen=panel._window_steps) for name in term_names}
        panel._render()

    def _update_camera_feeds(self, sim: Simulation, has_pending_updates: bool) -> None:
        super()._update_camera_feeds(sim, has_pending_updates)
        if self._should_update_cameras(self._is_paused, has_pending_updates):
            self._depth_panel.update(self._scene.env_idx)

    def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
        if not (isinstance(payload, dict) and payload.get("type") == "terrain_switch"):
            return super()._handle_custom_action(action, payload)

        env = self.env.unwrapped
        if payload["all_envs"]:
            env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        else:
            env_ids = torch.tensor([self._scene.env_idx], dtype=torch.int64, device=env.device)

        command = env.command_manager.get_term(payload["command"])
        with self._sim_lock:
            command.set_fixed_terrain(
                env_ids,
                terrain_type=payload["terrain_type"],
                terrain_level=payload["terrain_level"],
            )
            env.reset(env_ids=env_ids)
        return True

    def close(self) -> None:
        self._depth_panel.cleanup()
        super().close()
