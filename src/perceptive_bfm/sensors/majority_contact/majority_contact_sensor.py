"""Contact sensor with substep-majority voting.

A body is in contact when found > 0 in at least half the physics substeps of a control step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.sensor import ContactSensor, ContactSensorCfg


class MajorityContactSensor(ContactSensor):
    """A ``ContactSensor`` that also exposes a substep-majority ``current_contact``."""

    cfg: MajorityContactSensorCfg

    def __init__(self, cfg: MajorityContactSensorCfg):
        super().__init__(cfg)
        self._window = max(int(cfg.substeps), 1)
        self._found_ring: torch.Tensor | None = None  # [B, N, window]
        self._ring_index = 0
        self._current_contact: torch.Tensor | None = None  # [B, N] bool

    @property
    def current_contact(self) -> torch.Tensor:
        """[B, N] bool: substep-majority contact for the current control step.

        ``N`` and the column order match the sensor's ``found`` field (bodies in
        body-id order).
        """
        assert (
            self._current_contact is not None
        ), f"Sensor '{self.cfg.name}' has not been updated yet."
        return self._current_contact

    def update(self, dt: float) -> None:
        super().update(dt)  # refreshes data and invalidates the cache (call first).
        found = self.data.found
        assert (
            found is not None
        ), f"Sensor '{self.cfg.name}' must include the 'found' field for majority voting."
        contact = found > 0
        if self._found_ring is None:
            self._found_ring = torch.zeros(
                *contact.shape, self._window, dtype=torch.bool, device=contact.device
            )
            self._current_contact = torch.zeros_like(contact)
        self._found_ring[:, :, self._ring_index % self._window] = contact
        self._ring_index += 1
        votes = self._found_ring.sum(dim=-1)
        # A body is in contact when at least half of the substeps report contact.
        self._current_contact = votes >= ((self._window + 1) // 2)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)  # call first to invalidate the cache.
        if self._found_ring is not None:
            idx = slice(None) if env_ids is None else env_ids
            self._found_ring[idx] = False
            self._current_contact[idx] = False


@dataclass(kw_only=True)
class MajorityContactSensorCfg(ContactSensorCfg):
    class_type: type = MajorityContactSensor

    substeps: int = 1
    """Physics substeps per control step (set to the env ``decimation``). This is the
    majority-voting window; it should equal ``decimation`` so the window read after the
    decimation loop covers exactly one control step."""

    def build(self) -> MajorityContactSensor:
        return MajorityContactSensor(self)
