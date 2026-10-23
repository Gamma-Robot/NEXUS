"""Robot structure required by the motion adaptation pipeline."""

from dataclasses import dataclass

from terrain_motion_adaptation.optimization.keyframe_ik import ChainSpec


@dataclass(frozen=True)
class RobotConfig:
    root_body: str
    end_effectors: tuple[str, ...]
    chain_specs: dict[str, ChainSpec]

    @property
    def contact_bodies(self) -> tuple[str, ...]:
        return (self.root_body,) + self.end_effectors
