"""The arm tool server. verifiers launches it as `python -m sdk_arm_push.tools`."""

from __future__ import annotations

from typing import Any

from pydantic import Field

import verifiers.v1 as vf
from sdk_arm_push import sim


class ArmState(vf.State):
    sim: dict[str, Any] = Field(default_factory=dict)


class ArmToolsetConfig(vf.ToolsetConfig):
    budget: int = 40


class ArmToolset(vf.Toolset[ArmToolsetConfig, ArmState]):
    TOOL_PREFIX = "arm"

    @vf.tool
    def observe(self) -> dict:
        """Read joint positions, end-effector xyz, box xyz and goal xy."""
        return sim.observe(self.state.sim)

    @vf.tool
    def pose(self, x: float, y: float, z: float) -> dict:
        """Move the pusher to (x, y, z) in metres. Returns ok, or blocked with a reason."""
        return sim.pose(self.state.sim, x, y, z, self.config.budget)

    @vf.tool
    def jog(self, joint: str, delta: float) -> dict:
        """Move one joint (base_yaw, shoulder, elbow, wrist) by delta radians."""
        return sim.jog(self.state.sim, joint, delta, self.config.budget)

    @vf.tool
    def home(self) -> dict:
        """Return the arm to its home pose."""
        return sim.home(self.state.sim, self.config.budget)

    @vf.tool
    def estop(self) -> dict:
        """Latch the emergency stop. Motion commands are refused until reset_latch."""
        return sim.estop(self.state.sim)

    @vf.tool
    def reset_latch(self) -> dict:
        """Clear the emergency stop latch."""
        return sim.reset_latch(self.state.sim)


if __name__ == "__main__":
    ArmToolset.run()
