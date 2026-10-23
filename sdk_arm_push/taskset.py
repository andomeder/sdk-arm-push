"""sdk-arm-push: verifiers v1 taskset.

The model drives a simulated arm through SDK-style tool calls (observe, pose, jog, home,
estop, reset_latch) and must push a box onto a goal. MuJoCo's own state decides the
reward; nothing the model says is graded. Sim state lives in the rollout State, so each
tool call rebuilds MjData from it and the episode is reproducible from the seed.
"""

from __future__ import annotations

import verifiers.v1 as vf
from sdk_arm_push import reference, sim
from sdk_arm_push.tools import ArmState, ArmToolset, ArmToolsetConfig

PROMPT = """You control a 4-joint robot arm on a table through tool calls. Joints: base_yaw,
shoulder, elbow, wrist. Units are metres and radians; the arm's base is at (0, 0) and the
table surface is z = 0. The end effector is a small round pusher (radius 0.015 m); there
is no gripper.

Goal: push the orange box (a 0.04 m cube) so its centre ends within {tol_cm:.0f} cm of the
goal point in x-y, and leave it on the table. You have {budget} motion commands.

Tools:
- arm_observe(): joint positions, end-effector xyz, box xyz and goal xy.
- arm_pose(x, y, z): move the pusher to a point (on-robot IK). May return
  blocked = no_ik_solution | limit:<joint> | collision, in which case nothing moved.
- arm_jog(joint, delta): move one joint by delta radians (|delta| <= 0.5).
- arm_home(): return to the home pose.
- arm_estop() / arm_reset_latch(): emergency stop latch.

Touching the table with the arm counts as a collision and costs reward. When you are done,
reply with a short summary."""


class ArmTaskConfig(vf.TaskConfig):
    tools: ArmToolsetConfig = ArmToolsetConfig()


class ArmData(vf.TaskData):
    seed: int


class ArmTask(vf.Task[ArmData, ArmState, ArmTaskConfig]):
    @classmethod
    def toolsets(cls, config: ArmTaskConfig) -> list[vf.Toolset]:
        return [ArmToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.sim = sim.initial_state(sim.make_scene(self.data.seed))

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        s = trace.state.sim
        trace.info["arm"] = {
            "seed": self.data.seed,
            "final_distance_m": round(sim.box_goal_distance(s), 4),
            "commands": s["commands"],
            "blocked": s["blocked"],
            "collisions": s["collisions"],
            "log": s["log"],
        }

    @vf.reward(weight=1.0)
    async def box_on_goal(self, trace: vf.Trace) -> float:
        return float(sim.success(trace.state.sim))

    @vf.reward(weight=0.25)
    async def clean_success(self, trace: vf.Trace) -> float:
        s = trace.state.sim
        return float(sim.success(s) and s["collisions"] == 0)

    @vf.metric
    async def collisions(self, trace: vf.Trace) -> float:
        return float(trace.state.sim["collisions"])

    @vf.metric
    async def blocked_commands(self, trace: vf.Trace) -> float:
        return float(trace.state.sim["blocked"])

    @vf.metric
    async def commands_used(self, trace: vf.Trace) -> float:
        return float(trace.state.sim["commands"])

    @vf.metric
    async def final_distance(self, trace: vf.Trace) -> float:
        return sim.box_goal_distance(trace.state.sim)

    async def validate(self, runtime: vf.Runtime) -> bool | None:
        s = sim.initial_state(sim.make_scene(self.data.seed))
        return bool(reference.solve(s, self.config.tools.budget))


class ArmConfig(vf.TasksetConfig):
    num_tasks: int = 64
    seed_offset: int = 0  # use 1000 for the held-out range
    task: ArmTaskConfig = ArmTaskConfig()


class ArmTaskset(vf.Taskset[ArmTask, ArmConfig]):
    def load(self) -> list[ArmTask]:
        budget = self.config.task.tools.budget
        prompt = PROMPT.format(tol_cm=sim.SUCCESS_TOL * 100, budget=budget)
        return [
            ArmTask(
                ArmData(id=f"push-{seed}", name=f"push-{seed}", seed=seed, prompt=prompt),
                self.config.task,
            )
            for seed in range(self.config.seed_offset + 1, self.config.seed_offset + 1 + self.config.num_tasks)
        ]
