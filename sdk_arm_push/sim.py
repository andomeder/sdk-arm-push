"""MuJoCo core for sdk-arm-push: a 4-joint arm pushes a box to a goal through an
SDK-style command interface (observe / pose / jog / home / estop / reset_latch).

The command names and blocked reasons mirror the Nori A3 SDK (`pose` with on-robot IK,
`no_ik_solution`, `limit:<joint>`, `collision`, estop latch), so a model that learns this
interface is learning a real robot API. The arm here is a self-authored MJCF so the
package is Apache-2.0 clean; the A3 model (CC BY-NC-SA) would be an optional asset.

Everything is deterministic: the full sim state is a plain dict, rebuilt into MjData on
every call, so it can live in a verifiers `State` and survive tool-call serialization.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

import mujoco
import numpy as np

JOINTS = ("base_yaw", "shoulder", "elbow", "wrist")
ARM_GEOMS = ("upper", "fore", "hand", "pusher")
HAZARD_GEOMS = ("floor",)  # the arm touching the floor counts as a collision
SUCCESS_TOL = 0.03  # m, box centre to goal centre in xy
MAX_SPEED = 1.5  # rad/s joint speed during interpolated moves
DT = 0.002

MJCF = """
<mujoco model="sdk_arm_push">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="2" armature="0.01"/>
    <geom friction="0.8 0.005 0.0001" contype="1" conaffinity="3"/>
    <default class="arm"><geom contype="2" conaffinity="1"/></default>
    <position kp="400" forcerange="-60 60"/>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".85 .85 .85" rgb2=".75 .75 .75"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.5" dir="-0.2 0.2 -1" diffuse=".9 .9 .9"/>
    <camera name="overview" pos="0.95 -0.75 0.75" xyaxes="0.62 0.78 0 -0.38 0.3 0.87"/>
    <body name="goal_marker" mocap="true" pos="0.3 0.1 0">
      <geom name="goal_marker" type="cylinder" size="0.03 0.0008" rgba=".1 .7 .2 .6"
            contype="0" conaffinity="0"/>
    </body>
    <geom name="floor" type="plane" size="1 1 0.01" material="grid"/>
    <body name="pedestal" pos="0 0 0">
      <geom name="pedestal" type="cylinder" size="0.05 0.06" pos="0 0 0.06" rgba=".3 .3 .3 1"
            contype="0" conaffinity="0"/>
      <body name="yaw_link" pos="0 0 0.12">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-2.6 2.6"/>
        <geom name="turret" type="cylinder" size="0.03 0.015" mass="0.3" contype="0" conaffinity="0"/>
        <body name="upper_link" pos="0 0 0.02">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.8 1.0"/>
          <geom class="arm" name="upper" type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.6"/>
          <body name="fore_link" pos="0.25 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.6 2.6"/>
            <geom class="arm" name="fore" type="capsule" fromto="0 0 0 0.22 0 0" size="0.018" mass="0.4"/>
            <body name="hand_link" pos="0.22 0 0">
              <joint name="wrist" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom class="arm" name="hand" type="capsule" fromto="0 0 0 0.06 0 0" size="0.012" mass="0.1"/>
              <geom class="arm" name="pusher" type="sphere" pos="0.075 0 0" size="0.015" mass="0.05"/>
              <site name="ee" pos="0.075 0 0" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="box" pos="0.3 0 0.02">
      <freejoint name="box"/>
      <geom name="box" type="box" size="0.02 0.02 0.02" mass="0.08" rgba=".9 .4 .1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="base_yaw" joint="base_yaw" ctrlrange="-2.6 2.6"/>
    <position name="shoulder" joint="shoulder" ctrlrange="-1.8 1.0"/>
    <position name="elbow" joint="elbow" ctrlrange="-2.6 2.6"/>
    <position name="wrist" joint="wrist" ctrlrange="-2.0 2.0"/>
  </actuator>
</mujoco>
"""

HOME = (0.0, -1.0, 2.0, 0.6)  # pusher held above the near edge of the workspace


@lru_cache(maxsize=1)
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MJCF)


def _ids(m: mujoco.MjModel):
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    return {
        "arm": {gid(n) for n in ARM_GEOMS},
        "hazard": {gid(n) for n in HAZARD_GEOMS},
        "box_body": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box"),
        "ee": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee"),
        "jnt_qadr": [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS],
        "box_qadr": m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box")],
    }


# ---------------------------------------------------------------- scenes

def make_scene(seed: int) -> dict:
    """Seeded box and goal: box 0.25-0.40 m out within +-60 deg, goal 8-15 cm away and
    still inside the arm's comfortable workspace."""
    rng = random.Random(seed)
    while True:
        r, a = rng.uniform(0.25, 0.40), math.radians(rng.uniform(-60, 60))
        bx, by = r * math.cos(a), r * math.sin(a)
        d, g = rng.uniform(0.08, 0.15), rng.uniform(0, 2 * math.pi)
        gx, gy = bx + d * math.cos(g), by + d * math.sin(g)
        if 0.22 <= math.hypot(gx, gy) <= 0.42 and abs(math.atan2(gy, gx)) <= math.radians(70):
            return {"seed": seed, "box0": [bx, by], "goal": [gx, gy]}


def initial_state(scene: dict) -> dict:
    m = model()
    d = mujoco.MjData(m)
    ids = _ids(m)
    for q, adr in zip(HOME, ids["jnt_qadr"]):
        d.qpos[adr] = q
    d.ctrl[:] = HOME
    b = ids["box_qadr"]
    d.qpos[b : b + 3] = [scene["box0"][0], scene["box0"][1], 0.02]
    d.qpos[b + 3 : b + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    s = {
        "scene": scene,
        "qpos": d.qpos.tolist(),
        "qvel": d.qvel.tolist(),
        "ctrl": d.ctrl.tolist(),
        "time": 0.0,
        "latched": False,
        "commands": 0,
        "blocked": 0,
        "collisions": 0,
        "log": [],
    }
    _settle(s, 0.3)
    return s


# ---------------------------------------------------------------- state <-> MjData

def _load(s: dict) -> mujoco.MjData:
    m = model()
    d = mujoco.MjData(m)
    d.qpos[:] = s["qpos"]
    d.qvel[:] = s["qvel"]
    d.ctrl[:] = s["ctrl"]
    d.time = s["time"]
    mujoco.mj_forward(m, d)
    return d


def _store(s: dict, d: mujoco.MjData) -> None:
    s["qpos"], s["qvel"], s["ctrl"], s["time"] = (
        d.qpos.tolist(), d.qvel.tolist(), d.ctrl.tolist(), float(d.time))


def _hazard_contact(m, d, ids) -> bool:
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {c.geom1, c.geom2}
        if pair & ids["arm"] and pair & ids["hazard"]:
            return True
    return False


_RECORDER = None  # (callback(model, data, state), every_n_steps) while rendering


def set_recorder(callback=None, every: int = 17) -> None:
    """Call `callback(model, data, state)` every `every` physics steps (render.py uses it)."""
    global _RECORDER
    _RECORDER = (callback, every) if callback else None


def _run(s: dict, targets: np.ndarray | None, seconds: float) -> dict:
    """Interpolate ctrl to `targets` at MAX_SPEED, then settle. Counts hazard contacts."""
    m, d = model(), _load(s)
    ids = _ids(m)
    start = d.ctrl.copy()
    if targets is None:
        targets = start
    span = float(np.max(np.abs(targets - start))) if len(start) else 0.0
    move_steps = max(1, int(span / MAX_SPEED / DT))
    settle_steps = int(seconds / DT)
    touched = False
    for k in range(move_steps + settle_steps):
        a = min(1.0, (k + 1) / move_steps)
        d.ctrl[:] = start + a * (targets - start)
        mujoco.mj_step(m, d)
        if _RECORDER and k % _RECORDER[1] == 0:
            _RECORDER[0](m, d, s)
        if not touched and _hazard_contact(m, d, ids):
            touched = True
    _store(s, d)
    if touched:
        s["collisions"] += 1
    return {"collision_during_move": touched}


def _settle(s: dict, seconds: float) -> None:
    _run(s, None, seconds)


# ---------------------------------------------------------------- IK

def _ik(s: dict, xyz) -> tuple[np.ndarray | None, str | None]:
    """Position-only damped least squares from the current joint targets. Returns
    (joint targets, None) or (None, blocked_reason)."""
    m = model()
    d = _load(s)
    ids = _ids(m)
    adr = ids["jnt_qadr"]
    lo, hi = m.jnt_range[: len(JOINTS), 0], m.jnt_range[: len(JOINTS), 1]
    q = np.array(s["ctrl"], dtype=float)
    goal = np.asarray(xyz, dtype=float)
    jacp = np.zeros((3, m.nv))
    hit_limit: str | None = None
    for _ in range(300):
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = goal - d.site_xpos[ids["ee"]]
        if np.linalg.norm(err) < 0.004:
            break
        mujoco.mj_jacSite(m, d, jacp, None, ids["ee"])
        J = jacp[:, [m.jnt_dofadr[i] for i in range(len(JOINTS))]]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        q = q + np.clip(dq, -0.2, 0.2)
        clipped = np.clip(q, lo, hi)
        if np.any(clipped != q):
            hit_limit = JOINTS[int(np.argmax(np.abs(clipped - q)))]
        q = clipped
    else:
        return None, (f"limit:{hit_limit}" if hit_limit else "no_ik_solution")
    # collision pre-check at the solution: geometry overlap with the floor
    d.qpos[adr] = q
    mujoco.mj_forward(m, d)
    if _hazard_contact(m, d, ids):
        return None, "collision"
    return q, None


# ---------------------------------------------------------------- SDK-style commands

def observe(s: dict, privileged: bool = True) -> dict:
    m, d = model(), _load(s)
    ids = _ids(m)
    out = {f"{j}.pos": round(float(d.qpos[a]), 4) for j, a in zip(JOINTS, ids["jnt_qadr"])}
    out["ee.xyz"] = [round(float(v), 4) for v in d.site_xpos[ids["ee"]]]
    out["estop_latched"] = s["latched"]
    out["commands_used"] = s["commands"]
    if privileged:
        out["box.xyz"] = [round(float(v), 4) for v in d.xpos[ids["box_body"]]]
        out["goal.xy"] = [round(v, 4) for v in s["scene"]["goal"]]
    return out


def _guard(s: dict, budget: int) -> dict | None:
    if s["latched"]:
        return {"ok": False, "blocked": "estop_latched"}
    if s["commands"] >= budget:
        return {"ok": False, "blocked": "budget_exhausted"}
    s["commands"] += 1
    return None


def pose(s: dict, x: float, y: float, z: float, budget: int = 60) -> dict:
    if (g := _guard(s, budget)) is not None:
        return g
    q, reason = _ik(s, (x, y, z))
    if reason:
        s["blocked"] += 1
        s["log"].append(["pose", [x, y, z], reason])
        return {"ok": False, "blocked": reason}
    r = _run(s, q, 0.25)
    s["log"].append(["pose", [x, y, z], "ok"])
    return {"ok": True, **r}


def jog(s: dict, joint: str, delta: float, budget: int = 60) -> dict:
    if joint not in JOINTS:
        return {"ok": False, "blocked": "bad_argument"}
    if (g := _guard(s, budget)) is not None:
        return g
    m = model()
    i = JOINTS.index(joint)
    q = np.array(s["ctrl"], dtype=float)
    q[i] += float(np.clip(delta, -0.5, 0.5))
    lo, hi = m.jnt_range[i]
    if not lo <= q[i] <= hi:
        s["blocked"] += 1
        return {"ok": False, "blocked": f"limit:{joint}"}
    r = _run(s, q, 0.25)
    return {"ok": True, **r}


def home(s: dict, budget: int = 60) -> dict:
    if (g := _guard(s, budget)) is not None:
        return g
    return {"ok": True, **_run(s, np.array(HOME), 0.3)}


def estop(s: dict) -> dict:
    s["latched"] = True
    return {"ok": True, "estop_confirmed": True}


def reset_latch(s: dict) -> dict:
    s["latched"] = False
    return {"ok": True}


# ---------------------------------------------------------------- grading

def box_goal_distance(s: dict) -> float:
    m, d = model(), _load(s)
    b = d.xpos[_ids(m)["box_body"]]
    g = s["scene"]["goal"]
    return float(math.hypot(b[0] - g[0], b[1] - g[1]))


def success(s: dict) -> bool:
    """The simulator's own check: box within SUCCESS_TOL of the goal and still on the table."""
    m, d = model(), _load(s)
    bz = d.xpos[_ids(m)["box_body"]][2]
    return bool(box_goal_distance(s) <= SUCCESS_TOL and 0.0 < float(bz) < 0.05)
