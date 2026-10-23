"""A scripted, model-free solver used by `validate` and the tests: approach behind the
box on the goal line, drop to push height, push in short strokes, re-observing between
strokes and correcting the line. Proves tasks are solvable through the public tools."""

from __future__ import annotations

import math

from sdk_arm_push import sim

PUSH_Z = 0.026
CLEAR_Z = 0.09
BACKOFF = 0.045  # pusher centre behind box centre (box half 0.02 + pusher 0.015 + gap)


def solve(s: dict, budget: int = 60) -> bool:
    for _ in range(12):
        o = sim.observe(s)
        bx, by, _ = o["box.xyz"]
        gx, gy = o["goal.xy"]
        dist = math.hypot(gx - bx, gy - by)
        if dist <= sim.SUCCESS_TOL * 0.6:
            break
        ux, uy = (gx - bx) / dist, (gy - by) / dist
        px, py = bx - BACKOFF * ux, by - BACKOFF * uy
        sim.pose(s, px, py, CLEAR_Z, budget)
        sim.pose(s, px, py, PUSH_Z, budget)
        stroke = min(dist, 0.06)
        sim.pose(s, px + stroke * ux, py + stroke * uy, PUSH_Z, budget)
        sim.pose(s, px + stroke * ux - 0.02 * ux, py + stroke * uy - 0.02 * uy, CLEAR_Z, budget)
    sim.home(s, budget)
    return sim.success(s)
