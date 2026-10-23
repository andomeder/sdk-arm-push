"""Render the scripted solver on a few seeds to an MP4.

    python scripts/render_demo.py --seeds 1001 1003 1007 --out demo.mp4

MUJOCO_GL is read when mujoco is imported, so scripts/render_demo.py picks the backend
first: $MUJOCO_GL if set, else egl if a test render works, else osmesa (CPU).
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1003, 1007])
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    # MUJOCO_GL must be set before mujoco is imported; scripts/render_demo.py does that.
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}", file=sys.stderr)

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from sdk_arm_push import reference, sim

    m = sim.model()
    renderer = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
    goal_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")]
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
    label = {"text": ""}
    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = None

    def frame(model, data, state):
        gx, gy = state["scene"]["goal"]
        data.mocap_pos[goal_mocap] = [gx, gy, 0.0008]
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((16, 14), label["text"], fill=(235, 235, 235), font=font)
        writer.append_data(np.asarray(img))

    sim.set_recorder(frame, every=max(1, int(1 / (a.fps * sim.DT))))
    results = []
    for seed in a.seeds:
        s = sim.initial_state(sim.make_scene(seed))
        label["text"] = f"seed {seed}: scripted solver via observe/pose tool calls"
        ok = reference.solve(s)
        label["text"] = (
            f"seed {seed}: {'SUCCESS' if ok else 'FAIL'}  box-goal {sim.box_goal_distance(s)*100:.1f} cm  "
            f"commands {s['commands']}  collisions {s['collisions']}"
        )
        sim._settle(s, 1.0)  # hold the result on screen
        results.append((seed, ok, s["commands"], s["collisions"]))
    sim.set_recorder(None)
    writer.close()
    for r in results:
        print("seed %d success=%s commands=%d collisions=%d" % r)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
