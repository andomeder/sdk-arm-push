"""Pick the GL backend before anything imports mujoco, then run sdk_arm_push.render."""

import importlib.util
import os
import sys
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    path = Path(__file__).resolve().parent.parent / "sdk_arm_push" / "device.py"
    spec = importlib.util.spec_from_file_location("_device", path)
    device = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device)  # device.py imports no mujoco
    os.environ["MUJOCO_GL"] = device.pick_gl_backend()
os.environ.pop("PYOPENGL_PLATFORM", None)

from sdk_arm_push.render import main  # noqa: E402

sys.exit(main())
