"""Hardware probing. It follows the pattern in andomeder/act-mujoco-manipulation
(`src/utils/gpu_utils.py`: CUDA, then XPU, then CPU, with an lspci fallback) and
decision-robot-control (MUJOCO_GL=egl by default, llama.cpp SYCL build on Arc, CPU fallback).

What runs where in this package:
- MuJoCo physics: CPU always (MuJoCo has no GPU physics; MJX is a separate JAX port).
- Offscreen rendering (render.py): OpenGL through EGL on the GPU's Mesa driver (iris/xe
  on an Arc), falling back to OSMesa (software, CPU) when no EGL device is usable.
- Model inference: not in this package. The model under test is served by an
  OpenAI-compatible server (llama.cpp SYCL/Vulkan on Arc, vLLM XPU, Ollama, or an API).
  `torch_device()` is here for later tiers that run learned policies in-process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys


def detect_gpu_vendor() -> str:
    """nvidia | intel | amd | cpu. Uses torch if installed, else lspci."""
    try:
        import torch  # optional: `uv sync --extra xpu`

        if torch.cuda.is_available():
            return "nvidia"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "intel"
    except ImportError:
        pass
    try:
        out = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    vga = "\n".join(l for l in out.splitlines() if "VGA" in l or "Display" in l or "3D" in l)
    if "NVIDIA" in vga:
        return "nvidia"
    if "Intel" in vga and ("Arc" in vga or "Battlemage" in vga or "Xe" in vga or "DG2" in vga):
        return "intel"
    if "AMD" in vga and "Radeon" in vga:
        return "amd"
    return "cpu"


def torch_device(prefer: str | None = None):
    """torch.device: cuda, then xpu, then cpu. `prefer` or $TORCH_DEVICE overrides."""
    import torch

    want = (prefer or os.environ.get("TORCH_DEVICE", "auto")).lower()
    if want in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if want in ("xpu", "auto") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def level_zero_available() -> bool:
    """True when libze_loader loads (the same library mojo-intel-gpu binds to)."""
    name = ctypes.util.find_library("ze_loader") or "libze_loader.so.1"
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return False
    try:
        return lib.zeInit(0) == 0
    except Exception:
        return False


def sycl_devices() -> list[str]:
    """Lines from `sycl-ls` (oneAPI), or [] when oneAPI isn't on PATH."""
    if not shutil.which("sycl-ls"):
        return []
    try:
        out = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def gl_backend_ok(backend: str) -> bool:
    """Try a tiny offscreen render with MUJOCO_GL=backend in a child process."""
    code = (
        "import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>"
        "<geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>');"
        "r=mujoco.Renderer(m,32,32);d=mujoco.MjData(m);r.update_scene(d);r.render();print('ok')"
    )
    # mujoco sets PYOPENGL_PLATFORM from MUJOCO_GL on import; a stale inherited value
    # (e.g. egl from a parent that already imported mujoco) silently disables Renderer.
    env = dict(os.environ, MUJOCO_GL=backend)
    env.pop("PYOPENGL_PLATFORM", None)
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "ok" in r.stdout


def pick_gl_backend() -> str:
    """$MUJOCO_GL if set, else egl if it works, else osmesa."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    return "egl" if gl_backend_ok("egl") else "osmesa"


def report() -> dict:
    info = {"vendor": detect_gpu_vendor()}
    try:
        info["torch_device"] = str(torch_device())
    except ImportError:
        info["torch_device"] = "torch not installed (optional: uv sync --extra xpu)"
    info["level_zero"] = level_zero_available()
    info["sycl_ls"] = sycl_devices() or "sycl-ls not found (source /opt/intel/oneapi/setvars.sh)"
    info["mujoco_gl"] = {b: gl_backend_ok(b) for b in ("egl", "osmesa")}
    info["egl_render_device"] = os.environ.get("MUJOCO_EGL_DEVICE_ID", "default")
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(report(), indent=2))
