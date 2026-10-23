# sdk-arm-push

A model drives a 4-joint arm in MuJoCo and pushes a box onto a goal. The model calls tools. MuJoCo scores the box.

The arm is an MJCF written for this package. The license is Apache-2.0.

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). On Arch:

```bash
sudo pacman -S --needed uv make mesa
make setup
make validate
```

`make setup` creates `.venv` from `uv.lock` and copies `.env.example` to `.env`. `make validate` runs the scripted solver on held-out seeds 1001-1020. A pass prints `valid_rate` 1.0.

`make noop` must score 0. `make eval-scripted` runs that solver through `vf-eval`. `make demo` writes `demo.mp4`.

Pins are Python 3.12, `mujoco==3.16.0`, and verifiers at `c5fd058`.

## Results on an Arc B580

Measured on 2026-10-23 with llama.cpp SYCL, one server slot, seeds 1001-1032, 32 episodes, and a limit of 100 turns. Tokens per second are decode speed from llama-server.

| Run | Result |
| --- | --- |
| Scripted solver | 8/8, reward 1.25, 0 collisions |
| No-op | reward 0 |
| Qwen2.5-3B-Instruct Q4_K_M | 0/32, 119.1 tokens/s, 7.8 turns |
| Qwen2.5-7B-Instruct Q4_K_M | 0 of 32 started, 58.6 tokens/s, 15.2 turns |

`push-1010` and `push-1016` sent tool-call arguments that llama.cpp could not parse as JSON, so those two have no score. The other 30 scored 0.

[demo.mp4](demo.mp4) is the scripted solver on seeds 1001, 1003, 1007, and 1012. Each seed reaches the goal with 0 collisions.

## Run a local model

Set `LLAMA_SERVER` and `GGUF` in `.env`. Then:

```bash
make serve
N=32 MAX_TURNS=100 make eval
```

`make serve` loads the GGUF on one slot. Traces land in `outputs/`.

## Score

`box_on_goal` is 1.0 when the box centre is within 3 cm of the goal in x and y and the box is still on the table. `clean_success` adds 0.25 when that happens with zero arm-table collisions. The budget is 40 motion commands.

Tools are `arm_observe`, `arm_pose`, `arm_jog`, `arm_home`, `arm_estop`, and `arm_reset_latch`.
