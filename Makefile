# sdk-arm-push. `make help` lists targets. Needs uv (pacman -S uv) and ffmpeg for `make demo`.
SHELL := bash
.DEFAULT_GOAL := help

help: ## list targets
	@awk 'BEGIN{FS=":.*?## "} /^[a-z0-9_-]+:.*?## /{printf "  \033[32m%-14s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

setup: ## create .venv from uv.lock (python 3.12) + copy .env.example
	uv sync --frozen --extra render
	@[ -f .env ] || cp .env.example .env

setup-xpu: ## also install torch XPU wheels (optional; only gpu-info uses them today)
	uv sync --frozen --extra render --extra xpu

test: ## unit tests (CPU, ~10 s)
	uv run --frozen pytest -q tests

validate: ## scripted reference on held-out seeds through vf-validate (no model)
	uv run --frozen vf-validate sdk-arm-push -n 20 --runtime.type subprocess --taskset.seed-offset 1000 -o outputs/validate

noop: ## check that doing nothing scores zero
	uv run --frozen vf-validate sdk-arm-push -n 5 --runtime.type subprocess --only-noop -o outputs/noop

eval-scripted: ## whole eval path with a fake model that plays the scripted policy
	scripts/eval_scripted.sh

serve: ## start llama.cpp server (SYCL on Arc -> other GPU -> CPU); set GGUF/LLAMA_SERVER in .env
	scripts/serve_local.sh

eval: ## evaluate the model at MODEL_BASE_URL (local server or API; see .env)
	scripts/eval.sh

demo: ## render the scripted solver to demo.mp4 (MUJOCO_GL=egl, falls back to osmesa)
	. scripts/_env.sh && uv run --frozen python scripts/render_demo.py --seeds 1001 1003 1007 1012 --out demo.mp4

gpu-info: ## what this machine has: vendor, torch device, Level Zero, sycl-ls, EGL/OSMesa
	uv run --frozen python -m sdk_arm_push.device

clean: ## remove outputs and caches
	rm -rf outputs .pytest_cache **/__pycache__
