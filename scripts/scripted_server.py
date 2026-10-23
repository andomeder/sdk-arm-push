"""A fake OpenAI-compatible chat server that plays the scripted reference policy as
tool calls. No model, no GPU. It exists to test the full vf-eval path (harness -> model
endpoint -> MCP tools -> MuJoCo -> reward) before you point the eval at a real model.

    python scripts/scripted_server.py --port 8099
    vf-eval sdk-arm-push ... --client.base-url http://127.0.0.1:8099/v1 -m scripted
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUSH_Z, CLEAR_Z, BACKOFF, DONE_TOL = 0.026, 0.09, 0.045, 0.018


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _obs(text: str) -> dict | None:
    for m in re.finditer(r"\{.*\}", text, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "box.xyz" in d:
            return d
    return None


def _tool_name(tools: list, suffix: str) -> str:
    for t in tools or []:
        name = t.get("function", {}).get("name", "")
        if name.endswith(suffix):
            return name
    return suffix


def next_action(body: dict) -> tuple[str | None, dict | None]:
    """Return (tool_name, args) or (None, None) for a final text reply."""
    msgs, tools = body.get("messages", []), body.get("tools", [])
    last_obs, since = None, 0
    cycles = 0
    for m in msgs:
        if m.get("role") == "tool":
            o = _obs(_text(m.get("content")))
            if o is not None:
                last_obs, since = o, 0
                cycles += 1
        elif m.get("role") == "assistant":
            since += len(m.get("tool_calls") or [])
    observe = _tool_name(tools, "observe")
    if last_obs is None:
        return observe, {}
    bx, by, _ = last_obs["box.xyz"]
    gx, gy = last_obs["goal.xy"]
    dist = math.hypot(gx - bx, gy - by)
    if dist <= DONE_TOL or cycles > 12:
        return None, None
    ux, uy = (gx - bx) / dist, (gy - by) / dist
    px, py = bx - BACKOFF * ux, by - BACKOFF * uy
    s = min(dist, 0.06)
    plan = [
        (px, py, CLEAR_Z),
        (px, py, PUSH_Z),
        (px + s * ux, py + s * uy, PUSH_Z),
        (px + (s - 0.02) * ux, py + (s - 0.02) * uy, CLEAR_Z),
    ]
    if since < len(plan):
        x, y, z = plan[since]
        return _tool_name(tools, "pose"), {"x": round(x, 4), "y": round(y, 4), "z": z}
    return observe, {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": "scripted", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        tool, args = next_action(body)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if tool is None:
            msg = {"role": "assistant", "content": "Box is on the goal. Done."}
            finish = "stop"
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }],
            }
            finish = "tool_calls"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            delta = dict(msg)
            if "tool_calls" in delta:
                delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
            for chunk in (
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
            ):
                chunk.update(id=cid, object="chat.completion.chunk", created=int(time.time()), model="scripted")
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json({
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": "scripted",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage,
        })

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    print(f"scripted policy server on http://127.0.0.1:{a.port}/v1")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
