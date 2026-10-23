from sdk_arm_push import reference, sim


def test_scene_is_deterministic():
    assert sim.make_scene(7) == sim.make_scene(7)


def test_state_round_trips_through_json():
    import json

    s = sim.initial_state(sim.make_scene(3))
    s2 = json.loads(json.dumps(s))
    assert sim.observe(s) == sim.observe(s2)


def test_doing_nothing_fails():
    s = sim.initial_state(sim.make_scene(1))
    sim.home(s)
    assert not sim.success(s)


def test_unreachable_pose_is_blocked_and_nothing_moves():
    s = sim.initial_state(sim.make_scene(1))
    before = sim.observe(s)["ee.xyz"]
    r = sim.pose(s, 2.0, 0.0, 0.1)
    assert r["ok"] is False and r["blocked"] in ("no_ik_solution",) or r["blocked"].startswith("limit:")
    assert sim.observe(s)["ee.xyz"] == before


def test_pose_into_table_is_blocked():
    s = sim.initial_state(sim.make_scene(1))
    r = sim.pose(s, 0.3, 0.0, -0.05)
    assert r["ok"] is False


def test_estop_latch_refuses_motion():
    s = sim.initial_state(sim.make_scene(1))
    sim.estop(s)
    assert sim.pose(s, 0.3, 0.0, 0.1)["blocked"] == "estop_latched"
    sim.reset_latch(s)
    assert sim.pose(s, 0.3, 0.0, 0.1)["ok"]


def test_budget_is_enforced():
    s = sim.initial_state(sim.make_scene(1))
    for _ in range(3):
        sim.home(s, budget=2)
    assert sim.home(s, budget=2)["blocked"] == "budget_exhausted"


def test_reference_solves_held_out_seeds():
    solved = sum(reference.solve(sim.initial_state(sim.make_scene(1000 + i))) for i in range(10))
    assert solved >= 9


def test_tool_server_module_is_runnable():
    # verifiers launches the tool server with `python -m <module of the Toolset class>`.
    from sdk_arm_push.tools import ArmToolset

    assert ArmToolset.__module__ == "sdk_arm_push.tools"


def test_scripted_server_plans_observe_then_pose():
    import importlib.util
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "scripted_server.py"
    spec = importlib.util.spec_from_file_location("scripted_server", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = [{"function": {"name": "arm_observe"}}, {"function": {"name": "arm_pose"}}]
    assert mod.next_action({"messages": [], "tools": tools}) == ("arm_observe", {})
    obs = {"box.xyz": [0.3, 0.0, 0.02], "goal.xy": [0.3, 0.1]}
    msgs = [{"role": "assistant", "tool_calls": [{}]}, {"role": "tool", "content": json.dumps(obs)}]
    name, args = mod.next_action({"messages": msgs, "tools": tools})
    assert name == "arm_pose" and set(args) == {"x", "y", "z"}
