"""The LLM agent loop, driven by a scripted fake LLM: no network, no API key."""

import json

import pytest

from agent import agent as agent_module
from agent.agent import run_agent
from agent.tools import TOOL_SCHEMAS, FailureAtlasTools
from failure_atlas import web


def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> dict:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def final(text: str) -> dict:
    return {"role": "assistant", "content": text}


class ScriptedClient:
    """Plays back fixed assistant replies and records every request it receives."""

    model = "scripted-fake"

    def __init__(self, replies: list[dict], repeat_last: bool = False) -> None:
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.requests: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        self.requests.append([dict(message) for message in messages])
        assert tools, "the agent must always send the tool list"
        if self.repeat_last and len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)


class FakeTools:
    """Same interface as FailureAtlasTools, with canned results and no model."""

    schemas = TOOL_SCHEMAS

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: str) -> dict:
        args = json.loads(arguments)
        self.calls.append((name, args))
        if name == "evaluate_model":
            return {"accuracy": 0.93, "ece": 0.04, "sample_count": 10}
        if name == "apply_targeted_repair":
            return {"before": {"perturbed": {"accuracy": 0.6}},
                    "after": {"perturbed": {"accuracy": 0.7}}}
        if name == "find_failure_boundary":
            return {"boundary_severity": 0.05, "checkpoint": args.get("checkpoint", "baseline")}
        raise RuntimeError("tool exploded")


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "agent_trace.json"


def test_calls_tool_feeds_result_back_and_finishes(trace_path):
    client = ScriptedClient([
        tool_call("evaluate_model", {"model": "lenet", "dataset": "mnist"}),
        final("总结：准确率 0.93。"),
    ])
    tools = FakeTools()

    trace = run_agent("评测 LeNet", client, tools, trace_path=trace_path, system_prompt="sys")

    assert trace["status"] == "finished"
    assert trace["summary"] == "总结：准确率 0.93。"
    assert tools.calls == [("evaluate_model", {"model": "lenet", "dataset": "mnist"})]
    # The second LLM request carries the tool result, linked to the call id.
    tool_message = client.requests[1][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["accuracy"] == 0.93
    saved = json.loads(trace_path.read_text(encoding="utf-8"))
    assert saved["steps"][0]["tool_calls"][0]["result"]["ece"] == 0.04
    assert saved["verified_after_repair"] is None


def test_tool_errors_are_returned_to_the_model_and_the_loop_continues(trace_path):
    # Real tool box: the bad perturbation and the broken JSON fail validation before
    # any model is loaded, so this stays fast while exercising the real error path.
    client = ScriptedClient([
        tool_call("find_failure_boundary",
                  {"model": "lenet", "dataset": "mnist", "perturbation": "fog"}),
        tool_call("rank_vulnerability", "{not json", call_id="call_2"),
        tool_call("no_such_tool", {}, call_id="call_3"),
        final("已根据错误信息调整。"),
    ])

    trace = run_agent("找边界", client, FailureAtlasTools(), trace_path=trace_path,
                      system_prompt="sys")

    assert trace["status"] == "finished"
    results = [step["tool_calls"][0] for step in trace["steps"][:3]]
    assert all(not call["ok"] for call in results)
    assert "fog" in results[0]["result"]["error"]
    assert "JSON" in results[1]["result"]["error"]
    assert "未知工具" in results[2]["result"]["error"]
    assert "fog" in client.requests[1][-1]["content"]  # the model saw the error


def test_a_raising_tool_box_does_not_crash_the_loop(trace_path):
    client = ScriptedClient([tool_call("rank_vulnerability", {}), final("完成")])
    trace = run_agent("x", client, FakeTools(), trace_path=trace_path, system_prompt="sys")
    assert trace["status"] == "finished"
    assert trace["steps"][0]["tool_calls"][0]["result"] == {"error": "RuntimeError: tool exploded"}


def test_stops_at_max_steps(trace_path):
    client = ScriptedClient(
        [tool_call("evaluate_model", {"model": "lenet", "dataset": "mnist"})], repeat_last=True
    )
    tools = FakeTools()

    trace = run_agent("永不结束", client, tools, max_steps=3, trace_path=trace_path,
                      system_prompt="sys")

    assert trace["status"] == "max_steps_reached"
    assert len(client.requests) == 3 and len(tools.calls) == 3
    assert len(trace["steps"]) == 3
    assert json.loads(trace_path.read_text(encoding="utf-8"))["status"] == "max_steps_reached"


def test_flags_whether_the_repair_was_re_tested(trace_path):
    args = {"model": "lenet", "dataset": "mnist", "perturbation": "rotation"}
    unverified = run_agent("修复", ScriptedClient([
        tool_call("apply_targeted_repair", args), final("修好了")
    ]), FakeTools(), trace_path=trace_path, system_prompt="sys")
    verified = run_agent("修复", ScriptedClient([
        tool_call("apply_targeted_repair", args),
        tool_call("find_failure_boundary", {**args, "checkpoint": "repaired"}),
        final("复测后边界后移"),
    ]), FakeTools(), trace_path=trace_path, system_prompt="sys")

    assert unverified["verified_after_repair"] is False
    assert verified["verified_after_repair"] is True


def test_llm_failure_ends_the_run_with_a_trace(trace_path):
    class BrokenClient:
        def chat(self, messages, tools):
            raise RuntimeError("connection refused")

    trace = run_agent("x", BrokenClient(), FakeTools(), trace_path=trace_path,
                      system_prompt="sys")
    assert trace["status"] == "llm_error"
    assert "connection refused" in trace["summary"]


def test_agent_run_endpoint_returns_summary_and_trajectory(tmp_path, monkeypatch):
    client = ScriptedClient([
        tool_call("evaluate_model", {"model": "lenet", "dataset": "mnist"}),
        final("总结完成"),
    ])
    monkeypatch.setattr(web, "DEMO_MODE", False)
    monkeypatch.setattr(web, "build_agent_runtime", lambda: (client, FakeTools()))
    monkeypatch.setattr(agent_module, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(agent_module, "build_system_prompt", lambda: "sys")

    response = web.agent_run(web.AgentRunRequest(goal="评测 LeNet"))

    assert response["status"] == "finished"
    assert response["summary"] == "总结完成"
    assert response["steps"][0]["tool_calls"][0]["name"] == "evaluate_model"
    assert (tmp_path / "agent_trace.json").is_file()


def test_agent_run_endpoint_is_disabled_in_demo_mode(monkeypatch):
    monkeypatch.setattr(web, "DEMO_MODE", True)
    with pytest.raises(web.HTTPException) as error:
        web.agent_run(web.AgentRunRequest(goal="x"))
    assert error.value.status_code == 403
