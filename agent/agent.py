"""The agent loop: observe -> decide -> call a tool -> verify, driven by an LLM.

The LLM is reached through any OpenAI-compatible ``/chat/completions`` endpoint with
tool calling (OpenAI, DeepSeek, Qwen, a local Ollama, ...). It is configured with
three environment variables:

    BASE_URL   e.g. https://api.openai.com/v1  or  http://localhost:11434/v1 (Ollama)
    API_KEY    the provider's key (Ollama ignores it)
    MODEL      e.g. gpt-4o-mini  or  qwen2.5:7b

Run from the repository root:

    python -m agent.agent "帮我找出 MobileNetV3 在 CIFAR-10 上最脆弱的扰动并修复"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .tools import DEFAULT_OUTPUT_DIR, PERTURBATION_IDS, available_selections


DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = """你是 FailureAtlas 的模型鲁棒性分析 Agent。你通过调用工具分析图像分类模型在分布偏移下的失效边界，并决定是否做定向修复。

工作流程（必须按顺序）：
1. 先调用 evaluate_model 评测模型的基础准确率和 ECE。
2. 调用 rank_vulnerability 找出最脆弱的扰动，再用 find_failure_boundary 确认该扰动的失效边界。
3. 根据证据决定是否修复：如果最脆弱扰动在较小严重度就失效（准确率下降 ≥ 10 个百分点），调用 apply_targeted_repair 修复该扰动。
4. 修复后必须复测：用 checkpoint="repaired" 再次调用 find_failure_boundary（同一扰动、同一阈值），确认边界是否后移、干净准确率是否明显下降。
5. 最后不再调用工具，用中文给出总结：最脆弱的扰动和类别、修复前后的关键数字、结论是否可靠。

规则：
- 所有数字只能来自工具返回结果，不得编造；注明数据来源（source）和样本数（sample_count）。
- 工具返回 error 时，阅读错误信息，修正参数后重试或换一种做法，不要重复同样的错误调用。
- 每次只做必要的调用，修复很耗时，最多修复一次。

可用的模型/数据集组合（已有训练好的基准模型）：{selections}
扰动类型：{perturbations}"""


class ChatClient(Protocol):
    """Anything that turns (messages, tools) into one assistant message dict."""

    def chat(self, messages: list[dict], tools: list[dict]) -> dict: ...


class ToolBox(Protocol):
    schemas: list[dict]

    def call(self, name: str, arguments: str | dict | None) -> dict: ...


class OpenAICompatibleClient:
    """A tiny standard-library client for ``POST {base_url}/chat/completions``."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 180) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        model = os.getenv("MODEL")
        if not model:
            raise ValueError("请设置环境变量 MODEL（以及 BASE_URL、API_KEY）。")
        base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
        return cls(base_url, os.getenv("API_KEY", "not-needed"), model)

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"LLM 接口返回 HTTP {error.code}: {detail}") from error
        return payload["choices"][0]["message"]


def build_system_prompt() -> str:
    selections = ", ".join(f"{model}/{dataset}" for model, dataset in available_selections())
    return SYSTEM_PROMPT.format(
        selections=selections or "无", perturbations=", ".join(PERTURBATION_IDS)
    )


def run_agent(
    goal: str,
    client: ChatClient,
    tools: ToolBox,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_path: Path | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Loop until the model answers without tool calls or ``max_steps`` LLM calls are used.

    Returns the trace dict, which is also written to ``trace_path`` after every step so a
    crashed or interrupted run still leaves its evidence behind.
    """
    trace_path = trace_path or DEFAULT_OUTPUT_DIR / "agent_trace.json"
    messages = [
        {"role": "system", "content": system_prompt or build_system_prompt()},
        {"role": "user", "content": goal},
    ]
    trace = {
        "goal": goal,
        "model": getattr(client, "model", type(client).__name__),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_steps": max_steps,
        "steps": [],
        "status": "running",
        "summary": None,
    }

    for step_number in range(1, max_steps + 1):
        try:
            reply = client.chat(messages, tools.schemas)
        except Exception as error:  # network or provider failure: stop, keep the trace
            trace.update(status="llm_error", summary=f"调用大模型失败：{error}")
            break
        tool_calls = reply.get("tool_calls") or []
        messages.append({
            "role": "assistant",
            "content": reply.get("content") or "",
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })
        step = {"step": step_number, "thought": reply.get("content") or "", "tool_calls": []}
        trace["steps"].append(step)

        if not tool_calls:  # the model gave its final conclusion
            trace.update(status="finished", summary=reply.get("content") or "")
            break

        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments") or "{}"
            started = time.perf_counter()
            try:
                result = tools.call(name, arguments)
            except Exception as error:  # never let one tool crash the loop
                result = {"error": f"{type(error).__name__}: {error}"}
            step["tool_calls"].append({
                "name": name,
                "arguments": _parse_arguments(arguments),
                "result": result,
                "ok": "error" not in result,
                "seconds": round(time.perf_counter() - started, 2),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", name),
                "content": json.dumps(result, ensure_ascii=False),
            })
        _write_trace(trace_path, trace)
    else:
        trace.update(
            status="max_steps_reached",
            summary=f"已达到最大步数 {max_steps}，Agent 尚未给出最终结论。",
        )

    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    trace["verified_after_repair"] = _verified_after_repair(trace)
    _write_trace(trace_path, trace)
    trace["trace_path"] = str(trace_path)
    return trace


def _parse_arguments(arguments: str | dict) -> dict | str:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments  # keep what the model sent, so the trace shows the mistake


def _verified_after_repair(trace: dict) -> bool | None:
    """None if no repair ran; otherwise whether a successful repaired re-test followed it."""
    calls = [call for step in trace["steps"] for call in step["tool_calls"]]
    repairs = [i for i, call in enumerate(calls)
               if call["name"] == "apply_targeted_repair" and call["ok"]]
    if not repairs:
        return None
    return any(
        call["ok"] and isinstance(call["arguments"], dict)
        and call["arguments"].get("checkpoint") == "repaired"
        for call in calls[repairs[-1] + 1:]
    )


def _write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="FailureAtlas LLM agent")
    parser.add_argument("goal", help="自然语言目标")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--full", action="store_true", help="使用完整数据集（很慢），默认演示模式")
    parser.add_argument("--trace", type=Path, default=DEFAULT_OUTPUT_DIR / "agent_trace.json")
    args = parser.parse_args()

    from .tools import FailureAtlasTools, ToolSettings

    tools = FailureAtlasTools(ToolSettings(demo=not args.full))
    trace = run_agent(
        args.goal, OpenAICompatibleClient.from_env(), tools,
        max_steps=args.max_steps, trace_path=args.trace,
    )
    for step in trace["steps"]:
        for call in step["tool_calls"]:
            mark = "ok " if call["ok"] else "ERR"
            print(f"[step {step['step']}] {mark} {call['name']}({call['arguments']}) {call['seconds']}s")
    print(f"\n状态：{trace['status']}\n\n{trace['summary']}\n\n轨迹已写入 {trace['trace_path']}")


if __name__ == "__main__":
    main()
