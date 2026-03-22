import json
import os
import re
from types import SimpleNamespace
from typing import Any, cast


import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.common.context import Context
from src.executor_agent.graph import ExecutorState, call_executor
from src.planner_agent.graph import call_planner
from src.planner_agent.state import PlannerState
from src.supervisor_agent.graph import call_model
from src.supervisor_agent.state import State

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.asyncio,
]


def _show_output(tag: str, content: str) -> None:
    print(f"\n===== {tag} OUTPUT START =====")
    print(content)
    print(f"===== {tag} OUTPUT END =====\n")


def _extract_json_text(content: str) -> str:

    text = content.strip()
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)

    if len(matches) == 1:
        return matches[0].strip()
    return text


def _assert_plan_shape(content: str) -> None:
    payload = _extract_json_text(content)
    data = json.loads(payload)

    assert isinstance(data.get("goal"), str)
    assert data["goal"].strip()
    assert isinstance(data.get("overall_expected_output"), str)
    assert data["overall_expected_output"].strip()
    assert isinstance(data.get("steps"), list)
    assert len(data["steps"]) >= 1

    first_step = data["steps"][0]
    assert isinstance(first_step.get("step_id"), str)
    assert isinstance(first_step.get("intent"), str)
    assert isinstance(first_step.get("expected_output"), str)
    assert first_step.get("status") in {"pending", "completed", "failed", "skipped"}


@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY 未配置")
async def test_live_call_planner_returns_parseable_plan_json() -> None:
    state = PlannerState(
        messages=[
            HumanMessage(
                content="请输出一个最小可执行的图像分类计划，包含 1-2 个步骤。"
            )
        ]
    )

    result = await call_planner(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.name == "planner"
    planner_content = str(msg.content)
    _show_output("PLANNER", planner_content)
    _assert_plan_shape(planner_content)



@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY 未配置")
async def test_live_call_executor_returns_text_or_convergence_guard() -> None:
    state = ExecutorState(
        messages=[
            HumanMessage(
                content="这是连通性测试，不要调用任何工具，直接回复 `LIVE_EXECUTOR_OK`。"
            )
        ],
        is_last_step=True,
    )

    result = await call_executor(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    content = str(msg.content)
    _show_output("EXECUTOR", content)
    assert content.strip()
    assert ("LIVE_EXECUTOR_OK" in content) or ("已达到最大执行步数限制" in content)



@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY 未配置")
async def test_live_call_executor_tool_call_write_file_args_are_reasonable() -> None:
    state = ExecutorState(
        messages=[
            HumanMessage(
                content=(
                    "这是工具调用能力测试。请只调用一次 `write_file` 工具，"
                    "参数要求：path 使用相对路径并以 .txt 结尾，"
                    "content 写入 `LIVE_WRITE_OK`，overwrite 为 true。"
                    "不要输出最终总结文本。"
                )
            )
        ],
        is_last_step=False,
    )

    result = await call_executor(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    _show_output("EXECUTOR_TOOL_CALLS", json.dumps(msg.tool_calls or [], ensure_ascii=False, indent=2))

    assert msg.tool_calls, "Executor 未产生任何 tool_calls，无法验证 write_file 参数质量"

    write_calls = [tc for tc in msg.tool_calls if tc.get("name") == "write_file"]
    assert write_calls, f"未调用 write_file，实际 tool_calls={msg.tool_calls}"

    call = write_calls[0]
    args = call.get("args", {})

    assert isinstance(args.get("path"), str)
    assert args["path"].strip()
    assert args["path"].endswith(".txt")
    assert not os.path.isabs(args["path"])
    assert ".." not in args["path"]

    assert isinstance(args.get("content"), str)
    assert "LIVE_WRITE_OK" in args["content"]
    assert len(args["content"]) <= 5000

    if "overwrite" in args:
        assert isinstance(args["overwrite"], bool)


@pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="SILICONFLOW_API_KEY 未配置")
async def test_live_call_model_returns_text_or_convergence_guard() -> None:
    runtime = cast(
        Any,
        SimpleNamespace(
            context=Context(
                model="siliconflow:Pro/deepseek-ai/DeepSeek-V3.2",
                system_prompt="你是测试助手。只回复 `LIVE_SUPERVISOR_OK`，不要调用工具。",
            )
        ),
    )

    state = State(messages=[HumanMessage(content="连通性测试")], is_last_step=True)

    result = await call_model(state, runtime)

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    content = str(msg.content)
    _show_output("SUPERVISOR", content)
    assert content.strip()
    assert ("LIVE_SUPERVISOR_OK" in content) or ("已达到最大执行步数限制" in content)


