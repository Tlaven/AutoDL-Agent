import json
import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.context import Context


graph_module = importlib.import_module("src.supervisor_agent.graph")


@pytest.mark.asyncio
async def test_supervisor_smoke_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_tools() -> list:
        return []

    class FakeModel:
        def __init__(self) -> None:
            self._calls = 0

        def bind_tools(self, _tools: list) -> "FakeModel":
            return self

        async def ainvoke(self, _messages: list) -> AIMessage:
            self._calls += 1
            if self._calls == 1:
                return AIMessage(
                    content="need plan",
                    tool_calls=[{"id": "tc_plan", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 2:
                return AIMessage(
                    content="need execute",
                    tool_calls=[{"id": "tc_exec", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            return AIMessage(content="任务完成。", tool_calls=[])

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, state: dict) -> dict:
            last_ai = state.messages[-1]
            tool_name = last_ai.tool_calls[0]["name"]
            tool_call_id = last_ai.tool_calls[0]["id"]

            if tool_name == "generate_plan":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            meta = {
                "status": "completed",
                "error_detail": None,
                "updated_plan_json": json.dumps(
                    {"steps": [{"step_id": "s1", "status": "completed"}]},
                    ensure_ascii=False,
                ),
            }
            return {
                "messages": [
                    ToolMessage(
                        content=f"执行完成\n\n[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="请帮我完成一个任务")]},
        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    assert result["planner_session"].last_executor_status == "completed"
    assert result["planner_session"].last_executor_error is None
    assert "completed" in result["planner_session"].plan_json
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].tool_calls == []


@pytest.mark.asyncio
async def test_supervisor_smoke_failure_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_tools() -> list:
        return []

    class FakeModel:
        def __init__(self) -> None:
            self._calls = 0

        def bind_tools(self, _tools: list) -> "FakeModel":
            return self

        async def ainvoke(self, _messages: list) -> AIMessage:
            self._calls += 1
            if self._calls == 1:
                return AIMessage(
                    content="need plan",
                    tool_calls=[{"id": "tc_plan", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 2:
                return AIMessage(
                    content="need execute",
                    tool_calls=[{"id": "tc_exec", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            return AIMessage(content="执行失败，建议重规划。", tool_calls=[])

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, state: dict) -> dict:
            last_ai = state.messages[-1]
            tool_name = last_ai.tool_calls[0]["name"]
            tool_call_id = last_ai.tool_calls[0]["id"]

            if tool_name == "generate_plan":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            meta = {
                "status": "failed",
                "error_detail": "RuntimeError: executor crashed",
                "updated_plan_json": json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": "s1",
                                "status": "failed",
                                "failure_reason": "RuntimeError: executor crashed",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
            return {
                "messages": [
                    ToolMessage(
                        content=f"执行失败\n\n[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="请帮我完成一个任务")]},
        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    assert result["planner_session"].last_executor_status == "failed"
    assert result["planner_session"].last_executor_error == "RuntimeError: executor crashed"
    assert "failure_reason" in result["planner_session"].plan_json
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].tool_calls == []
