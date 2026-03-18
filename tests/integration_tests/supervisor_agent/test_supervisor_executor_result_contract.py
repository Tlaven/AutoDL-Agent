import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.context import Context


graph_module = importlib.import_module("src.supervisor_agent.graph")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("executor_tool_content", "expected_status", "expected_error"),
    [
        ("执行失败\n\n[EXECUTOR_RESULT] {invalid_json}", None, None),
        ("执行失败，无结构化结果", None, None),
        (
            "执行失败\n\n[EXECUTOR_RESULT] "
            + json.dumps(
                {"status": "failed", "error_detail": "RuntimeError: partial meta"},
                ensure_ascii=False,
            ),
            "failed",
            "RuntimeError: partial meta",
        ),
    ],
)
async def test_supervisor_executor_result_contract_keeps_flow_stable(
    monkeypatch: pytest.MonkeyPatch,
    executor_tool_content: str,
    expected_status: str | None,
    expected_error: str | None,
) -> None:
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
            return AIMessage(content="任务结束。", tool_calls=[])

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

            return {
                "messages": [
                    ToolMessage(content=executor_tool_content, tool_call_id=tool_call_id)
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="请规划并执行")]},

        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    planner_session = result["planner_session"]
    assert planner_session is not None
    assert planner_session.last_executor_status == expected_status
    assert planner_session.last_executor_error == expected_error
    assert '"status": "pending"' in str(planner_session.plan_json)

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].tool_calls == []
