import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.context import Context


graph_module = importlib.import_module("src.supervisor_agent.graph")


def _assert_message_trajectory(result: dict, expected_status: str, expected_error: str | None) -> None:
    messages = result["messages"]
    assert len(messages) == 6

    assert isinstance(messages[0], HumanMessage)

    first_ai = messages[1]
    assert isinstance(first_ai, AIMessage)
    assert first_ai.tool_calls[0]["name"] == "generate_plan"
    assert first_ai.tool_calls[0]["id"] == "tc_plan"

    first_tool = messages[2]
    assert isinstance(first_tool, ToolMessage)
    assert first_tool.tool_call_id == "tc_plan"

    second_ai = messages[3]
    assert isinstance(second_ai, AIMessage)
    assert second_ai.tool_calls[0]["name"] == "execute_plan"
    assert second_ai.tool_calls[0]["id"] == "tc_exec"

    second_tool = messages[4]
    assert isinstance(second_tool, ToolMessage)
    assert second_tool.tool_call_id == "tc_exec"

    final_ai = messages[5]
    assert isinstance(final_ai, AIMessage)
    assert final_ai.tool_calls == []

    planner_session = result["planner_session"]
    assert planner_session is not None
    assert planner_session.last_executor_status == expected_status
    assert planner_session.last_executor_error == expected_error

    if expected_status == "completed":
        assert '"status": "completed"' in str(planner_session.plan_json)
    else:
        assert '"status": "failed"' in str(planner_session.plan_json)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("executor_status", "error_detail"),
    [
        ("completed", None),
        ("failed", "RuntimeError: executor crashed"),
    ],
)
async def test_supervisor_internal_trajectory_is_machine_verifiable(
    monkeypatch: pytest.MonkeyPatch,
    executor_status: str,
    error_detail: str | None,
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
                    tool_calls=[
                        {
                            "id": "tc_plan",
                            "name": "generate_plan",
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
                )
            if self._calls == 2:
                return AIMessage(
                    content="need execute",
                    tool_calls=[
                        {
                            "id": "tc_exec",
                            "name": "execute_plan",
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
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

            plan_step_status = "completed" if executor_status == "completed" else "failed"
            meta = {
                "status": executor_status,
                "error_detail": error_detail,
                "updated_plan_json": json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": "s1",
                                "status": plan_step_status,
                                "failure_reason": error_detail,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
            return {
                "messages": [
                    ToolMessage(
                        content=f"executor result\n\n[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)


    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="请自动完成规划与执行")]},
        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    _assert_message_trajectory(result, expected_status=executor_status, expected_error=error_detail)
