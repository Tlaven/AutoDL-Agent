import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.supervisor_agent.graph import call_model, dynamic_tools_node
from src.supervisor_agent.state import PlannerSession, State


graph_module = importlib.import_module("src.supervisor_agent.graph")
tools_module = importlib.import_module("src.supervisor_agent.tools")


@pytest.mark.asyncio
async def test_call_model_forces_convergence_on_last_step(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_tools() -> list:
        return []

    class FakeModel:
        def bind_tools(self, _tools: list) -> "FakeModel":
            return self

        async def ainvoke(self, _messages: list) -> AIMessage:
            return AIMessage(
                id="ai_force_stop",
                content="still wants tools",
                tool_calls=[{"id": "call_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
            )

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: FakeModel())

    runtime = SimpleNamespace(context=SimpleNamespace(model="qwen:qwen-flash", system_prompt="sys"))
    state = State(messages=[HumanMessage(content="hi")], is_last_step=True)

    result = await call_model(state, runtime)

    assert len(result["messages"]) == 1
    forced_reply = result["messages"][0]
    assert isinstance(forced_reply, AIMessage)
    assert forced_reply.id == "ai_force_stop"
    assert "已达到最大执行步数限制" in str(forced_reply.content)


@pytest.mark.asyncio
async def test_call_model_returns_model_response_when_not_last_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    expected = AIMessage(id="ai_ok", content="normal response", tool_calls=[])

    class FakeModel:
        def bind_tools(self, _tools: list) -> "FakeModel":
            return self

        async def ainvoke(self, _messages: list) -> AIMessage:
            return expected

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: FakeModel())

    runtime = SimpleNamespace(context=SimpleNamespace(model="qwen:qwen-flash", system_prompt="sys"))
    state = State(messages=[HumanMessage(content="hi")], is_last_step=False)


    result = await call_model(state, runtime)

    assert result == {"messages": [expected]}


@pytest.mark.asyncio
async def test_dynamic_tools_node_updates_session_for_execute_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    meta = {
        "status": "failed",
        "error_detail": "RuntimeError: boom",
        "updated_plan_json": json.dumps({"steps": [{"step_id": "s1", "status": "failed"}]}, ensure_ascii=False),
    }
    tool_message = ToolMessage(
        content=f"summary\n\n[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}",
        tool_call_id="call_exec_1",
    )

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, _state: State) -> dict:
            return {"messages": [tool_message]}

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    state = State(

        messages=[
            AIMessage(
                content="call execute",
                tool_calls=[{"id": "call_exec_1", "name": "execute_plan", "args": {}, "type": "tool_call"}],
            )
        ],
        planner_session=PlannerSession(session_id="plan_001", plan_json='{"steps": []}'),
    )

    runtime = SimpleNamespace(context=SimpleNamespace())
    updates = await dynamic_tools_node(state, runtime)

    assert updates["messages"] == [tool_message]
    assert updates["planner_session"].session_id == "plan_001"
    assert updates["planner_session"].last_executor_status == "failed"
    assert updates["planner_session"].last_executor_error == "RuntimeError: boom"
    assert '"status": "failed"' in str(updates["planner_session"].plan_json)


@pytest.mark.asyncio
async def test_dynamic_tools_node_updates_session_for_generate_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    tool_message = ToolMessage(content='{"steps": [{"step_id": "s1"}]}', tool_call_id="call_plan_1")

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, _state: State) -> dict:
            return {"messages": [tool_message]}

    class _FixedUUID:
        hex = "abcdef0123456789"

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)
    monkeypatch.setattr(graph_module.uuid, "uuid4", lambda: _FixedUUID())


    state = State(
        messages=[
            AIMessage(
                content="call plan",
                tool_calls=[{"id": "call_plan_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
            )
        ],
        planner_session=None,
    )

    runtime = SimpleNamespace(context=SimpleNamespace())
    updates = await dynamic_tools_node(state, runtime)

    assert updates["messages"] == [tool_message]
    assert updates["planner_session"].session_id == "plan_abcdef01"
    assert updates["planner_session"].plan_json == '{"steps": [{"step_id": "s1"}]}'


@pytest.mark.asyncio
async def test_dynamic_tools_node_execute_plan_dirty_output_keeps_previous_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    tool_message = ToolMessage(
        content="执行失败，但没有返回可解析的执行元数据",
        tool_call_id="call_exec_dirty",
    )

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, _state: State) -> dict:
            return {"messages": [tool_message]}

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    previous_plan = '{"steps": [{"step_id": "s1", "status": "pending"}]}'
    state = State(
        messages=[
            AIMessage(
                content="call execute",
                tool_calls=[{"id": "call_exec_dirty", "name": "execute_plan", "args": {}, "type": "tool_call"}],
            )
        ],
        planner_session=PlannerSession(
            session_id="plan_001",
            plan_json=previous_plan,
            last_executor_status="completed",
            last_executor_error=None,
        ),
    )

    runtime = SimpleNamespace(context=SimpleNamespace())
    updates = await dynamic_tools_node(state, runtime)

    assert updates["messages"] == [tool_message]
    assert updates["planner_session"].session_id == "plan_001"
    assert updates["planner_session"].plan_json == previous_plan
    assert updates["planner_session"].last_executor_status is None
    assert updates["planner_session"].last_executor_error is None


@pytest.mark.asyncio
async def test_dynamic_tools_node_execute_plan_without_updated_plan_still_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    meta = {
        "status": "failed",
        "error_detail": "RuntimeError: partial meta",
    }
    tool_message = ToolMessage(
        content=f"summary\n\n[EXECUTOR_RESULT] {json.dumps(meta, ensure_ascii=False)}",
        tool_call_id="call_exec_partial",
    )

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, _state: State) -> dict:
            return {"messages": [tool_message]}

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    previous_plan = '{"steps": [{"step_id": "s1", "status": "pending"}]}'
    state = State(
        messages=[
            AIMessage(
                content="call execute",
                tool_calls=[{"id": "call_exec_partial", "name": "execute_plan", "args": {}, "type": "tool_call"}],
            )
        ],
        planner_session=PlannerSession(session_id="plan_001", plan_json=previous_plan),
    )

    runtime = SimpleNamespace(context=SimpleNamespace())
    updates = await dynamic_tools_node(state, runtime)

    assert updates["messages"] == [tool_message]
    assert updates["planner_session"].session_id == "plan_001"
    assert updates["planner_session"].plan_json == previous_plan
    assert updates["planner_session"].last_executor_status == "failed"
    assert updates["planner_session"].last_executor_error == "RuntimeError: partial meta"


@pytest.mark.asyncio
async def test_dynamic_tools_node_execute_plan_handles_executor_exception_and_marks_plan_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_executor(_plan_json: str):
        raise RuntimeError("executor exploded")

    monkeypatch.setattr(tools_module, "run_executor", fake_run_executor)

    previous_plan = json.dumps(
        {
            "steps": [
                {"step_id": "s1", "status": "pending"},
                {"step_id": "s2", "status": "completed"},
            ]
        },
        ensure_ascii=False,
    )
    state = State(
        messages=[
            AIMessage(
                content="call execute",
                tool_calls=[
                    {"id": "call_exec_exception", "name": "execute_plan", "args": {}, "type": "tool_call"}
                ],
            )
        ],
        planner_session=PlannerSession(session_id="plan_001", plan_json=previous_plan),
    )

    runtime = SimpleNamespace(context=SimpleNamespace())
    updates = await dynamic_tools_node(state, runtime)

    assert len(updates["messages"]) == 1
    tool_message = updates["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert "[EXECUTOR_RESULT]" in str(tool_message.content)

    assert updates["planner_session"].session_id == "plan_001"
    assert updates["planner_session"].last_executor_status == "failed"
    assert updates["planner_session"].last_executor_error == "RuntimeError: executor exploded"
    assert '"step_id": "s1"' in str(updates["planner_session"].plan_json)
    assert '"status": "failed"' in str(updates["planner_session"].plan_json)
    assert "Executor 异常中断：RuntimeError: executor exploded" in str(
        updates["planner_session"].plan_json
    )


