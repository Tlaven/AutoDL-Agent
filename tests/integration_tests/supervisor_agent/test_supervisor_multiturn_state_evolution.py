import importlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.context import Context


graph_module = importlib.import_module("src.supervisor_agent.graph")


@pytest.mark.asyncio
async def test_supervisor_multiturn_state_evolution(monkeypatch: pytest.MonkeyPatch) -> None:
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
                    content="先生成初始计划",
                    tool_calls=[{"id": "tc_plan_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 2:
                return AIMessage(
                    content="执行初始计划",
                    tool_calls=[{"id": "tc_exec_1", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 3:
                return AIMessage(
                    content="根据失败结果重新规划",
                    tool_calls=[{"id": "tc_plan_2", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 4:
                return AIMessage(
                    content="执行修订后的计划",
                    tool_calls=[{"id": "tc_exec_2", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            return AIMessage(content="任务已完成。", tool_calls=[])

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, state: dict) -> dict:
            last_ai = state.messages[-1]
            tool_name = last_ai.tool_calls[0]["name"]
            tool_call_id = last_ai.tool_calls[0]["id"]

            if tool_name == "generate_plan" and tool_call_id == "tc_plan_1":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "execute_plan" and tool_call_id == "tc_exec_1":
                failed_meta = {
                    "status": "failed",
                    "error_detail": "RuntimeError: first attempt failed",
                    "updated_plan_json": json.dumps(
                        {
                            "steps": [
                                {
                                    "step_id": "s1",
                                    "status": "failed",
                                    "failure_reason": "RuntimeError: first attempt failed",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
                return {
                    "messages": [
                        ToolMessage(
                            content=f"第一次执行失败\n\n[EXECUTOR_RESULT] {json.dumps(failed_meta, ensure_ascii=False)}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "generate_plan" and tool_call_id == "tc_plan_2":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1_retry", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            completed_meta = {
                "status": "completed",
                "error_detail": None,
                "updated_plan_json": json.dumps(
                    {"steps": [{"step_id": "s1_retry", "status": "completed"}]},
                    ensure_ascii=False,
                ),
            }
            return {
                "messages": [
                    ToolMessage(
                        content=f"第二次执行完成\n\n[EXECUTOR_RESULT] {json.dumps(completed_meta, ensure_ascii=False)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="先执行，失败后重规划再执行")]},
        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    messages = result["messages"]
    assert len(messages) == 10
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].tool_calls == []

    planner_session = result["planner_session"]
    assert planner_session is not None
    assert planner_session.last_executor_status == "completed"
    assert planner_session.last_executor_error is None
    assert '"step_id": "s1_retry"' in str(planner_session.plan_json)
    assert '"status": "completed"' in str(planner_session.plan_json)

    execute_round_1 = messages[4]
    execute_round_2 = messages[8]
    assert isinstance(execute_round_1, ToolMessage)
    assert isinstance(execute_round_2, ToolMessage)
    assert '"status": "failed"' in str(execute_round_1.content)
    assert '"status": "completed"' in str(execute_round_2.content)


@pytest.mark.asyncio
async def test_supervisor_multiturn_converges_after_two_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
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
                    content="初始规划",
                    tool_calls=[{"id": "tc_plan_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 2:
                return AIMessage(
                    content="第一次执行",
                    tool_calls=[{"id": "tc_exec_1", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 3:
                return AIMessage(
                    content="第一次失败后重规划",
                    tool_calls=[{"id": "tc_plan_2", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 4:
                return AIMessage(
                    content="第二次执行",
                    tool_calls=[{"id": "tc_exec_2", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 5:
                return AIMessage(
                    content="第二次失败后重规划",
                    tool_calls=[{"id": "tc_plan_3", "name": "generate_plan", "args": {}, "type": "tool_call"}],
                )
            if self._calls == 6:
                return AIMessage(
                    content="第三次执行",
                    tool_calls=[{"id": "tc_exec_3", "name": "execute_plan", "args": {}, "type": "tool_call"}],
                )
            return AIMessage(content="最终完成。", tool_calls=[])

    class FakeToolNode:
        def __init__(self, _tools: list) -> None:
            pass

        async def ainvoke(self, state: dict) -> dict:
            last_ai = state.messages[-1]
            tool_name = last_ai.tool_calls[0]["name"]
            tool_call_id = last_ai.tool_calls[0]["id"]

            if tool_name == "generate_plan" and tool_call_id == "tc_plan_1":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "execute_plan" and tool_call_id == "tc_exec_1":
                failed_meta_1 = {
                    "status": "failed",
                    "error_detail": "RuntimeError: first failure",
                    "updated_plan_json": json.dumps(
                        {
                            "steps": [
                                {
                                    "step_id": "s1",
                                    "status": "failed",
                                    "failure_reason": "RuntimeError: first failure",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
                return {
                    "messages": [
                        ToolMessage(
                            content=f"第一次执行失败\n\n[EXECUTOR_RESULT] {json.dumps(failed_meta_1, ensure_ascii=False)}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "generate_plan" and tool_call_id == "tc_plan_2":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1_retry_1", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "execute_plan" and tool_call_id == "tc_exec_2":
                failed_meta_2 = {
                    "status": "failed",
                    "error_detail": "RuntimeError: second failure",
                    "updated_plan_json": json.dumps(
                        {
                            "steps": [
                                {
                                    "step_id": "s1_retry_1",
                                    "status": "failed",
                                    "failure_reason": "RuntimeError: second failure",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
                return {
                    "messages": [
                        ToolMessage(
                            content=f"第二次执行失败\n\n[EXECUTOR_RESULT] {json.dumps(failed_meta_2, ensure_ascii=False)}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            if tool_name == "generate_plan" and tool_call_id == "tc_plan_3":
                return {
                    "messages": [
                        ToolMessage(
                            content='{"steps": [{"step_id": "s1_retry_2", "status": "pending"}]}',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

            completed_meta = {
                "status": "completed",
                "error_detail": None,
                "updated_plan_json": json.dumps(
                    {"steps": [{"step_id": "s1_retry_2", "status": "completed"}]},
                    ensure_ascii=False,
                ),
            }
            return {
                "messages": [
                    ToolMessage(
                        content=f"第三次执行完成\n\n[EXECUTOR_RESULT] {json.dumps(completed_meta, ensure_ascii=False)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

    model = FakeModel()
    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: model)
    monkeypatch.setattr(graph_module, "ToolNode", FakeToolNode)

    result = await graph_module.graph.ainvoke(
        {"messages": [HumanMessage(content="连续失败后重规划，直到成功")]},
        context=Context(model="qwen:qwen-flash", system_prompt="sys"),
    )

    messages = result["messages"]
    assert len(messages) == 14
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].tool_calls == []

    planner_session = result["planner_session"]
    assert planner_session is not None
    assert planner_session.last_executor_status == "completed"
    assert planner_session.last_executor_error is None
    assert '"step_id": "s1_retry_2"' in str(planner_session.plan_json)
    assert '"status": "completed"' in str(planner_session.plan_json)

    execute_round_1 = messages[4]
    execute_round_2 = messages[8]
    execute_round_3 = messages[12]
    assert isinstance(execute_round_1, ToolMessage)
    assert isinstance(execute_round_2, ToolMessage)
    assert isinstance(execute_round_3, ToolMessage)
    assert '"status": "failed"' in str(execute_round_1.content)
    assert '"status": "failed"' in str(execute_round_2.content)
    assert '"status": "completed"' in str(execute_round_3.content)

