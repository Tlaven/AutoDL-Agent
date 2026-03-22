import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

import src.supervisor_agent.tools as tools_module
from src.supervisor_agent.state import PlannerSession, State


class _FixedUUID:
    hex = "abcdef0123456789"


@pytest.mark.asyncio
async def test_generate_plan_uses_new_session_id_and_forwards_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_planner(*, messages, thread_id: str) -> str:
        captured["messages"] = list(messages)
        captured["thread_id"] = thread_id
        return '{"steps": []}'

    monkeypatch.setattr(tools_module, "run_planner", fake_run_planner)
    monkeypatch.setattr(tools_module.uuid, "uuid4", lambda: _FixedUUID())

    state = State(messages=[HumanMessage(content="用户需求")], planner_session=None)

    result = await tools_module.generate_plan.coroutine(state=state)

    assert result == '{"steps": []}'
    assert captured["thread_id"] == "plan_abcdef01"
    sent = captured["messages"]
    assert isinstance(sent, list)
    assert len(sent) == 1
    assert isinstance(sent[0], HumanMessage)
    assert str(sent[0].content) == "用户需求"


@pytest.mark.asyncio
async def test_generate_plan_appends_plan_context_when_session_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    existing_plan = '{"steps": [{"step_id": "s1", "status": "failed"}]}'

    async def fake_run_planner(*, messages, thread_id: str) -> str:
        captured["messages"] = list(messages)
        captured["thread_id"] = thread_id
        return '{"steps": [{"step_id": "s2"}]}'

    monkeypatch.setattr(tools_module, "run_planner", fake_run_planner)

    state = State(
        messages=[HumanMessage(content="继续")],
        planner_session=PlannerSession(session_id="plan_001", plan_json=existing_plan),
    )

    result = await tools_module.generate_plan.coroutine(state=state)

    assert result == '{"steps": [{"step_id": "s2"}]}'
    assert captured["thread_id"] == "plan_001"
    sent = captured["messages"]
    assert isinstance(sent, list)
    assert len(sent) == 2
    assert isinstance(sent[-1], HumanMessage)
    assert "[当前计划执行状态]" in str(sent[-1].content)
    assert existing_plan in str(sent[-1].content)


@pytest.mark.asyncio
async def test_execute_plan_requires_planner_session() -> None:
    state = State(messages=[HumanMessage(content="run")], planner_session=None)

    result = await tools_module.execute_plan.coroutine(state=state)

    assert "尚未生成计划" in result


@pytest.mark.asyncio
async def test_execute_plan_requires_non_empty_plan() -> None:
    state = State(
        messages=[HumanMessage(content="run")],
        planner_session=PlannerSession(session_id="plan_001", plan_json=""),
    )

    result = await tools_module.execute_plan.coroutine(state=state)

    assert "计划内容为空" in result


@pytest.mark.asyncio
async def test_execute_plan_success_returns_summary_and_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_executor(_plan_json: str):
        return SimpleNamespace(
            status="completed",
            summary="执行成功",
            updated_plan_json='{"steps": [{"step_id": "s1", "status": "completed"}]}',
        )

    monkeypatch.setattr(tools_module, "run_executor", fake_run_executor)
    monkeypatch.setattr(tools_module.uuid, "uuid4", lambda: _FixedUUID())

    state = State(
        messages=[HumanMessage(content="run")],
        planner_session=PlannerSession(session_id="plan_001", plan_json='{"steps": []}'),
    )

    result = await tools_module.execute_plan.coroutine(state=state)

    assert "执行成功" in result
    assert "[EXECUTOR_RESULT]" in result

    meta_json = result.split("[EXECUTOR_RESULT]", maxsplit=1)[1].strip()
    meta = json.loads(meta_json)

    assert meta["executor_session_id"] == "exec_abcdef01"
    assert meta["planner_session_id"] == "plan_001"
    assert meta["status"] == "completed"
    assert meta["error_detail"] is None
    assert '"status": "completed"' in meta["updated_plan_json"]


@pytest.mark.asyncio
async def test_execute_plan_exception_marks_pending_steps_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_executor(_plan_json: str):
        raise ValueError("bad executor")

    monkeypatch.setattr(tools_module, "run_executor", fake_run_executor)
    monkeypatch.setattr(tools_module.uuid, "uuid4", lambda: _FixedUUID())

    plan_json = json.dumps(
        {
            "steps": [
                {"step_id": "s1", "status": "pending"},
                {"step_id": "s2", "status": "completed"},
            ]
        },
        ensure_ascii=False,
    )
    state = State(
        messages=[HumanMessage(content="run")],
        planner_session=PlannerSession(session_id="plan_001", plan_json=plan_json),
    )

    result = await tools_module.execute_plan.coroutine(state=state)

    assert "[EXECUTOR_RESULT]" in result
    meta_json = result.split("[EXECUTOR_RESULT]", maxsplit=1)[1].strip()
    meta = json.loads(meta_json)

    assert meta["status"] == "failed"
    assert meta["error_detail"] == "ValueError: bad executor"

    updated = json.loads(meta["updated_plan_json"])
    assert updated["steps"][0]["status"] == "failed"
    assert "bad executor" in str(updated["steps"][0]["failure_reason"])
    assert updated["steps"][1]["status"] == "completed"
