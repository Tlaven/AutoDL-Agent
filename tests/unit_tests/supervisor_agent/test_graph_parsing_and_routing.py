import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.supervisor_agent.graph import (
    _build_id_to_name,
    _extract_executor_status,
    _extract_updated_plan_from_executor,
    route_model_output,
)
from src.supervisor_agent.state import State


def test_extract_updated_plan_from_executor_valid_meta() -> None:
    updated_plan = json.dumps({"steps": [{"step_id": "step_1", "status": "completed"}]}, ensure_ascii=False)
    content = (
        "执行摘要\n\n"
        f"[EXECUTOR_RESULT] {json.dumps({'updated_plan_json': updated_plan, 'status': 'completed'}, ensure_ascii=False)}"
    )

    assert _extract_updated_plan_from_executor(content) == updated_plan


def test_extract_updated_plan_from_executor_missing_marker() -> None:
    assert _extract_updated_plan_from_executor("no meta here") is None


def test_extract_updated_plan_from_executor_invalid_json_after_marker() -> None:
    content = "summary\n[EXECUTOR_RESULT] {not a valid json}"
    assert _extract_updated_plan_from_executor(content) is None


def test_extract_executor_status_valid_meta() -> None:
    content = (
        "summary\n\n"
        f"[EXECUTOR_RESULT] {json.dumps({'status': 'failed', 'error_detail': 'boom'}, ensure_ascii=False)}"
    )

    assert _extract_executor_status(content) == ("failed", "boom")


def test_extract_executor_status_missing_marker() -> None:
    assert _extract_executor_status("plain text") == (None, None)


def test_extract_executor_status_invalid_json_after_marker() -> None:
    assert _extract_executor_status("[EXECUTOR_RESULT] {invalid}") == (None, None)


def test_build_id_to_name_from_last_ai_message() -> None:
    state = State(
        messages=[
            AIMessage(
                content="tool calls",
                tool_calls=[
                    {"id": "call_1", "name": "generate_plan", "args": {}, "type": "tool_call"},
                    {"id": "call_2", "name": "execute_plan", "args": {}, "type": "tool_call"},
                ],
            )
        ]
    )

    assert _build_id_to_name(state) == {
        "call_1": "generate_plan",
        "call_2": "execute_plan",
    }


def test_build_id_to_name_returns_empty_when_last_message_not_ai() -> None:
    state = State(messages=[HumanMessage(content="hello")])
    assert _build_id_to_name(state) == {}


def test_route_model_output_to_tools() -> None:
    state = State(
        messages=[
            AIMessage(
                content="need tools",
                tool_calls=[{"id": "call_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
            )
        ]
    )

    assert route_model_output(state) == "tools"


def test_route_model_output_to_end() -> None:
    state = State(messages=[AIMessage(content="final answer", tool_calls=[])])
    assert route_model_output(state) == "__end__"


def test_route_model_output_raises_on_non_ai_message() -> None:
    state = State(messages=[HumanMessage(content="user")])
    with pytest.raises(ValueError):
        route_model_output(state)
