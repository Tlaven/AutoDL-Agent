import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common.prompts import SYSTEM_PROMPT
from src.planner_agent.graph import call_planner
from src.planner_agent.state import PlannerState


graph_module = importlib.import_module("src.planner_agent.graph")


class FakeModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.received_messages = None

    async def ainvoke(self, messages):
        self.received_messages = messages
        return self.response


@pytest.mark.asyncio
async def test_call_planner_filters_ai_messages_with_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = FakeModel(AIMessage(content='```json\n{"goal":"g","steps":[],"overall_expected_output":"o"}\n```'))

    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_planner_system_prompt", lambda _caps: "planner_prompt")

    state = PlannerState(
        messages=[
            HumanMessage(content="u1"),
            AIMessage(
                content="tool call msg",
                tool_calls=[{"id": "call_1", "name": "x", "args": {}, "type": "tool_call"}],
            ),
            AIMessage(content="plain ai"),
            HumanMessage(content="u2"),
        ]
    )

    await call_planner(state)

    assert fake_model.received_messages is not None
    received = fake_model.received_messages

    assert isinstance(received[0], SystemMessage)
    assert SYSTEM_PROMPT.strip() in str(received[0].content)
    assert [type(m).__name__ for m in received] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "SystemMessage",
    ]
    assert str(received[1].content) == "u1"
    assert str(received[2].content) == "plain ai"
    assert str(received[3].content) == "u2"
    assert str(received[-1].content) == "planner_prompt"


@pytest.mark.asyncio
async def test_call_planner_injects_prompts_in_expected_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = FakeModel(AIMessage(content='```json\n{"goal":"g","steps":[],"overall_expected_output":"o"}\n```'))

    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_planner_system_prompt", lambda _caps: "planner_prompt")

    state = PlannerState(messages=[HumanMessage(content="u")])

    await call_planner(state)

    assert fake_model.received_messages is not None
    received = fake_model.received_messages
    assert isinstance(received[0], SystemMessage)
    assert SYSTEM_PROMPT.strip() in str(received[0].content)
    assert isinstance(received[-1], SystemMessage)
    assert str(received[-1].content) == "planner_prompt"


@pytest.mark.asyncio
async def test_call_planner_returns_named_ai_message(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = FakeModel(AIMessage(content="  {\"goal\":\"g\"}  "))

    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_planner_system_prompt", lambda _caps: "planner_prompt")

    result = await call_planner(PlannerState(messages=[HumanMessage(content="u")]))

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.name == "planner"
    assert msg.content == '{"goal":"g"}'


@pytest.mark.asyncio
async def test_call_planner_raises_when_model_returns_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel(AIMessage(content="   "))

    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_planner_system_prompt", lambda _caps: "planner_prompt")

    with pytest.raises(RuntimeError, match="Planner 未返回文本内容"):
        await call_planner(PlannerState(messages=[HumanMessage(content="u")]))
