import importlib
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.supervisor_agent.graph import call_model
from src.supervisor_agent.state import State


graph_module = importlib.import_module("src.supervisor_agent.graph")


class FakeModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_tools = None
        self.received_messages = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.received_messages = messages
        return self.response


@pytest.mark.asyncio
async def test_call_model_binds_tools_and_passes_system_plus_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [object()]

    async def fake_get_tools() -> list:
        return tools

    fake_model = FakeModel(AIMessage(id="ai_ok", content="ok", tool_calls=[]))

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)

    runtime = SimpleNamespace(context=SimpleNamespace(model="qwen:qwen-flash", system_prompt="sys_prompt"))
    state = State(messages=[HumanMessage(content="hello")], is_last_step=False)

    result = await call_model(state, runtime)

    assert fake_model.bound_tools == tools
    assert fake_model.received_messages is not None
    assert fake_model.received_messages[0] == {"role": "system", "content": "sys_prompt"}
    assert isinstance(fake_model.received_messages[1], HumanMessage)
    assert str(fake_model.received_messages[1].content) == "hello"
    assert result == {"messages": [fake_model.response]}


@pytest.mark.asyncio
async def test_call_model_forces_convergence_when_last_step_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    fake_model = FakeModel(
        AIMessage(
            id="ai_force_stop",
            content="need tools",
            tool_calls=[{"id": "call_1", "name": "generate_plan", "args": {}, "type": "tool_call"}],
        )
    )

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)

    runtime = SimpleNamespace(context=SimpleNamespace(model="qwen:qwen-flash", system_prompt="sys_prompt"))
    state = State(messages=[HumanMessage(content="hello")], is_last_step=True)

    result = await call_model(state, runtime)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.id == "ai_force_stop"
    assert "已达到最大执行步数限制" in str(msg.content)


@pytest.mark.asyncio
async def test_call_model_returns_original_message_on_last_step_without_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_tools() -> list:
        return []

    expected = AIMessage(id="ai_done", content="final answer", tool_calls=[])
    fake_model = FakeModel(expected)

    monkeypatch.setattr(graph_module, "get_tools", fake_get_tools)
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)

    runtime = SimpleNamespace(context=SimpleNamespace(model="qwen:qwen-flash", system_prompt="sys_prompt"))
    state = State(messages=[HumanMessage(content="hello")], is_last_step=True)

    result = await call_model(state, runtime)

    assert result == {"messages": [expected]}
