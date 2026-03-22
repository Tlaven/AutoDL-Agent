import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.executor_agent.graph import ExecutorState, _parse_executor_output, call_executor, tools_node



graph_module = importlib.import_module("src.executor_agent.graph")


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
async def test_call_executor_binds_tools_and_passes_system_plus_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [object()]
    fake_model = FakeModel(AIMessage(id="ai_ok", content="ok", tool_calls=[]))

    monkeypatch.setattr(graph_module, "get_executor_tools", lambda: tools)
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_executor_system_prompt", lambda _caps: "exec_prompt")
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)

    state = ExecutorState(messages=[HumanMessage(content="execute plan")], is_last_step=False)

    result = await call_executor(state)

    assert fake_model.bound_tools == tools
    assert fake_model.received_messages is not None
    assert fake_model.received_messages[0] == {"role": "system", "content": "exec_prompt"}
    assert isinstance(fake_model.received_messages[1], HumanMessage)
    assert str(fake_model.received_messages[1].content) == "execute plan"
    assert result == {"messages": [fake_model.response]}


@pytest.mark.asyncio
async def test_call_executor_forces_convergence_when_last_step_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel(
        AIMessage(
            id="ai_force_stop",
            content="still needs tools",
            tool_calls=[{"id": "call_1", "name": "run_local_command", "args": {}, "type": "tool_call"}],
        )
    )

    monkeypatch.setattr(graph_module, "get_executor_tools", lambda: [])
    monkeypatch.setattr(graph_module, "get_executor_capabilities_docs", lambda: "capabilities")
    monkeypatch.setattr(graph_module, "get_executor_system_prompt", lambda _caps: "exec_prompt")
    monkeypatch.setattr(graph_module, "load_chat_model", lambda _model: fake_model)

    state = ExecutorState(messages=[HumanMessage(content="execute plan")], is_last_step=True)

    result = await call_executor(state)

    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.id == "ai_force_stop"
    assert "已达到最大执行步数限制" in str(msg.content)


@pytest.mark.asyncio
async def test_tools_node_passes_write_file_args_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    @tool
    def write_file(path: str, content: str, overwrite: bool = True) -> dict[str, object]:
        """测试用 write_file 工具。"""
        captured["path"] = path
        captured["content"] = content
        captured["overwrite"] = overwrite
        return {"ok": True, "path": path}


    monkeypatch.setattr(graph_module, "get_executor_tools", lambda: [write_file])

    state = ExecutorState(
        messages=[
            AIMessage(
                content="call write_file",
                tool_calls=[
                    {
                        "id": "call_write_1",
                        "name": "write_file",
                        "args": {
                            "path": "artifacts/sample.txt",
                            "content": "hello tool",
                            "overwrite": False,
                        },
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    result = await tools_node(state)

    assert captured == {
        "path": "artifacts/sample.txt",
        "content": "hello tool",
        "overwrite": False,
    }
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)


def test_parse_executor_output_extracts_protocol_fields() -> None:
    content = """执行完成
```json
{
  "status": "completed",
  "summary": "all good",
  "updated_plan": {
    "goal": "g",
    "steps": [
      {
        "step_id": "step_1",
        "intent": "i",
        "expected_output": "o",
        "status": "completed",
        "result_summary": "done",
        "failure_reason": null
      }
    ],
    "overall_expected_output": "final"
  }
}
```
"""

    result = _parse_executor_output(content)

    assert result.status == "completed"
    assert result.summary == "all good"
    assert '"step_id": "step_1"' in result.updated_plan_json
    assert '"status": "completed"' in result.updated_plan_json

