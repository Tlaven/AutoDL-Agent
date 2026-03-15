"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import uuid
import logging
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from src.common.context import Context
from src.common.utils import load_chat_model
from src.supervisor_agent.tools import get_tools
from src.supervisor_agent.state import InputState, State, PlannerSession

logger = logging.getLogger(__name__)


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """调用 LLM 支持 Agent。
    负责准备提示、初始化模型并处理响应。
    """
    available_tools = await get_tools()
    model = load_chat_model(runtime.context.model).bind_tools(available_tools)
    system_message = runtime.context.system_prompt
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # 达到最大步数时，若模型仍想调用工具，强制终止并返回说明
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="已达到最大执行步数限制，无法继续调用工具。请根据已有信息给出最终答复。",
                )
            ]
        }

    return {"messages": [response]}


async def dynamic_tools_node(
    state: State, runtime: Runtime[Context]
) -> Dict:
    """动态执行工具，并同步更新 PlannerSession 状态。

    执行完工具后，检查是否有 generate_plan 的输出，
    若有则将 plan_json 写入 state.planner_session，供 execute_plan 读取。
    """
    available_tools = await get_tools()
    tool_node = ToolNode(available_tools)
    result = await tool_node.ainvoke(state)

    # 找出本次工具调用中 generate_plan 对应的 ToolMessage
    tool_messages: List[ToolMessage] = result.get("messages", [])
    plan_json_from_tool = _extract_generate_plan_output(state, tool_messages)

    updates: Dict = {"messages": tool_messages}

    if plan_json_from_tool is not None:
        # 复用已有 session_id，或新建
        if state.planner_session is not None:
            session_id = state.planner_session.session_id
        else:
            session_id = f"plan_{uuid.uuid4().hex[:8]}"

        updates["planner_session"] = PlannerSession(
            session_id=session_id,
            plan_json=plan_json_from_tool,
        )
        logger.info("PlannerSession 已更新，session_id=%s", session_id)

    return updates


def _extract_generate_plan_output(
    state: State, tool_messages: List[ToolMessage]
) -> str | None:
    """从工具执行结果中提取 generate_plan 的输出。

    通过匹配 AIMessage 中的 tool_calls name 来定位对应的 ToolMessage。

    Returns:
        str: generate_plan 的返回内容；若本次未调用 generate_plan 则返回 None。
    """
    if not state.messages:
        return None

    last_ai = state.messages[-1]
    if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
        return None

    # 建立 tool_call_id → tool_name 的映射
    id_to_name: Dict[str, str] = {
        tc["id"]: tc["name"] for tc in last_ai.tool_calls if "id" in tc and "name" in tc
    }

    for tm in tool_messages:
        if not isinstance(tm, ToolMessage):
            continue
        if id_to_name.get(tm.tool_call_id) == "generate_plan":
            content = tm.content if isinstance(tm.content, str) else str(tm.content)
            if content.strip():
                return content.strip()

    return None


# ==================== 图定义 ====================

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node(call_model)
builder.add_node("tools", dynamic_tools_node)

builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """根据模型输出决定下一个节点。"""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"路由时期望 AIMessage，但收到 {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

graph = builder.compile(name="ReAct Agent")
