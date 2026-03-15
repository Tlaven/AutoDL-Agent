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

    - generate_plan 执行后：将新 plan_json 写入 planner_session
    - execute_plan 执行后：将 updated_plan_json（带执行状态）写回 planner_session
    """
    available_tools = await get_tools()
    tool_node = ToolNode(available_tools)
    result = await tool_node.ainvoke(state)

    tool_messages: List[ToolMessage] = result.get("messages", [])
    id_to_name = _build_id_to_name(state)

    updates: Dict = {"messages": tool_messages}

    for tm in tool_messages:
        if not isinstance(tm, ToolMessage):
            continue
        tool_name = id_to_name.get(tm.tool_call_id, "")
        content = tm.content if isinstance(tm.content, str) else str(tm.content)

        if tool_name == "generate_plan" and content.strip():
            session_id = (
                state.planner_session.session_id
                if state.planner_session is not None
                else f"plan_{uuid.uuid4().hex[:8]}"
            )
            updates["planner_session"] = PlannerSession(
                session_id=session_id,
                plan_json=content.strip(),
            )
            logger.info("PlannerSession 已更新（generate_plan），session_id=%s", session_id)

        elif tool_name == "execute_plan":
            updated_plan = _extract_updated_plan_from_executor(content)
            exec_status, exec_error = _extract_executor_status(content)
            if state.planner_session is not None:
                updates["planner_session"] = PlannerSession(
                    session_id=state.planner_session.session_id,
                    plan_json=updated_plan if updated_plan else state.planner_session.plan_json,
                    last_executor_status=exec_status,
                    last_executor_error=exec_error,
                )
                logger.info(
                    "PlannerSession 已更新（execute_plan 回填），session_id=%s，status=%s",
                    state.planner_session.session_id,
                    exec_status,
                )

    return updates


def _build_id_to_name(state: State) -> Dict[str, str]:
    """从最后一条 AIMessage 中构建 tool_call_id → tool_name 的映射。"""
    if not state.messages:
        return {}
    last_ai = state.messages[-1]
    if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
        return {}
    return {
        tc["id"]: tc["name"]
        for tc in last_ai.tool_calls
        if "id" in tc and "name" in tc
    }


def _extract_updated_plan_from_executor(content: str) -> str | None:
    """从 execute_plan 返回的 ToolMessage 内容中提取 updated_plan_json。

    约定格式：内容末尾有一行 `[EXECUTOR_RESULT] {...json...}`
    """
    import json as _json
    import re as _re
    match = _re.search(r'\[EXECUTOR_RESULT\]\s*(\{.*\})', content, _re.DOTALL)
    if not match:
        return None
    try:
        meta = _json.loads(match.group(1))
        updated = meta.get("updated_plan_json", "")
        return updated if updated else None
    except _json.JSONDecodeError:
        logger.warning("execute_plan 返回的 EXECUTOR_RESULT 解析失败")
        return None


def _extract_executor_status(content: str) -> tuple[str | None, str | None]:
    """从 execute_plan 返回的 ToolMessage 内容中提取 status 和 error_detail。

    返回 (status, error_detail)，解析失败时返回 (None, None)。
    """
    import json as _json
    import re as _re
    match = _re.search(r'\[EXECUTOR_RESULT\]\s*(\{.*\})', content, _re.DOTALL)
    if not match:
        return None, None
    try:
        meta = _json.loads(match.group(1))
        return meta.get("status"), meta.get("error_detail")
    except _json.JSONDecodeError:
        return None, None


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
