# planner_agent/graph.py
"""Planner Agent - 使用自定义 StateGraph 实现（高度灵活版）

核心特点：
1. PLANNER_RULES 严格放在消息列表最底部（符合你的要求）
2. 强制输出干净 JSON（无任何多余文字）
3. 单次调用即可完成规划（后续可轻松改成多轮）
4. 与项目整体风格完全一致（StateGraph + async）
"""

from datetime import UTC, datetime
from typing import List, Dict, Any
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from common.utils import load_chat_model
from common.prompts import SYSTEM_PROMPT
from .state import PlannerState
from .prompts import PLANNER_SYSTEM_PROMPT

# MemorySaver 全局实例（或在 compile 时创建）
checkpointer = MemorySaver()

async def call_planner(state: PlannerState) -> Dict[str, List[BaseMessage]]:
    """Planner 核心节点：把规则放在最后 + 强制 JSON 输出"""
    model = load_chat_model("siliconflow:Pro/deepseek-ai/DeepSeek-V3.2")

    messages = state.messages[:-1].copy()

    if SYSTEM_PROMPT.strip():
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    messages.append(
        SystemMessage(
            content=PLANNER_SYSTEM_PROMPT
        )
    )

    # 调用模型（不绑定任何工具，Planner 只负责输出 JSON 文本）
    response = await model.ainvoke(messages)

    content = response.content.strip() if isinstance(response.content, str) else ""

    # 防御：若 LLM 误将输出写入 tool_calls 而非 content，提前报错而非静默返回空
    if not content:
        raise RuntimeError(
            "Planner 模型未返回文本内容。"
            "请检查提示词是否触发了意外的工具调用模式。"
        )

    return {
        "messages": [AIMessage(content=content, name="planner")]
    }


# ==================== 构建 Graph ====================
builder = StateGraph(PlannerState)

builder.add_node("call_planner", call_planner)

builder.add_edge(START, "call_planner")
builder.add_edge("call_planner", END)

# 编译成可执行的 graph
planner_graph = builder.compile(name="Planner Agent")


# ==================== 对外暴露的运行函数 ====================
async def run_planner(
    messages: List[BaseMessage],
    thread_id: str,   # 必须传入，通常来自 supervisor 的 planner_session.session_id
    config: Dict | None = None
) -> str:
    """
    运行 planner，返回生成的 JSON 计划字符串
    
    Args:
        messages: 当前输入的消息列表（可以是 supervisor 传来的全部历史）
        thread_id: planner 会话 ID（与 supervisor.planner_session.session_id 对应）
        config: 可选的额外配置（通常不需要手动传）
    
    Returns:
        str: 干净的 JSON 字符串
    """
    if config is None:
        config = {"configurable": {"thread_id": thread_id}}

    input_state = {"messages": messages}
    result = await planner_graph.ainvoke(
            input_state,
            config=config
        )
    
    final_message = result["messages"][-1]
    content = final_message.content.strip()
    
    matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', content, re.IGNORECASE | re.DOTALL)
    # 检查匹配数量
    if len(matches) == 0 or len(matches) > 1:
        return content
    else:
        return matches[0].strip()
